import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision.models import vit_b_16, ViT_B_16_Weights
from tqdm import tqdm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


lower_red1 = np.array([0, 100, 100])
upper_red1 = np.array([15, 255, 255])
lower_red2 = np.array([125, 100, 100])
upper_red2 = np.array([160, 255, 255])
lower_red3 = np.array([160, 100, 100])
upper_red3 = np.array([180, 255, 255])
# allocate blue areas
lower_blue = np.array([85, 100, 100])
upper_blue = np.array([130, 255, 255])
# remove grey dahsed lines
lower_gray = np.array([0, 0, 185])   
upper_gray = np.array([180, 30, 200])  
# remove purple lines
lower_purple = np.array([110, 50, 200])   
upper_purple = np.array([130, 100, 230])


class MultiStreamNetwork(nn.Module):
    def __init__(self, height, width, patch_size=16, embed_dim=256, num_heads=8, lstm_hidden_dim=512, lstm_layers=2):
        super(MultiStreamNetwork, self).__init__()
        self.height = height
        self.width = width
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.color_conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.color_attention_fc1 = nn.Linear(32, 32 // 8)
        self.color_attention_fc2 = nn.Linear(32 // 8, 32)
        self.color_conv3x3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.color_conv5x5 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.color_conv7x7 = nn.Conv2d(32, 64, kernel_size=7, padding=3)
        self.vit_conv = nn.Conv2d(192, 256, kernel_size=self.patch_size, stride=self.patch_size)
        self.transformer_layer = nn.MultiheadAttention(embed_dim=256, num_heads=self.num_heads)
        
        self.color_fc_final = nn.Linear(embed_dim, 512)

        self.lstm = nn.LSTM(32, lstm_hidden_dim, lstm_layers, batch_first=True)
        self.fc_final = nn.Linear(lstm_hidden_dim, 512)

        weights = ViT_B_16_Weights.IMAGENET1K_V1
        self.vit_model = vit_b_16(weights=weights)
        

        if isinstance(self.vit_model.heads, nn.Sequential):
            self.vit_model.heads[-1] = nn.Identity()  
        else:
            self.vit_model.heads = nn.Identity()


        self.multihead_attn = nn.MultiheadAttention(embed_dim=768, num_heads=self.num_heads)


        self.spatial_fc = nn.Linear(768, 512)


        # self.spatial_weight = nn.Parameter(torch.tensor(1.0))  


        self.all_color = 0.5
        self.red_weight = 0.25
        self.bin_weight = 0.25
        
        self.ori_color_weight = 0.4
        self.ori_time_weight = 0.3
        self.ori_spatial_weight = 0.3
        
        self.red_color_weight = 0.6
        self.red_time_weight = 0.2
        self.red_spatial_weight = 0.2

        self.bin_time_weight = 0.5
        self.bin_spatial_weight = 0.5

    def color_attention_module(self, x):
        batch_size, channels, _, _ = x.size()
        avg_pool = F.adaptive_avg_pool2d(x, 1).view(batch_size, channels)
        attention = F.relu(self.color_attention_fc1(avg_pool))
        attention = torch.sigmoid(self.color_attention_fc2(attention)).view(batch_size, channels, 1, 1)
        return x * attention.expand_as(x)

    def color_stream(self, ori_image): 
        x = F.relu(self.color_conv1(ori_image))

        x = self.color_attention_module(x)

        conv3x3 = F.relu(self.color_conv3x3(x))
        conv5x5 = F.relu(self.color_conv5x5(x))
        conv7x7 = F.relu(self.color_conv7x7(x))
        multi_scale_output = torch.cat([conv3x3, conv5x5, conv7x7], dim=1)

        patches = self.vit_conv(multi_scale_output)
        batch_size, channels, patch_h, patch_w = patches.size()
        patches = patches.view(batch_size, channels, -1).permute(2, 0, 1)
        vit_output, _ = self.transformer_layer(patches, patches, patches)

        color_output = self.color_fc_final(vit_output.mean(dim=0))

        return color_output

    def time_stream(self, x):
        cnn_out = F.relu(self.color_conv1(x))
        cnn_out = F.adaptive_avg_pool2d(cnn_out, (1, cnn_out.size(3)))
        cnn_out = cnn_out.squeeze(2)
        lstm_input = cnn_out.permute(0, 2, 1)
        lstm_out, _ = self.lstm(lstm_input)
        time_output = self.fc_final(lstm_out[:, -1, :])
        return time_output

    def spatial_stream(self, x):
        batch_size, channels, height, width = x.size()
                
        if channels == 1:
            proj_layer = nn.Conv2d(1, 768, kernel_size=16, stride=16)
        else:
            proj_layer = self.vit_model.conv_proj

        embedded_patches = proj_layer(x)  

        _, hidden_dim, num_patches_height, num_patches_width = embedded_patches.size()

        num_patches = num_patches_height * num_patches_width

        embedded_patches = embedded_patches.flatten(2).transpose(1, 2)  

        pos_embedding = self.vit_model.encoder.pos_embedding[:, :num_patches, :]
        pos_embedding = pos_embedding.expand(batch_size, -1, -1)

        encoded_patches = embedded_patches + pos_embedding

        layer_outputs = []
        for layer in self.vit_model.encoder.layers:
            layer_output = layer(encoded_patches)
            pooled_output = layer_output.mean(dim=1)
            layer_outputs.append(pooled_output)

        layer_outputs = torch.stack(layer_outputs)
        fused_features, _ = self.multihead_attn(layer_outputs, layer_outputs, layer_outputs)

        spatial_output = fused_features.mean(dim=0)

        spatial_output_512 = self.spatial_fc(spatial_output)
        return spatial_output_512


    def fusion_layer(self, initial_image, red_image, bin_image):
        
        fused_output = (self.all_color * initial_image + 
                        self.red_weight * red_image +
                        self.bin_weight * bin_image)
        return fused_output

    def forward(self, ori_image, red_image, bin_image): 
        ori_color_output = self.color_stream(ori_image)
        red_color_output = self.color_stream(red_image)
        
        ori_time_output = self.time_stream(ori_image)
        red_time_output = self.time_stream(red_image)
        bin_time_output = self.time_stream(bin_image)
        ori_spatial_output = self.spatial_stream(ori_image)
        red_spatial_output = self.spatial_stream(red_image)
        bin_spatial_output = self.spatial_stream(bin_image)
        
        all_color_output = self.ori_color_weight*ori_color_output + self.ori_time_weight*ori_time_output + self.ori_spatial_weight*ori_spatial_output   
        
        red_color_output = self.red_color_weight*red_color_output + self.red_time_weight*red_time_output + self.red_spatial_weight*red_spatial_output
        
        bin_color_output = self.bin_time_weight*bin_time_output + self.bin_spatial_weight*bin_spatial_output

        final_output = self.fusion_layer(all_color_output, red_color_output, bin_color_output)

        return final_output


def extract_color_features(img):
    hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    red_mask1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
    red_mask2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
    red_mask3 = cv2.inRange(hsv_image, lower_red3, upper_red3)
    red_mask = red_mask1 + red_mask2 + red_mask3
    blue_part_mask = cv2.inRange(hsv_image, lower_blue, upper_blue)
    gray_mask = cv2.inRange(hsv_image, lower_gray, upper_gray)
    purple_mask = cv2.inRange(hsv_image, lower_purple, upper_purple)
    blue_mask = blue_part_mask + gray_mask + purple_mask

    red_image = cv2.bitwise_and(img, img, mask=red_mask)
    bin_image = np.where(blue_mask == 0, 255, 0).astype(np.uint8)

    red_image_tensor = torch.from_numpy(red_image).float().permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
    bin_image_tensor = torch.from_numpy(bin_image).float().unsqueeze(0).unsqueeze(0).to(device) / 255.0
    bin_image_tensor = bin_image_tensor.repeat(1, 3, 1, 1)  

    return red_image_tensor, bin_image_tensor


multi_stream_model = MultiStreamNetwork(height=224, width=224).to(device)

multi_stream_model.eval()  

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
base_path = os.path.join(parent_dir, 'data/pic')

patient_folders = [f for f in os.listdir(base_path) if f.startswith('P')]

all_features = []
patient_ids = []
image_ids = []

x_start, x_end = 90, 800  
y_start, y_end = 20, 540  

for patient_folder in tqdm(patient_folders):
    patient_path = os.path.join(base_path, patient_folder)
    image_files = [f for f in os.listdir(patient_path) if f.endswith('.png')]

    for image_file in image_files:
        image_path = os.path.join(patient_path, image_file)
        
        img = cv2.imread(image_path)

        if img is None:
            print(f"Failed to load image: {image_path}")
            continue

        stretched_img = cv2.resize(img, (804, 564))  #(831, 544)

        cropped_img = stretched_img[y_start:y_end, x_start:x_end]
        resized_img = cv2.resize(cropped_img, (224, 224)) 
        image_tensor = torch.from_numpy(resized_img).float().permute(2, 0, 1).unsqueeze(0).to(device) / 255.0

        red_image_tensor, bin_image_tensor = extract_color_features(resized_img)

        with torch.no_grad():
            features = multi_stream_model(image_tensor, red_image_tensor, bin_image_tensor)
            flattened_features = features.view(-1).cpu().numpy()

        all_features.append(flattened_features)
        patient_ids.append(patient_folder)
        image_ids.append(image_file)






