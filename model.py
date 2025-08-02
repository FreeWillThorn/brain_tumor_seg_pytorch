
import torch
import torch.nn as nn
from torchvision.models import resnet50, swin_transformer, swin_v2_b, ResNet50_Weights, Swin_V2_B_Weights, \
    convnext_large, ConvNeXt_Large_Weights, ResNet152_Weights, resnet101
from torchvision.ops import SqueezeExcitation
#from torchinfo import summary
import torch.nn.functional as F


class MultiScaleExtraction(nn.Module):
    def __init__(self, inchannels, outchannels):
        super(MultiScaleExtraction, self).__init__()
        self.conv3 = nn.Conv2d(inchannels, inchannels, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(inchannels, inchannels, kernel_size=5, padding=2)
        self.conv7 = nn.Conv2d(inchannels, inchannels, kernel_size=7, padding=3)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gap_conv = nn.Conv2d(inchannels, inchannels, kernel_size=1)

        self.attention_mechanism = SqueezeExcitation(inchannels * 4, inchannels * 4)  # SqueezeExcitation block
        #self.attention_mechanism = CBAM(inchannels * 4,r=8)  # CBAM block
        self.channel_reducer = nn.Conv2d(inchannels * 4, outchannels, kernel_size=1, padding=0)  # 4 feature maps concatenated
        self.dropout = nn.Dropout2d(p=0.2)


    def forward(self, x):
        high = self.conv3(x)
        mid = self.conv5(x)
        low = self.conv7(x)
        original = x

        # Concatenate along the channel dimension
        fused = self.attention_mechanism(torch.cat([high, mid, low, original], dim=1))  # fused shape [B, 256*3, H, W]
        fused = self.dropout(fused)
        fused = self.channel_reducer(fused)
        return fused

class MultiScaleAdaptiveFusion(nn.Module):
    def __init__(self, inchannels_list):
        super(MultiScaleAdaptiveFusion, self).__init__()
        # Neural network layers (like nn.Conv2d) are stateful and their parameters (weights, biases)
        # need to be initialized during model creation, not during forward pass
        self.channel_reducers = nn.ModuleList()
        self.attention_mechanism = nn.ModuleList()
        self.dropout = nn.Dropout2d(p=0.2)

        for i in range(len(inchannels_list) - 2, -1, -1):  # 2,1,0
            if i == 2:
                in_channel= inchannels_list[i] + inchannels_list[i + 1] # 2048 + 1024 = 3072
            else:
                in_channel = inchannels_list[i] + out_channel # 512 + 1536 = 2048, 256 + 1024 = 1280
            out_channel = in_channel // 2  #  3072// 2 = 1536, 2048 // 2 = 1024, 1280 // 2 = 640
            # in_channels = [3072,2048,1280] # inchannels_list
            # out_channels = [1536,1024,640] # outchannels_list
            self.channel_reducers.append(nn.Conv2d(in_channel, out_channel, kernel_size=1, padding=0))
            self.attention_mechanism.append(SqueezeExcitation(out_channel, out_channel))  # SqueezeExcitation block
            #self.attention_mechanism.append(CBAM(out_channel, r=8))  # CBAM block
        self.final_out_channel = out_channel


    def forward(self, x):
        # x is a list of feature maps from different scales
        final_feat = x[-1]  # The last feature map feat_4 that gained from MultiScaleExtraction

        for idx, feat_i in enumerate(range(len(x)-2, -1,-1)):  # ( SE & Conv1 , feature 3) :(0,2),(1,1),(2,0)

            final_feat = F.interpolate(final_feat, size=x[feat_i].shape[-2:], mode='bilinear', align_corners=False)  # Upsample high resolution feature map to lower
            Combined_feat = torch.cat((final_feat, x[feat_i]), dim=1)
            behalf_Combined_feat = self.channel_reducers[idx](Combined_feat)  # Apply 1x1 convolution to combine features
            behalf_Combined_feat = self.dropout(behalf_Combined_feat)
            final_feat = self.attention_mechanism[idx](behalf_Combined_feat)
            #final_feat = torch.cat((adjusted_feat, x[feat_i]), dim=1)

        return final_feat

class ConvNeXtBackbone(nn.Module):
    def __init__(self, weights='IMAGENET1K_V1'):
        super(ConvNeXtBackbone, self).__init__()
        model = convnext_large(weights=weights)

        self.stage0 = model.features[0]               # Conv2d(3, 192, kernel_size=4, stride=4)
        self.stage1 = model.features[1]               # CNBlocks
        self.down1 = model.features[2]                # Downsample 192 → 384
        self.stage2 = model.features[3]               # CNBlocks
        self.down2 = model.features[4]                # Downsample 384 → 768
        self.stage3 = model.features[5]               # CNBlocks
        self.down3 = model.features[6]                # Downsample 768 → 1536
        self.stage4 = model.features[7]               # CNBlocks

    def forward(self, x):
        x = self.stage0(x)
        x = self.stage1(x)
        feat1 = x                                      # [B, 192, H/4, W/4]

        x = self.down1(x)
        x = self.stage2(x)
        feat2 = x                                      # [B, 384, H/8, W/8]

        x = self.down2(x)
        x = self.stage3(x)
        feat3 = x                                      # [B, 768, H/16, W/16]

        x = self.down3(x)
        x = self.stage4(x)
        feat4 = x                                      # [B, 1536, H/32, W/32]

        return [feat1, feat2, feat3, feat4]


class Model(nn.Module):
    def __init__(self, num_classes, weights=None):
        super(Model, self).__init__()
        #self.backbone = ResnetBackbone()
        self.backbone = ConvNeXtBackbone()
        #self.msaf = MultiScaleAdaptiveFusion(inchannels_list=[256, 512, 1024, 2048])
        #self.mse = MultiScaleExtraction(2048,2048)
        self.msaf = MultiScaleAdaptiveFusion(inchannels_list=[192, 384, 768, 1536])
        self.mse = MultiScaleExtraction(1536, 1536)
        final_out_channels = self.msaf.final_out_channel
        self.classifier = nn.Conv2d(final_out_channels, num_classes, kernel_size=1)
        self.log_var_head = nn.Conv2d(final_out_channels, 1, kernel_size=1)
        self.seg_head = nn.Sequential(
            nn.Conv2d(final_out_channels, final_out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(final_out_channels),
            nn.ReLU(),
            nn.Conv2d(final_out_channels, num_classes, kernel_size=1),
        )

    def forward(self, x):
        features = self.backbone(x)
        features[3] = self.mse(features[3])
        fused_features = self.msaf(features)
        #fused_features = self.dropout(fused_features)
        seg_logits = self.seg_head(fused_features)
        var_logits = self.log_var_head(fused_features)
        upsampled_logits = F.interpolate(seg_logits, size=640, mode='bilinear', align_corners=False)
        upsampled_var_logits = F.interpolate(var_logits, size=640, mode='bilinear', align_corners=False)
        return upsampled_logits, upsampled_var_logits  # Return both segmentation logits and log variance for uncertainty estimation



# #
# # #
# if __name__ == "__main__":
#     #model = Model(num_classes=8)
#     model = Model(8)
#
#     #model = Resnet_backbone(num_classes=8, weights=None)
#     #summary(model, (1,3, 640, 640))  # Print model summary for input size (3, 640, 640)
#
#     input = torch.randn(1, 3, 640, 640)  # Example input tensor
#     output = model(input)  # Forward pass
#     #print("Output shape:", output.shape)
#     print(model)