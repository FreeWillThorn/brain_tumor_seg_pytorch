from pyexpat import features

import torch
import torch.nn as nn
import torchvision
import torchvision.models.segmentation as segmentation
from tensorboard import summary
from torch import ops
from torchvision.models import resnet50, swin_transformer, swin_v2_b, ResNet50_Weights, Swin_V2_B_Weights, \
    convnext_large, ConvNeXt_Large_Weights, ResNet152_Weights, resnet101
from torchvision.ops import SqueezeExcitation
#from torchinfo import summary
import torch.nn.functional as F

from src.CBAM import CBAM


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


class MultiScaleExtraction_paper(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MultiScaleExtraction_paper, self).__init__()

        # Part A: Global context path (GAP + 1x1 conv + upsample)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gap_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        # Part B: Multi-dilation atrous convolutions
        self.conv_d1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, dilation=1)
        self.conv_d2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=2, dilation=2)
        self.conv_d3 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=3, dilation=3)
        self.conv_d5 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=5, dilation=5)
        self.conv_d7 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=7, dilation=7)

        # Part C: Identity (original input)
        # No layer needed; use `x` directly

        # Channel fusion
        total_channels = in_channels * 7  # 6 conv outputs + 1 identity
        self.attention_mechanism = SqueezeExcitation(total_channels, total_channels)
        self.channel_reducer = nn.Conv2d(total_channels, out_channels, kernel_size=1)
        self.dropout = nn.Dropout2d(p=0.2)

    def forward(self, x):
        size = x.size()[2:]  # (H, W)

        # Part A
        gap_feat = self.gap(x)                        # [B, C, 1, 1]
        gap_feat = self.gap_conv(gap_feat)            # [B, C, 1, 1]
        gap_feat = F.interpolate(gap_feat, size=size, mode='bilinear', align_corners=False)

        # Part B
        d1 = self.conv_d1(x)
        d2 = self.conv_d2(x)
        d3 = self.conv_d3(x)
        d5 = self.conv_d5(x)
        d7 = self.conv_d7(x)

        # Part C: Original input
        identity = x

        # Concat + fuse
        fused = torch.cat([gap_feat, d1, d2, d3, d5, d7, identity], dim=1)
        fused = self.attention_mechanism(fused)
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

class MultiScaleAdaptiveFusion_160(nn.Module):
    def __init__(self, inchannels_list):
        super(MultiScaleAdaptiveFusion_160, self).__init__()
        # Neural network layers (like nn.Conv2d) are stateful and their parameters (weights, biases)
        # need to be initialized during model creation, not during forward pass
        self.channel_reducers = nn.ModuleList()
        self.attention_mechanism = nn.ModuleList()
        self.dropout = nn.Dropout2d(p=0.2)
        out_channel = 0
        for i in range(len(inchannels_list) - 2, -1, -1):  # 2,1,0

            if i == 2:
                in_channel= inchannels_list[i] + inchannels_list[i + 1]
            else:
                in_channel = inchannels_list[i] + out_channel

            out_channel = in_channel // 2
            # out channels = [1536,1024,640] # outchannels_list
            # resnet_in_channels = [3072,2048,1280] # inchannels_list
            # resnet_out_channels = [1536,1024,640] # outchannels_list
            #print(f"in_channel: {in_channel}, out_channel: {out_channel}")
            self.channel_reducers.append(nn.Conv2d(in_channel, out_channel, kernel_size=1, padding=0))
            self.attention_mechanism.append(SqueezeExcitation(out_channel, out_channel))  # SqueezeExcitation block
            #self.attention_mechanism.append(CBAM(out_channel, r=8))  # CBAM block
        self.final_out_channel = out_channel


    def forward(self, x):
        # x is a list of feature maps from different scales
        final_feat = x[-1]  # The last feature map feat_4 that gained from MultiScaleExtraction

        for idx, feat_i in enumerate(range(len(x)-2, -1,-1)):  # ( SE & Conv1 , feature) :(0,2),(1,1),(2,0)

            final_feat = F.interpolate(final_feat, size=x[feat_i].shape[-2:], mode='bilinear', align_corners=False)  # Upsample high resolution feature map to lower
            Combined_feat = torch.cat((final_feat, x[feat_i]), dim=1)
            behalf_Combined_feat = self.channel_reducers[idx](Combined_feat)  # Apply 1x1 convolution to combine features
            behalf_Combined_feat = self.dropout(behalf_Combined_feat)
            final_feat = self.attention_mechanism[idx](behalf_Combined_feat) #out_channel 1536, 1024, 640
            #final_feat = torch.cat((final_feat, x[feat_i]), dim=1) #

        return final_feat



class ResnetBackbone(nn.Module):
    def __init__(self, weights='DEFAULT'):
        super(ResnetBackbone, self).__init__()

        # self.backbone = segmentation.deeplabv3_mobilenet_v3_large(weights=weights)
        # self.backbone.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))
        self.backbone = resnet101(weights=weights)
        #self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])
        self.conv1 = self.backbone.conv1
        self.bn1 = self.backbone.bn1
        self.relu = self.backbone.relu
        self.maxpool = self.backbone.maxpool
        self.layer1 = self.backbone.layer1
        self.layer2 = self.backbone.layer2
        self.layer3 = self.backbone.layer3
        self.layer4 = self.backbone.layer4
        # below will be deleted since we are doing segmentation task
        #self.avgpool = self.backbone.avgpool
        #self.fc= self.backbone.fc


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        feat_1 = self.layer1(x) # shape [B, 256, H/4, W/4]
        feat_2 = self.layer2(feat_1) # shape [B, 512, H/8, W/8]
        feat_3 = self.layer3(feat_2) # shape [B, 1024, H/16, W/16]
        feat_4 = self.layer4(feat_3) # shape [B, 2048, H/32, W/32]
        return [feat_1,feat_2,feat_3,feat_4]  # return a list of feature maps from different layers



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

class MobileNetV3Backbone(nn.Module):
    def __init__(self, weights='DEFAULT'):
        super(MobileNetV3Backbone, self).__init__()
        self.backbone = torchvision.models.mobilenet_v3_large()
        # Modify the classifier to output the desired number of classes
        # self.backbone.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x):
        features = self.backbone.backbone(x)
        return [features['out'], features['aux']]  # Return the main output and auxiliary output

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

class Resnet_model_backbone_mse(nn.Module):
    def __init__(self, num_classes, weights=ResNet50_Weights.IMAGENET1K_V2):
        super(Resnet_model_backbone_mse, self).__init__()
        self.backbone = resnet50(weights=weights)
        #self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])
        self.conv1 = self.backbone.conv1
        self.bn1 = self.backbone.bn1
        self.relu = self.backbone.relu
        self.maxpool = self.backbone.maxpool
        self.layer1 = self.backbone.layer1
        self.layer2 = self.backbone.layer2
        self.layer3 = self.backbone.layer3
        self.layer4 = self.backbone.layer4
        self.mse = MultiScaleExtraction(2048, 2048)
        self.classifier = nn.Conv2d(2048, num_classes, kernel_size=1)
        # below will be deleted since we are doing segmentation task
        #self.avgpool = self.backbone.avgpool
        #self.fc= self.backbone.fc


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        feat_1 = self.layer1(x) # shape [B, 256, H/4, W/4]
        feat_2 = self.layer2(feat_1) # shape [B, 512, H/8, W/8]
        feat_3 = self.layer3(feat_2) # shape [B, 1024, H/16, W/16]
        feat_4 = self.layer4(feat_3) # shape [B, 2048, H/32, W/32]
        final_feat = self.mse(feat_4)
        logits = self.classifier(final_feat)
        upsampled_logits = F.interpolate(logits, size=640, mode='bilinear', align_corners=False)
        return upsampled_logits


class MultiScaleAdaptiveFusion_swin(nn.Module):
    def __init__(self, inchannels_list):
        super(MultiScaleAdaptiveFusion_swin, self).__init__()
        self.channel_reducers = nn.ModuleList()
        self.attention_mechanism = nn.ModuleList()
        self.dropout = nn.Dropout2d(p=0.2)

        out_channel = None
        for i in range(len(inchannels_list) - 2, -1, -1): # 2,1,0
            if i == 2: # i = 2
                in_channel = inchannels_list[i] + inchannels_list[i + 1]
            else:
                in_channel = inchannels_list[i] + out_channel
            out_channel = in_channel // 2

            self.channel_reducers.append(nn.Conv2d(in_channel, out_channel, kernel_size=1))
            #self.attention_mechanism.append(SqueezeExcitation(out_channel, out_channel))
            self.attention_mechanism.append(CBAM(out_channel, r=8))


    def forward(self, features):  # features = [x1, x2, x3, x4]
        x = features[-1] # Start with the last feature map (highest resolution)
        for idx, i in enumerate(range(len(features) - 2, -1, -1)): # (SE & Conv1 , feature 3) :(0,2),(1,1),(2,0)
            up = F.interpolate(x, size=features[i].shape[2:], mode='bilinear', align_corners=False)
            x = torch.cat([up, features[i]], dim=1)
            #print(x.shape)
            x = self.channel_reducers[idx](x) #
            x = self.attention_mechanism[idx](x)
            x = self.dropout(x)
        return x  # Final fused output


class Swin_backbone(nn.Module):
    def __init__(self, num_classes, weights=Swin_V2_B_Weights.IMAGENET1K_V1):
        super(Swin_backbone, self).__init__()
        self.backbone = swin_v2_b(weights=weights)
        self.feature1 = self.backbone.features[0] # Patch embedding         [B, 128, 160, 160]
        self.feature2 = self.backbone.features[1] # Stage 1 (Swin blocks)   [B, 128, 160, 160]
        self.feature3 = self.backbone.features[2] # Patch Merging           [B, 256, 80, 80]
        self.feature4 = self.backbone.features[3] # Stage 2 (Swin blocks)   [B, 256, 80, 80]
        self.feature5 = self.backbone.features[4] # Patch Merging           downsample
        self.feature6 = self.backbone.features[5] # Stage 3 (Swin blocks)   [B, 512, 40, 40]
        self.feature7 = self.backbone.features[6] # Patch Merging           downsample
        self.feature8 = self.backbone.features[7] # Stage 4 (Swin blocks)   [B, 1024, 20, 20]

        self.fusion = MultiScaleAdaptiveFusion_swin([128, 256, 512, 1024])
        self.mse = MultiScaleExtraction(320, 320)
        self.log_var_head = nn.Conv2d(320, 1, kernel_size=1)
        self.seg_head = nn.Sequential(
            nn.Conv2d(320, 320, kernel_size=3, padding=1),
            nn.BatchNorm2d(320),
            nn.ReLU(),
            nn.Conv2d(320, num_classes, kernel_size=1),
        )
        #self.edge_head = nn.Conv2d(320, 1, kernel_size=1)  # Edge detection head
        # self.edge_head = nn.Sequential(
        #     nn.Conv2d(320, 64, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(64, 1, kernel_size=1)
        # )

    def forward(self, x):
        feats = []
        x = self.feature1(x)
        x = self.feature2(x)
        feats.append(x.permute(0, 3, 1, 2).contiguous())
        x = self.feature3(x)
        x = self.feature4(x)
        feats.append(x.permute(0, 3, 1, 2).contiguous())
        x = self.feature5(x)
        x = self.feature6(x)
        feats.append(x.permute(0, 3, 1, 2).contiguous())
        x = self.feature7(x)
        x = self.feature8(x)
        feats.append(x.permute(0, 3, 1, 2).contiguous())
        #print([f.shape for f in feats])
        x = self.fusion(feats)    # [B, 320,160,160]
        x = self.mse(x)                     # multi-scale enhancement
        logits = self.seg_head(x)          # [B, num_classes, 80, 80]
        log_var = self.log_var_head(x)
        edge_logits = self.edge_head(x)
        upsampled_logits = F.interpolate(logits, size=(640, 640), mode='bilinear', align_corners=False)
        upsampled_log_var = F.interpolate(log_var, size=(640, 640), mode='bilinear', align_corners=False)
        return upsampled_logits, upsampled_log_var  # Return both segmentation logits and log variance for uncertainty estimation



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