import torch
import torch.nn as nn
import torchvision.models.segmentation as segmentation
from tensorboard import summary
from torch import ops
from torchvision.models import resnet50, swin_transformer, swin_v2_b, ResNet50_Weights, Swin_V2_B_Weights
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.ops import SqueezeExcitation
from torchinfo import summary
import torch.nn.functional as F

class MultiScaleExtraction(nn.Module):
    def __init__(self, inchannels, outchannels):
        super(MultiScaleExtraction, self).__init__()
        self.conv3 = nn.Conv2d(inchannels, inchannels, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(inchannels, inchannels, kernel_size=5, padding=2)
        self.conv7 = nn.Conv2d(inchannels, inchannels, kernel_size=7, padding=3)
        self.SE = SqueezeExcitation(inchannels*4, inchannels*4)  # SqueezeExcitation block
        self.channel_reducer = nn.Conv2d(inchannels * 4, outchannels, kernel_size=1, padding=0)  # 4 feature maps concatenated
        self.dropout = nn.Dropout2d(p=0.2)


    def forward(self, x):
        high = self.conv3(x)
        mid = self.conv5(x)
        low = self.conv7(x)
        original = x

        # Concatenate along the channel dimension
        fused = self.SE(torch.cat([high, mid, low,original], dim=1))  # fused shape [B, 256*3, H, W]
        fused = self.dropout(fused)
        fused = self.channel_reducer(fused)
        return fused



class MultiScaleAdaptiveFusion(nn.Module):
    def __init__(self, inchannels_list):
        super(MultiScaleAdaptiveFusion, self).__init__()
        # Neural network layers (like nn.Conv2d) are stateful and their parameters (weights, biases)
        # need to be initialized during model creation, not during forward pass
        self.channel_reducers = nn.ModuleList()
        self.SE_blocks = nn.ModuleList()
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
            self.SE_blocks.append(SqueezeExcitation(out_channel,out_channel))  # SqueezeExcitation block


    def forward(self, x):
        # x is a list of feature maps from different scales
        final_feat = x[-1]  # The last feature map feat_4 that gained from MultiScaleExtraction

        for idx, feat_i in enumerate(range(len(x)-2, -1,-1)):  # ( SE & Conv1 , feature 3) :(0,2),(1,1),(2,0)

            final_feat = F.interpolate(final_feat, size=x[feat_i].shape[-2:], mode='bilinear', align_corners=False)  # Upsample high resolution feature map to lower
            Combined_feat = torch.cat((final_feat, x[feat_i]), dim=1)
            behalf_Combined_feat = self.channel_reducers[idx](Combined_feat)  # Apply 1x1 convolution to combine features
            behalf_Combined_feat = self.dropout(behalf_Combined_feat)
            final_feat = self.SE_blocks[idx](behalf_Combined_feat)
            #final_feat = torch.cat((adjusted_feat, x[feat_i]), dim=1)

        return final_feat

class ResnetBackbone(nn.Module):
    def __init__(self, weights=None):
        super(ResnetBackbone, self).__init__()

        # self.backbone = segmentation.deeplabv3_mobilenet_v3_large(weights=weights)
        # self.backbone.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))
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



class Model(nn.Module):
    def __init__(self, num_classes, weights=ResNet50_Weights.IMAGENET1K_V2):
        super(Model, self).__init__()
        self.backbone = ResnetBackbone(weights)
        self.msaf = MultiScaleAdaptiveFusion(inchannels_list=[256, 512, 1024, 2048])
        self.mse = MultiScaleExtraction(2048,2048)
        self.classifier = nn.Conv2d(640, num_classes, kernel_size=1)
        self.dropout = nn.Dropout2d(p=0.2)
        self.edge_head = nn.Conv2d(640, 1, kernel_size=1)  # Edge detection head

        #self.bn = nn.BatchNorm2d(num_classes)

    def forward(self, x):
        features = self.backbone(x)
        features[3] = self.mse(features[3])
        fused_features = self.msaf(features)
        #fused_features = self.dropout(fused_features)
        logits = self.classifier(fused_features)
        #logits = self.bn(logits)
        upsampled_logits = F.interpolate(logits, size=640, mode='bilinear', align_corners=False)
        #output = F.softmax(upsampled_logits, dim=1)  # Softmax over class dimension
        edge_logits = self.edge_head(fused_features)
        edge_logits = F.interpolate(edge_logits, size=640, mode='bilinear', align_corners=False)
        edge_logits = edge_logits.squeeze(1)
        return upsampled_logits, edge_logits  # Return both segmentation and edge detection outputs

#model = Model(num_classes=8, weights=None)
#summary(model, (1,3, 640, 640))  # Print model summary for input size (3, 640, 640)
#
#input = torch.randn(1, 3, 640, 640)  # Example input tensor
#output = model(input)  # Forward pass
#print("Output shape:", output.shape)
#print(model)


class Swin_backbone(nn.Module):
    def __init__(self, num_classes, weights= Swin_V2_B_Weights.IMAGENET1K_V1):
        super(Swin_backbone, self).__init__()
        self.backbone = swin_v2_b(weights=weights)

    def forward(self, x):
        x = self.backbone.patch_embed(x)
        x = self.backbone.pos_drop(x)
        x = self.backbone.layers(x)
        x = self.backbone.norm(x)
        return x

