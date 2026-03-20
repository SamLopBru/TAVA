import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights

class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling"""
    def __init__(self, in_channels, out_channels, atrous_rates):
        super(ASPP, self).__init__()
        modules = []
        
        # 1x1 Conv
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ))

        # Atrous Convs
        for rate in atrous_rates:
            modules.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=rate, dilation=rate, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ))

        # Image Pooling
        self.image_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.convs = nn.ModuleList(modules)
        
        # Project output of concatenated feature maps
        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels + out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        
        pool = self.image_pool(x)
        pool = F.interpolate(pool, size=x.shape[2:], mode='bilinear', align_corners=False)
        res.append(pool)
        
        res = torch.cat(res, dim=1)
        return self.project(res)


class DeepLabV3Plus(nn.Module):
    """
    DeepLabV3+ architecture with ResNet-50 backbone.
    """
    def __init__(self, num_classes=1):
        super(DeepLabV3Plus, self).__init__()
        
        # Load Pretrained ResNet50 Backbone
        resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        
        # Modify strides in deeper layers for output stride 16 or 8 depending on rates
        # Here we use output stride = 16 (often used with 6,12,18 rates)
        resnet.layer4[0].conv2.stride = (1, 1)
        resnet.layer4[0].downsample[0].stride = (1, 1)
        
        # Modify first layer to accept 1 channel (grayscale) instead of 3 (RGB)
        # Summing the pretrained weights over the channel dimension is standard practice
        old_conv1 = resnet.conv1
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        with torch.no_grad():
            resnet.conv1.weight.copy_(old_conv1.weight.sum(dim=1, keepdim=True))

        # Modify dilation in layer 4 to maintain spatial resolution without losing receptive field
        for m in resnet.layer4.modules():
            if isinstance(m, nn.Conv2d) and m.kernel_size == (3, 3):
                m.padding = (2, 2)
                m.dilation = (2, 2)
                
        # Low-level features extracting from layer1
        self.low_level_features = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1
        )
        
        # High-level features extracting from the rest of the network
        self.high_level_features = nn.Sequential(
            resnet.layer2,
            resnet.layer3,
            resnet.layer4
        )
        
        # ASPP with standard rates: 6, 12, 18
        self.aspp = ASPP(in_channels=2048, out_channels=256, atrous_rates=[6, 12, 18])
        
        # Decoder
        self.reduce_low_level = nn.Sequential(
            nn.Conv2d(256, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )
        
        self.decoder = nn.Sequential(
            nn.Conv2d(256 + 48, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(256, num_classes, 1)
        )
        
    def forward(self, x):
        h, w = x.shape[2:]
        
        # Encoder
        low_level_feat = self.low_level_features(x)
        high_level_feat = self.high_level_features(low_level_feat)
        
        # ASPP
        aspp_feat = self.aspp(high_level_feat)
        
        # Upsample ASPP features to match low-level features shape
        aspp_feat = F.interpolate(aspp_feat, size=low_level_feat.shape[2:], mode='bilinear', align_corners=False)
        
        # Reduce low-level channels
        low_level_feat = self.reduce_low_level(low_level_feat)
        
        # Concatenate
        feat = torch.cat([aspp_feat, low_level_feat], dim=1)
        
        # Decode to prediction
        out = self.decoder(feat)
        
        # Final upsample to original image size
        out = F.interpolate(out, size=(h, w), mode='bilinear', align_corners=False)
        return out
