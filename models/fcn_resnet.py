import torch
from collections import OrderedDict
import torch.nn as nn
from typing import Dict
import torch.nn.functional as F
from torchvision.models import resnet50
from torchvision.models._utils import IntermediateLayerGetter


class FCNHead(nn.Sequential):
    def __init__(self, in_channels, channels):
        inter_channels = in_channels // 4
        layers = [
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(inter_channels, channels, 1)
        ]

        super(FCNHead, self).__init__(*layers)


class FCN(nn.Module):
    def __init__(self, backbone, classifier):
        super(FCN, self).__init__()
        self.backbone = backbone
        self.classifier = classifier

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        input_shape = x.shape[-2:]
        features = self.backbone(x)
        x = self.classifier(features[-1])
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        result = OrderedDict()
        result["out"] = x
        return result


class classifier(nn.Module):
    def __init__(self, in_channels, out_channels, expansion=1):
        super(classifier, self).__init__()
        self.expansion = expansion
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, self.expansion * 1024, 3, padding=1, bias=False),
            nn.BatchNorm2d(self.expansion * 1024),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(self.expansion * 2048, self.expansion * 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(self.expansion * 512),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(self.expansion * 1024, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.conv4 = nn.Conv2d(256, out_channels, 1)

    def forward(self, x, aux1, aux2, input_shape):
        x = self.conv1(x)
        aux1_shape = aux1.shape[-2:]
        x = F.interpolate(x, size=aux1_shape, mode='bilinear', align_corners=False)
        x = torch.cat((x, aux1), dim=1)
        x = self.conv2(x)
        aux2_shape = aux2.shape[-2:]
        x = F.interpolate(x, size=aux2_shape, mode='bilinear', align_corners=False)
        x = torch.cat((x, aux2), dim=1)
        x = self.conv3(x)
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        x = self.conv4(x)
        return x


class fcn_resnet50(nn.Module):
    def __init__(self):
        super(fcn_resnet50, self).__init__()
        backbone = resnet50(pretrained=True)
        return_layers = {'layer4': 'out'}
        self.backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    def forward(self, x):
        if x.shape[1] == 1:
            x = torch.cat([x, x, x], dim=1)
        features = self.backbone(x)
        x = features['out']  # Shape: (batch_size, 2048, H', W')
        return x  # Return per-pixel embeddings




