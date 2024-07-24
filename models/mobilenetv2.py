"""
Paper:      MobileNetV2: Inverted Residuals and Linear Bottlenecks
Url:        https://arxiv.org/abs/1801.04381
Create by:  zh320
Date:       2024/07/13
"""

import torch.nn as nn
from torchvision.models import mobilenet_v2


class Mobilenetv2(nn.Module):
    def __init__(self, num_class, pretrained=True, downsample_rate=32):
        super(Mobilenetv2, self).__init__()
        self.model = mobilenet_v2(pretrained=pretrained)
        self.model.classifier = nn.Sequential(
                                        nn.Dropout(p=0.2),
                                        nn.Linear(self.model.last_channel, num_class),
                                        )

        if downsample_rate != 32:
            if downsample_rate in [8, 16]:
                self.modify_downsample_rate(self.model.features[0], 1)
                if downsample_rate == 8:
                    self.modify_downsample_rate(self.model.features[2], 1)
            else:
                raise NotImplementedError

    def modify_downsample_rate(self, layer, stride):
        for module in layer.modules():
            if isinstance(module, nn.Conv2d) and module.stride[0] != 1:
                module.stride = stride

    def forward(self, x):
        x = self.model(x)

        return x
