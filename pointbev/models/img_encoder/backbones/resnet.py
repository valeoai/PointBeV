"""ResNet101 and ResNet50 used in SimpleBEV."""
from pathlib import Path

import torch
import torchvision
from torch import nn

from pointbev.models.img_encoder.backbones.common import Backbone


class Encoder_res101(Backbone):
    def __init__(
        self,
        checkpoint_path=None,
        downsample: int = 8,
        pth_path="resnet101-63fe2227.pth",
        checkpointing=False,
    ):
        super().__init__()
        resnet = torchvision.models.resnet101()
        resnet.load_state_dict(torch.load(Path(checkpoint_path) / pth_path))
        self.backbone = nn.Sequential(*list(resnet.children())[:-4])
        self.layer3 = resnet.layer3
        self.downsample = downsample
        self.checkpointing = checkpointing

    def forward(self, x, return_all=False):
        x1 = x
        for layer in self.backbone:
            if self.training and self.checkpointing:
                x1 = torch.utils.checkpoint.checkpoint(layer, x1)
            else:
                x1 = layer(x1)

        if self.training and self.checkpointing:
            x2 = torch.utils.checkpoint.checkpoint(self.layer3, x1)
        else:
            x2 = self.layer3(x1)

        return {f"out{i}": o for i, o in enumerate([x1, x2])}


class Encoder_res50(Backbone):
    def __init__(
        self,
        checkpoint_path=None,
        downsample: int = 8,
        pth_path="resnet50-0676ba61.pth",
        checkpointing=False,
    ):
        super().__init__()
        resnet = torchvision.models.resnet50()
        resnet.load_state_dict(torch.load(Path(checkpoint_path) / pth_path))
        self.backbone = nn.Sequential(*list(resnet.children())[:-4])
        self.layer3 = resnet.layer3
        self.downsample = downsample
        self.checkpointing = checkpointing

    def forward(self, x, return_all=False):
        x1 = x
        for layer in self.backbone:
            if self.training and self.checkpointing:
                x1 = torch.utils.checkpoint.checkpoint(layer, x1)
            else:
                x1 = layer(x1)

        if self.training and self.checkpointing:
            x2 = torch.utils.checkpoint.checkpoint(self.layer3, x1)
        else:
            x2 = self.layer3(x1)

        return {f"out{i}": o for i, o in enumerate([x1, x2])}
