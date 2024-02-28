from typing import Iterable, Optional

import torch
from torch import nn

from pointbev.utils.debug import debug_hook


class AlignRes(nn.Module):
    """Align resolutions of the outputs of the backbone."""

    def __init__(
        self,
        mode="upsample",
        scale_factors: Iterable[int] = [1, 2],
        in_channels: Iterable[int] = [56, 160],
    ):
        super().__init__()

        self.layers = nn.ModuleList()
        if mode == "upsample":
            for s in scale_factors:
                if s != 1:
                    self.layers.append(
                        nn.Upsample(
                            scale_factor=s, mode="bilinear", align_corners=False
                        )
                    )
                else:
                    self.layers.append(nn.Identity())

        elif mode == "conv2dtranspose":
            for i, in_c in enumerate(in_channels):
                if scale_factors[i] != 1:
                    self.layers.append(
                        nn.ConvTranspose2d(
                            in_c, in_c, kernel_size=2, stride=2, padding=0
                        )
                    )
                else:
                    self.layers.append(nn.Identity())

        else:
            raise NotImplementedError
        return

    def forward(self, x):
        return [self.layers[i](xi) for i, xi in enumerate(x.values())]


class PrepareChannel(nn.Module):
    """Transform the feature map to align with Network."""

    def __init__(
        self,
        in_channels=[56, 160],
        interm_c=128,
        out_c: Optional[int] = 128,
        mode="doubleconv",
        tail_mode="identity",
    ):
        super().__init__()
        assert mode in ["simpleconv", "doubleconv", "doubleconv_w_depth_layer"]
        assert tail_mode in ["identity", "conv2d"]

        in_c = sum(in_channels)
        if "simpleconv" in mode:
            self.layers = nn.Sequential(
                nn.Conv2d(in_c, interm_c, kernel_size=1, bias=False),
                nn.BatchNorm2d(interm_c),
            )

        elif "doubleconv" in mode:
            # Used in SimpleBEV
            self.layers = nn.Sequential(
                nn.Conv2d(in_c, interm_c, kernel_size=3, padding=1, bias=False),
                nn.InstanceNorm2d(interm_c),
                nn.ReLU(inplace=True),
                nn.Conv2d(interm_c, interm_c, kernel_size=3, padding=1, bias=False),
                nn.InstanceNorm2d(interm_c),
                nn.ReLU(inplace=True),
            )

        if tail_mode == "identity":
            self.tail = nn.Identity()
            self.out_c = interm_c
        elif tail_mode == "conv2d":
            # Used in SimpleBEV
            self.tail = nn.Conv2d(interm_c, out_c, kernel_size=1, padding=0)
            self.out_c = out_c

        return

    def forward(self, x):
        return self.tail(self.layers(x))


class AGPNeck(nn.Module):
    """
    Upsample outputs of the backbones, group them and align them to be compatible with Network.

    Note: mimics UpsamplingConcat in SimpleBEV.
    """

    def __init__(
        self,
        align_res_layer,
        prepare_c_layer,
        group_method=lambda x: torch.cat(x, dim=1),
    ):
        """
        Args:
            - align_res_layer: upsample layers at different resolution to the same.
            - group_method: how to gather the upsampled layers.
            - prepare_c_layer: change the channels of the upsampled layers in order to align with the network.
        """
        super().__init__()
        self.register_forward_hook(debug_hook)

        self.align_res_layer = align_res_layer
        self.group_method = group_method
        self.prepare_c_layer = prepare_c_layer
        self.out_c = prepare_c_layer.out_c
        return

    def forward(self, x: Iterable[torch.Tensor]):
        # Align resolution of inputs.
        x = self.align_res_layer(x)

        # Group inputs.
        x = self.group_method(x)

        # Change channels of final input.
        x = self.prepare_c_layer(x)
        assert x.shape[1] == self.out_c
        return x
