from collections import OrderedDict
from pathlib import Path

from efficientnet_pytorch import EfficientNet as EfficientNet_extractor
from pytorch_lightning.utilities import rank_zero_only
from torch import nn

from pointbev.models.img_encoder.backbones.common import Backbone

CKPT_MAP = {"b0": "efficientnet-b0-355c32eb.pth", "b4": "efficientnet-b4-6ed6700e.pth"}


class EfficientNet(Backbone):
    def __init__(self, checkpoint_path=None, version="b4", downsample=8):
        super().__init__()
        self.version = version

        assert downsample == 8, "EfficientNet only supported for downsample=8"
        self.downsample = downsample
        self._init_efficientnet(checkpoint_path, version)

    def _init_efficientnet(self, weights_path, version):
        if weights_path is not None:
            weights_path = Path(weights_path) / CKPT_MAP[version]
            if not weights_path.exists():
                message = f"EfficientNet weights file does not exists at weights_path {weights_path}"
                weights_path = None
            else:
                message = (
                    f"EfficientNet exists and is loaded at weights_path {weights_path}"
                )
                weights_path = str(weights_path)
        else:
            message = "EfficientNet weights file not given, downloading..."

        trunk = EfficientNet_extractor.from_pretrained(
            f"efficientnet-{version}", weights_path=weights_path
        )

        self._conv_stem, self._bn0, self._swish = (
            trunk._conv_stem,
            trunk._bn0,
            trunk._swish,
        )
        self.drop_connect_rate = trunk._global_params.drop_connect_rate

        self._blocks = nn.ModuleList()
        for idx, block in enumerate(trunk._blocks):
            if version == "b0" and idx > 10 or version == "b4" and idx > 21:
                break
            self._blocks.append(block)

        del trunk
        self._print_loaded_file(message)

    @rank_zero_only
    def _print_loaded_file(self, message):
        print("# -------- Backbone -------- #")
        print(message, end="\n")

    def forward(self, x, return_all=False):
        endpoints = dict()

        # Stem
        x = self._swish(self._bn0(self._conv_stem(x)))
        prev_x = x

        # Blocks
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(
                    self._blocks
                )  # scale drop connect_rate
            x = block(x, drop_connect_rate=drop_connect_rate)
            if prev_x.size(2) > x.size(2):
                endpoints[f"reduction_{len(endpoints)+1}"] = prev_x
            prev_x = x

            if self.version == "b0" and idx == 10:
                break
            if self.version == "b4" and idx == 21:
                break

        # Head
        endpoints[f"reduction_{len(endpoints)+1}"] = x

        if not return_all:
            list_keys = ["reduction_3", "reduction_4"]
        else:
            list_keys = list(endpoints.keys())
        return OrderedDict({f"out{i}": endpoints[k] for i, k in enumerate(list_keys)})
