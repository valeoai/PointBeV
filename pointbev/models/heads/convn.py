from copy import deepcopy
from typing import List

import spconv.pytorch as spconv
from torch import nn
from torch.nn import functional as F

from pointbev.models.layers.common import ConvNormAct, SubMConvNormAct
from pointbev.utils.debug import debug_hook

algo = spconv.ConvAlgo.Native


class BEVConvHead(nn.Module):
    def __init__(
        self,
        shared_out_c: int = 64,
        # Outputs
        with_centr_offs: bool = False,
        with_hdmap: bool = False,
        hdmap_names: List[str] = [],
        with_binimg: bool = True,
        # Expect either dense or sparse input.
        dense_input: bool = True,
    ):
        super().__init__()

        self.register_forward_hook(debug_hook)
        dict_input = dict(
            with_binimg=with_binimg,
            with_centr_offs=with_centr_offs,
            with_hdmap=with_hdmap,
        )
        self.with_centr_offs = with_centr_offs
        self.dense_input = dense_input
        self.hdmap_names = hdmap_names
        self._get_layers(dict_input, shared_out_c, self.dense_input)

    def _get_layers(self, dict_input, shared_out_c, dense_input):
        # Unpack
        (with_binimg, with_centr_offs, with_hdmap) = (
            dict_input["with_binimg"],
            dict_input["with_centr_offs"],
            dict_input["with_hdmap"],
        )

        # Prepare out
        map_out = nn.ModuleDict()

        # Freezed arguments.
        activ = nn.ReLU

        # Initialize layers.
        if dense_input:
            norm = nn.InstanceNorm2d
            conv_c1 = nn.Conv2d(shared_out_c, out_channels=1, kernel_size=1, padding=0)
            convnormact_conv_c1 = nn.Sequential(
                ConvNormAct(shared_out_c, shared_out_c, 3, 1, False, norm, activ),
                deepcopy(conv_c1),
            )
            convnormact_conv_c2 = nn.Sequential(
                ConvNormAct(shared_out_c, shared_out_c, 3, 1, False, norm, activ),
                nn.Conv2d(shared_out_c, out_channels=2, kernel_size=1, padding=0),
            )
            if with_hdmap:
                convnormact_conv_chdmap = nn.Sequential(
                    ConvNormAct(shared_out_c, shared_out_c, 3, 1, False, norm, activ),
                    nn.Conv2d(
                        shared_out_c, len(self.hdmap_names), kernel_size=1, padding=0
                    ),
                )
        else:
            norm = nn.InstanceNorm1d
            conv_c1 = spconv.SubMConv2d(
                shared_out_c, 1, kernel_size=1, padding=0, algo=algo
            )
            convnormact_conv_c1 = spconv.SparseSequential(
                SubMConvNormAct(shared_out_c, shared_out_c, 3, 1, False, norm, activ),
                deepcopy(conv_c1),
            )
            convnormact_conv_c2 = spconv.SparseSequential(
                SubMConvNormAct(shared_out_c, shared_out_c, 3, 1, False, norm, activ),
                spconv.SubMConv2d(
                    shared_out_c, out_channels=2, kernel_size=1, padding=0, algo=algo
                ),
            )
            if with_hdmap:
                convnormact_conv_chdmap = spconv.SparseSequential(
                    SubMConvNormAct(
                        shared_out_c, shared_out_c, 3, 1, False, norm, activ
                    ),
                    spconv.SubMConv2d(
                        shared_out_c,
                        out_channels=len(self.hdmap_names),
                        kernel_size=1,
                        padding=0,
                        algo=algo,
                    ),
                )

        # Initialize heads.
        if with_binimg:
            map_out.update({"binimg": deepcopy(convnormact_conv_c1)})

        if with_centr_offs:
            map_out.update(
                {
                    "offsets": deepcopy(convnormact_conv_c2),
                    "centerness": deepcopy(convnormact_conv_c1),
                }
            )

        if with_hdmap:
            map_out.update({"hdmap": convnormact_conv_chdmap})
        self.map_out = map_out

    def forward_layers(self, x):
        out_dict = {}
        out_dict.update({k: (layer(x)) for k, layer in self.map_out.items()})
        return out_dict

    def _apply_final_activation(self, out_dict):
        """Since spconv can not apply sigmoid to sparse tensor, we do it here."""
        if self.dense_input:
            apply_activ_ = lambda x, func: func(x)
        else:
            apply_activ_ = lambda x, func: x.replace_feature(func(x.features))

        if self.with_centr_offs:
            feats = out_dict["centerness"]
            out_dict["centerness"] = apply_activ_(feats, lambda x: F.sigmoid(x))
        return out_dict

    def forward(self, x):
        out_dict = self.forward_layers(x)
        out_dict = self._apply_final_activation(out_dict)
        return out_dict
