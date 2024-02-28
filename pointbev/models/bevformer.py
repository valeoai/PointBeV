"""
Base model for BeVFormer reproduction.

Note: 
- BeVFormer is only static.
- It is inspired by SimpleBeV reproduction code.
"""

from typing import Dict, List, Optional, Tuple

import torch
from einops import rearrange
from torch import nn

from pointbev.models.common import Network
from pointbev.utils.debug import debug_hook


class QueryGenerator(nn.Module):
    """Interface for query generators.
    It generates the query we are looking for.
    """

    def __init__(self, query_shape, in_c, out_c):
        super().__init__()
        self._query_seq_len = query_shape
        self._output_c = out_c
        self.register_forward_hook(debug_hook)

    @property
    def query_shape(self):
        return self._query_seq_len

    @property
    def out_c(self):
        return self._output_c

    def forward(self, x) -> Dict:
        raise NotImplementedError()


class Latent2DQueryGenerator(QueryGenerator):
    def __init__(
        self,
        channels,
        bev_shape: Optional[Tuple[int, int]] = None,
        query_flat_shape: Optional[int] = None,
    ):
        if query_flat_shape is not None:
            query_shape = [query_flat_shape, 1]
        elif bev_shape is not None:
            query_shape = bev_shape
        else:
            raise ValueError("Either bev_shape or query_flat_shape must be provided")

        super().__init__(
            query_shape=query_shape,
            in_c=channels,
            out_c=channels,
        )
        self.query = nn.Parameter(0.1 * torch.randn(*query_shape, channels))

    def forward(self, bs: List[int]):
        h, w, c = self.query.shape
        query = self.query.view(*[1 for _ in range(len(bs))], h, w, c)
        query = query.repeat(*bs, 1, 1, 1)
        return query, (h, w)


class BevFormerNetwork(Network):
    def __init__(
        self,
        # Modules
        backbone=None,
        neck=None,
        embedding=None,
        query_gen=None,
        projector=None,
        view_transform=None,
        autoencoder=None,
        heads=None,
        in_c: Dict[str, int] = {},
        out_c: Dict[str, int] = {},
        in_shape: Dict[str, int] = {},
    ):
        super().__init__(
            backbone=backbone,
            neck=neck,
            projector=projector,
            view_transform=view_transform,
            autoencoder=autoencoder,
            heads=heads,
            in_c=in_c,
            in_shape=in_shape,
            out_c=out_c,
        )
        # Images
        self.embedding = embedding

        # View Transform
        self.query_gen = query_gen

    # Decoder.
    def _prepare_decoder(self, query, hq_wq):
        # Alias
        b, nq, Nq, c = query.shape
        hq, wq = hq_wq
        query = rearrange(
            query, "b nq (hq wq) c -> (b nq) c hq wq", b=b, nq=nq, hq=hq, wq=wq, c=c
        )
        return query

    def forward_decoder(self, query, hq_wq):
        # Alias
        b, nq, Nq, c = query.shape
        query = self._prepare_decoder(query, hq_wq)
        query = self.decoder(query)
        return self._arrange_decoder(query, (b, nq))

    def forward(self, imgs, rots, trans, intrins, bev_aug, egoTin_to_seq, **kwargs):
        (
            dict_shape,
            dict_vox,
            dict_img,
            dict_mat,
        ) = self._common_init_backneck_prepare_vt(
            imgs, rots, trans, intrins, bev_aug, egoTin_to_seq
        )

        # Projector
        self._prepare_dict_vox(dict_vox, dict_shape)
        dict_vox.update(self.projector(dict_mat, dict_shape, dict_vox))

        # VT
        b_t = (dict_shape["b"], dict_shape["t"])
        query, hq_wq = self.query_gen(b_t)
        query_pos, hq_wq = self.query_gen(b_t)
        bev_query, *_ = self.view_transform(
            query, query_pos, dict_img["img_feats"], dict_vox
        )

        # Optional: decoder
        if self.decoder is not None:
            bev_query = self.forward_decoder(bev_query, hq_wq)
        else:
            hq, wq = hq_wq
            bev_query = rearrange(
                bev_query, "b nq (hq wq) c -> b nq c hq wq", hq=hq, wq=wq
            )

        # Heads
        dict_out = self.forward_heads(bev_query)

        return {"bev": dict_out}
