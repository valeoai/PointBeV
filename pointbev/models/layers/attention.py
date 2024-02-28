from typing import Optional

import torch
from einops import rearrange, repeat
from torch import Tensor, nn

from .common import MLP

USE_FAISS = True

try:
    from pointbev.ops.defattn.modules import MSDeformAttn, MSDeformAttn3D
except:
    print("Deformable attention not installed.")


# Deformable Attention
class SADefnAttn(nn.Module):
    def __init__(
        self,
        in_c=128,
        dropout=0.1,
        query_shape=[200, 200],
        msdef_kwargs={"n_levels": 1, "n_heads": 4, "n_points": 8},
    ):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.deformable_attention = MSDeformAttn(d_model=in_c, **msdef_kwargs)
        self.mlp_out = nn.Linear(in_c, in_c)
        self.ref_points = self._init_ref_points(*query_shape)

    def _init_ref_points(self, Z, X):
        ref_z, ref_x = torch.meshgrid(
            torch.linspace(0.5, Z - 0.5, Z), torch.linspace(0.5, X - 0.5, X)
        )
        ref_z = rearrange(ref_z, "z x -> (z x)") / Z
        ref_x = rearrange(ref_x, "z x -> (z x)") / X
        ref_points = torch.stack((ref_z, ref_x), dim=-1)
        return ref_points

    def forward(self, query, query_pos=None):
        # Alias
        B, N, C = query.shape
        device = query.device

        # Get residual
        query_residual = query.clone()

        # Add positional encoding
        if query_pos is not None:
            query = query + query_pos

        # Reference points
        ref_points = self.ref_points.to(device)
        ref_points = repeat(ref_points, "zx coords -> b zx coords", b=B, coords=2)
        ref_points = rearrange(
            ref_points, "b zx coords -> b zx 1 coords", b=B, coords=2
        )

        # Define shapes
        input_spatial_shapes = query.new_full([1, 2], fill_value=200).long()
        input_level_start_index = query.new_zeros([1]).long()

        # Process
        queries = self.deformable_attention(
            query,
            ref_points,
            query.clone(),
            input_spatial_shapes,
            input_level_start_index,
        )

        return self.dropout(self.mlp_out(queries)) + query_residual


class CADefnAttn(nn.Module):
    def __init__(
        self,
        in_c=128,
        dropout=0.1,
        msdef_kwargs={"num_levels": 1, "num_heads": 4, "num_points": 8},
    ):
        super().__init__()
        self.in_c = in_c
        self.dropout = nn.Dropout(dropout)
        self.deformable_attention = MSDeformAttn3D(embed_dims=in_c, **msdef_kwargs)
        self.output_proj = nn.Linear(in_c, in_c)

    def forward(
        self,
        query,
        key,
        value,
        query_pos=None,
        ref_pts_cam=None,
        spatial_shapes=None,
        bev_mask=None,
    ):
        # Alias
        b, Nq, c = query.shape
        n, hw, _, _ = key.shape
        y = ref_pts_cam.size(3)

        # Init
        query_out = torch.zeros_like(query)
        level_start_index = query.new_zeros([1]).long()

        # Get residual
        query_residual = query

        # Add positional encoding
        if query_pos is not None:
            query = query + query_pos

        # Available queries in images.
        idx_per_cam = []
        for mask_cam in bev_mask:
            idx_per_batch = []
            for j in range(b):
                # b zx y -> b zx
                mask_cam_b = mask_cam[j]
                idx_query_cam_b = mask_cam_b.sum(dim=-1).nonzero().squeeze(-1)
                idx_per_batch.append(idx_query_cam_b)
            idx_per_cam.append(idx_per_batch)
        max_len = max([max([len(idx) for idx in idx_b]) for idx_b in idx_per_cam])

        # Init
        queries_rebatch = query.new_zeros([b, n, max_len, c])
        ref_pts_rebatch = ref_pts_cam.new_zeros([b, n, max_len, y, 2])

        # Rebatch
        for i, ref_pts_c in enumerate(ref_pts_cam):
            for j in range(b):
                idx_query_cam_b = idx_per_cam[i][j]
                queries_rebatch[j, i, : len(idx_query_cam_b)] = query[
                    j, idx_query_cam_b
                ]
                ref_pts_rebatch[j, i, : len(idx_query_cam_b)] = ref_pts_c[
                    j, idx_query_cam_b
                ]

        # Prepare
        key = rearrange(key, "n hw b c -> (b n) hw c", n=n, hw=hw, b=b)
        value = rearrange(value, "n hw b c -> (b n) hw c", n=n, hw=hw, b=b)
        queries_rebatch = rearrange(
            queries_rebatch, "b n max c -> (b n) max c", b=b, n=n, c=c
        )
        ref_pts_rebatch = rearrange(
            ref_pts_rebatch, "b n max y c -> (b n) max y c", c=2
        )

        # Process
        queries = self.deformable_attention(
            query=queries_rebatch,
            key=key,
            value=value,
            reference_points=ref_pts_rebatch,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
        ).view(b, n, max_len, self.in_c)

        # Copy back
        for i, idx_query_cam in enumerate(idx_per_cam):
            for j in range(b):
                query_out[j, idx_query_cam[j]] += queries[j, i, : len(idx_query_cam[j])]

        # Number of activated queries
        count = bev_mask.sum(-1) > 0
        count = rearrange(count, "n b Nq -> b Nq 1 n", b=b, Nq=Nq, n=n).sum(-1)
        count = torch.clamp(count, min=1.0)

        # Average by activated
        query_out = query_out / count

        # Output
        return self.dropout(self.output_proj(query_out)) + query_residual


# Torch geometric
class TorchGeomAttnBlock(nn.Module):
    def __init__(
        self,
        in_c,
        mid_c,
        out_c,
        attnlayer,
        residual: bool = True,
        widening_factor_mlp: int = 1,
        dropout_mlp: float = 0.0,
    ):
        super().__init__()
        self.attnlayer = attnlayer
        self.norm1 = nn.LayerNorm(in_c)
        self.residual = residual
        self.mlp = nn.Sequential(
            nn.Linear(mid_c, int(mid_c * widening_factor_mlp)),
            nn.GELU(),
            nn.Linear(int(mid_c * widening_factor_mlp), out_c),
            nn.Dropout(dropout_mlp),
        )
        self.norm2 = nn.LayerNorm(out_c)
        self.register = False
        self.registered_weights = []
        self.registered_x = []
        return

    def forward(self, x, edges=None):
        x_o = self.norm1(x)
        if self.register:
            self.registered_x.append(x_o)
        if edges is None:
            if self.register:
                x_o, attn_weights = self.attnlayer(x_o, return_attention=self.register)
                self.registered_weights = attn_weights
            else:
                x_o = self.attnlayer(x_o)
        else:
            if self.register:
                x_o, attn_weights = self.attnlayer(
                    x_o, edges, return_attention_weights=self.register
                )
                self.registered_weights = attn_weights
            else:
                x_o = self.attnlayer(x_o, edges)

        if self.residual:
            x_o = x_o + x
        x_o = self.mlp(self.norm2(x_o)) + x_o
        return x_o


# Positional embedding
def positional_encoding(v: Tensor, bvals: Tensor, avals: Tensor) -> Tensor:
    vp = 2 * torch.pi * bvals * torch.unsqueeze(v, -1)
    vp_cat = torch.cat(
        (avals * torch.cos(vp), avals * torch.sin(vp)), dim=-1
    ) / torch.norm(avals)
    return vp_cat.flatten(-2, -1)


class PositionalEncodingMap(nn.Module):
    def __init__(
        self,
        m: Optional[int] = 8,
        bvals: Optional[Tensor] = None,
        avals: Optional[Tensor] = None,
        with_mlp: bool = False,
        in_c=1,
        out_c=1,
        num_hidden_layers=2,
        mid_c=256,
    ):
        """
        Returns: (sin(2^0 * pi * x), cos(2^0 * pi * x), ..., sin(2^(m-1) * pi * x), cos(2^(m-1) * pi * x))

        Note: in NeRF it is applied to the 3 coordinate values normalized in [-1,1] (m=10) and to the three components
        of the cartesian viewing direction (m=4).

        Source:
        - https://arxiv.org/pdf/2003.08934.pdf (sec. 5.1)
        - https://github.com/jmclong/random-fourier-features-pytorch
        """

        if bvals is None:
            if m is not None:
                bvals = 2 ** (torch.arange(m) - 1)
            else:
                raise ValueError("Either bvals or m must be provided")
        super().__init__()

        self.register_buffer("bvals", bvals)

        if avals is None:
            avals = torch.tensor([1.0])
        self.register_buffer("avals", avals)

        if with_mlp:
            act = nn.GELU()
            self.layer = MLP(
                int(2 * in_c * len(bvals)),
                mid_c,
                out_c,
                num_hidden_layers + 1,
                act,
            )
        else:
            self.layer = nn.Identity()

    def forward(self, v: Tensor) -> Tensor:
        return self.layer(positional_encoding(v, self.bvals, self.avals))
