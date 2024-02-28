from einops import rearrange
from torch import nn

from pointbev.models.layers import CADefnAttn, SADefnAttn
from pointbev.utils.debug import debug_hook


class DefAttnVT(nn.Module):
    def __init__(
        self,
        sa_defattn_kwargs,
        ca_defattn_kwargs,
        n_layers=6,
        query_c=128,
        ffn_dim=1028,
        sa_mode="SADefnAttn",
    ):
        super().__init__()
        self.register_forward_hook(debug_hook)

        self.n_layers = n_layers

        self.sa_mode = sa_mode
        self.sa_layers = nn.ModuleList(
            [eval(sa_mode)(**sa_defattn_kwargs) for _ in range(n_layers)]
        )
        self.sa_norm_layers = nn.ModuleList(
            [nn.LayerNorm(query_c) for _ in range(n_layers)]
        )
        self.ca_layers = nn.ModuleList(
            [CADefnAttn(**ca_defattn_kwargs) for _ in range(n_layers)]
        )
        self.ca_norm_layers = nn.ModuleList(
            [nn.LayerNorm(query_c) for _ in range(n_layers)]
        )
        self.mlp_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(query_c, ffn_dim), nn.ReLU(), nn.Linear(ffn_dim, query_c)
                )
                for _ in range(n_layers)
            ]
        )
        self.last_norm_layers = nn.ModuleList(
            [nn.LayerNorm(query_c) for _ in range(n_layers)]
        )

    def forward(self, query, query_pos, img_feats, dict_vox):
        # Unpack
        voxcam_coords, vox_valid, vox_idx = (
            dict_vox["voxcam_coords"],
            dict_vox["vox_valid"],
            dict_vox.get("vox_idx", None),
        )
        # Alias
        h, w = img_feats.shape[-2:]
        b, nq, *_, C = query.shape

        # Forward
        voxcam_coords = voxcam_coords[..., :2]
        ref_pts_cam = rearrange(
            voxcam_coords, "b t n z y x i -> n (b t) (z x) y i", i=2
        )
        mask = rearrange(vox_valid, "b t n z y x 1 -> n (b t) (z x) y")

        spatial_shapes = query.new_zeros([1, 2]).long().to("cuda")
        spatial_shapes[0, 0] = h
        spatial_shapes[0, 1] = w

        query = rearrange(query, "b nq h w c -> (b nq) (h w) c")
        query_pos = rearrange(query_pos, "b nq h w c -> (b nq) (h w) c")
        img_feats = rearrange(img_feats, "bt n c h w -> n (h w) bt c")

        for i in range(self.n_layers):
            queries = self.sa_layers[i](query, query_pos)

            queries = self.sa_norm_layers[i](queries)

            queries = self.ca_layers[i](
                queries,
                img_feats,
                img_feats,
                query_pos,
                ref_pts_cam,
                spatial_shapes,
                mask,
            )
            queries = self.ca_norm_layers[i](queries)

            queries = self.mlp_layers[i](queries) + queries

            queries = self.last_norm_layers[i](queries)

        queries = rearrange(queries, "(b nq) Nq c -> b nq Nq c", b=b, nq=nq)
        mask = None
        return queries, mask
