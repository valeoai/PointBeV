import torch
from einops import rearrange
from hydra.utils import instantiate
from spconv.pytorch import SparseConvTensor
from timm.models.layers import trunc_normal_
from torch import nn

from pointbev.models.layers.attention import PositionalEncodingMap
from pointbev.utils.debug import debug_hook

from sparse_gs import find_indices  # isort:skip


# Temporal
class SparseTemporal(nn.Module):
    def __init__(
        self,
        # Time
        cam_T_P=[[0, 0]],
        bev_T_P=[[0, 0]],
        ws: int = 1,
        layer=None,
        sparse=False,
        embd_dim=128,
        out_mode="present",
        temp_embd_mode=None,
        forward_mode="dense",
        win_neigh=(1, 1),
    ):
        """Given a set of latents, return a set of updated latents.

        - Recurrent: the update takes into account only the previous latent and the memory.
        F(M_{t-1}, L_{t-1}) -> M_{t}
        """
        super().__init__()
        self.register_forward_hook(debug_hook)
        self._init_layer(layer, forward_mode)

        bev_T_P = torch.tensor(bev_T_P)
        cam_T_P = torch.tensor(cam_T_P)
        self._init_embed(temp_embd_mode, cam_T_P, embd_dim)
        self.bev_T = bev_T_P[:, 0]
        self.cam_T = cam_T_P[:, 0]
        self.check_config(out_mode)
        assert forward_mode in [
            "dense",
            "sparse_torchgeom",
            "sparse_hackattn",
        ]
        self.forward_mode = forward_mode
        self.ws = ws
        self.sparse = sparse
        self.win_neigh = win_neigh

    def check_config(self, out_mode):
        assert out_mode in ["present", "same"]
        if out_mode == "present":
            self.keep = torch.where(self.cam_T == 0)[0]

        if out_mode == "same":
            assert torch.equal(
                self.bev_T, self.cam_T
            ), "Are you sure to query the same time?"
            self.keep = [True] * len(self.bev_T)
        self.mode = out_mode

    def _init_layer(self, layer, forward_mode):
        if forward_mode is None:
            self.layer = None
            self.layer_mode = None
            return

        if layer is None:
            return
        self.layer_mode = layer.pop("mode")
        layer["_target_"] = layer.pop("classname")
        self.layer = instantiate(layer)

    def _init_embed(self, embed_mode, cam_T_P, embd_dim):
        assert embed_mode in ["learn", "fourier", None]
        if embed_mode == "learn":
            embed = nn.Parameter(
                torch.randn(len(torch.unique(cam_T_P[:, 0])), embd_dim)
            )
            self.temp_embed = embed
            trunc_normal_(self.temp_embed, std=0.02)
            self.embed_mode_apply = "slice"
        elif embed_mode == "fourier":
            self.temp_embed = PositionalEncodingMap(
                mid_c=embd_dim, out_c=embd_dim, with_mlp=True
            )
            self.embed_mode_apply = "call"
        else:
            self.temp_embed = None
            self.embed_mode_apply = None

    def _get_current_bev_feats(self, bev_feats, i):
        """Get the bev features considered given the window context."""
        # Alias
        ws = self.ws
        b, nq, c, h, w = bev_feats.shape
        assert i <= nq

        if ws < i:
            # Ex: ws=3, nq=5, i=1 -> [0, 1]
            # Ex: ws=3, nq=5, i=2 -> [0, 1, 2]
            cur_bev_feats = bev_feats[:, : i + 1]
        else:
            # Ex: ws=3, nq=5, i=3 -> [1, 2, 3]
            # Ex: ws=3, nq=5, i=4 -> [2, 3, 4]
            cur_bev_feats = bev_feats[:, i - ws + 1 : i + 1]
        return cur_bev_feats

    def _apply_temp_embd(self, features, indices_t, t):
        if self.embed_mode_apply == "slice":
            features = features + self.temp_embed[indices_t]
        elif self.embed_mode_apply == "call":
            features = features + self.temp_embed((indices_t.view(-1, 1) / t) * 2 - 1)
        else:
            raise NotImplementedError
        return features

    # Convn
    def _prepare_convn(self, bev_feats, memory):
        """Merge window frames with channels."""
        state = torch.cat([bev_feats, memory], dim=1)
        state = rearrange(state, "b nw c h w -> b (nw c) h w")
        return state

    def _arange_convn(self, state):
        return rearrange(state, "b c h w -> b 1 c h w")

    # Dispatcher
    def _prepare_state(self, cur_bev_feats, memory):
        if self.layer_mode == "convn":
            state = self._prepare_convn(cur_bev_feats, memory)
        return state

    def _arange_state(self, state):
        if self.layer_mode == "convn":
            state = self._arange_convn(state)
        return state

    # Dense forward
    def forward_dense(self, bev_feats):
        """Update latents
        Args:
            bev_feats (Tensor): bev features representing. Shape: (B, nq, C, H, W)

        Returns:
            bev_feats: Updated version of bev features. Shape: (B, nqout, C, H, W)
        """
        # Alias
        b, nq, c, h, w = bev_feats.shape
        device = bev_feats.device

        # Out: init
        bev_feats_out = torch.empty(b, len(self.bev_T), c, h, w, device=device)

        # Process latents iteratively.
        for i in range(nq):
            if i < self.ws:
                # If lower than the window size, memory is set to the last frame.
                memory = bev_feats[:, i : i + 1]
            else:
                # Otherwise it is set as a function of the window frames.
                cur_bev_feats = self._get_current_bev_feats(bev_feats, i)
                state = self._prepare_state(cur_bev_feats, memory)
                memory = self.layer(state)
                memory = self._arange_state(memory)
            bev_feats_out[:, i] = memory
        return bev_feats_out

    # Sparse forward
    def forward_sparse_hackattn(self, bev_feats, b_t):
        (b, t) = b_t
        indices = bev_feats.indices
        features = bev_feats.features

        # Temporal embedding
        features = self._apply_temp_embd(features, indices[:, 0] % t, t)

        # Considers only the present
        indices[:, 0] = indices[:, 0] // t
        indices = torch.stack(indices.chunk(b * t), dim=1)
        indices = rearrange(indices, "N (b t) c -> (b N) t c", t=t)
        indices = indices[:, t - 1, :]

        features = torch.stack(features.chunk(b * t), dim=1)
        features = rearrange(features, "N (b t) c -> (b N) t c", t=t)
        # Consider only the present
        features = self.layer(features)[:, t - 1]
        return SparseConvTensor(
            features, indices.contiguous(), bev_feats.spatial_shape, b
        )

    def forward_sparse_torchgeom(self, bev_feats, b_t):
        (b, t) = b_t
        X, Y = bev_feats.spatial_shape
        device = bev_feats.features.device
        indices = bev_feats.indices
        indices_flat = indices[:, 0] * X * Y + indices[:, 1] * Y + indices[:, 2]

        # Get query, keys: batch x time level.
        img_mask = torch.zeros(b * t * X * Y, device=device, dtype=indices.dtype)
        img_mask[indices_flat] = 1
        img_mask = rearrange(img_mask, "(b t x y) -> b t x y", b=b, t=t, x=X, y=Y)

        indices_txy = torch.empty(
            (indices.size(0), 4), dtype=indices.dtype, device=device
        )
        indices_txy[:, 0] = indices[:, 0] // t
        indices_txy[:, 1] = indices[:, 0] % t
        indices_txy[:, [2, 3]] = indices[:, 1:3]
        # temporal window size: ((2t+1)-1)/2=t
        index_edge_q, index_edge_k = find_indices(
            indices_txy,
            img_mask,
            (2 * t + 1, self.win_neigh[0], self.win_neigh[1]),
            True,
        )

        features = bev_feats.features
        # Temporal encoding
        features = self._apply_temp_embd(features, indices_txy[:, 1], t)

        # Layer
        new_features = self.layer(
            features, torch.stack([index_edge_k, index_edge_q], dim=0).long()
        )

        unique_q = torch.unique(index_edge_q)
        new_indices = indices[unique_q]
        new_indices[:, 0] = new_indices[:, 0] // t
        new_features = new_features[unique_q]
        return SparseConvTensor(
            new_features, new_indices.contiguous(), bev_feats.spatial_shape, b
        )

    def forward_sparse(self, bev_feats, b_t):
        """Update latents and memory with the window frames.
        Args:
            bev_feats (SparseTensor): sparse tensor with features of shape (Npts, C).

        Returns:
            bev_feats (SparseTensor): Updated sparse tensor of shape (Npts_out, C)
        """
        if "torchgeom" in self.forward_mode:
            return self.forward_sparse_torchgeom(bev_feats, b_t)
        elif "hackattn" in self.forward_mode:
            return self.forward_sparse_hackattn(bev_feats, b_t)

    def forward(self, bev_feats, *args, **kwargs):
        if self.sparse:
            assert "sparse" in self.forward_mode
            return self.forward_sparse(bev_feats, *args, **kwargs)
        else:
            assert "dense" in self.forward_mode
            return self.forward_dense(bev_feats, *args, **kwargs)
