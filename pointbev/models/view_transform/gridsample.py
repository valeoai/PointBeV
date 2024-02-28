""" 
Author: Loick Chambon

Extract BeV features given their projected coordinates using the efficient feature-pulling module.
"""

from math import prod
from typing import Dict

import spconv.pytorch as spconv
import torch
from einops import rearrange, repeat
from hydra.utils import instantiate
from torch import nn

from pointbev.ops.gs.functions import sparsed_grid_sample
from pointbev.utils.debug import debug_hook


class GridSampleVT(nn.Module):
    def __init__(
        self,
        voxel_shape=[200, 200, 8],
        in_c: int = 128,
        out_c: int = 128,
        # GS
        N_group: int = 1,
        grid_sample_mode="base",
        # Embedding
        coordembd: Dict = {},
        # Height Compressor
        heightcomp: Dict = {},
        # Optional: defattn after GS.
        defattn: Dict = {},
        # Sparsity
        input_sparse: bool = False,
        return_sparse: bool = False,
    ):
        super().__init__()
        self.register_forward_hook(debug_hook)
        self.voxel_shape = voxel_shape
        self.Z_cam, self.X_cam, self.Y_cam = voxel_shape

        self._init_gs(N_group)
        self.coordembd = coordembd
        self._init_height(heightcomp, in_c, out_c)
        self.input_sparse = input_sparse
        self.return_sparse = return_sparse

        assert grid_sample_mode in [
            "base",
            "sparse_optim",
            None,
        ]
        self.grid_sample_mode = grid_sample_mode

    def _init_gs(self, N_group):
        """Initialize the projection coordinate layer adding the camera projection coordinates to the channels
        of the features."""
        self.N_group = N_group

    def _init_height(self, heightcomp, in_c, out_c):
        """Initialize the height compressor layer."""
        compressor = heightcomp.get("comp", None)
        mode = compressor.pop("mode")
        self.heightcompr_mode = mode
        assert mode in [
            "convn",
            "mlp",
            None,
        ], "heightcompr: mode not supported."
        if mode in ["mean", "sum"]:
            assert in_c == out_c

        compressor["_target_"] = compressor.pop("classname")
        if mode in ["mlp", "convn"]:
            compressor["in_c"] *= self.Y_cam
        if mode == "convn":
            compressor["norm"] = eval(compressor["norm"])
            compressor["activ"] = eval(compressor["activ"])
        layer = instantiate(compressor)
        self.compressor = layer
        return

    def _set_axis(self, voxcam_coords):
        # B,T,N,Zcam,Ycam,Xcam,3
        voxel_shape = voxcam_coords.shape[-4:-1]
        self.Z_cam, self.Y_cam, self.X_cam = voxel_shape
        return

    # GS
    def get_feats_from_grid_sample(self, voxcam_coords, img_feats, vox_valid):
        """Associate features to the voxel grid using grid sample.

        Args:
            - voxcam_coords: Voxel coordinates. Shape: (bt ,n, Zcam, Ycam, Xcam, 3)
            - img_feats: image features. Shape: (bt, n, c, h w)
            - vox_valid: Voxel mask. Shape: (bt ,n, Zcam, Ycam, Xcam, 1)

        Returns:
        Dense:
            - vox_feats: Voxel features. Shape: (bt ,n, c, Zcam, Ycam, Xcam)
            - vox_valid: Voxel mask. Shape: (bt ,n, 1, Zcam, Ycam, Xcam)
            - index: None

        Sparse:
            - vox_feats: Voxel features. Shape: (Npts, c)
            - vox_valid: None
            - index: index containing indices of non zero voxels in voxel_valid.
        """
        # Alias
        bt, n, *Npts, coord = voxcam_coords.shape
        bt, n, c, h, w = img_feats.shape

        voxcam_coords = rearrange(
            voxcam_coords,
            "bt n zcam ycam xcam i -> (bt n) zcam ycam xcam i",
            bt=bt,
            n=n,
        )
        vox_valid = rearrange(
            vox_valid,
            "bt n zcam ycam xcam 1 -> (bt n) 1 zcam ycam xcam",
            bt=bt,
            n=n,
        )
        img_feats = rearrange(
            img_feats,
            "bt n (c ng) h w -> (bt n) c ng h w",
            bt=bt,
            n=n,
            c=c,
            h=h,
            w=w,
            ng=self.N_group,
        )

        # Grid sample
        if self.grid_sample_mode in ["base"]:
            vox_feats, vox_valid, index = self.get_feats_dense_grid_sample(
                voxcam_coords, img_feats, vox_valid
            )
            vox_feats = vox_feats.unflatten(0, (bt, n))
            vox_valid = vox_valid.unflatten(0, (bt, n))
            assert index is None

            return vox_feats, vox_valid, index
        elif self.grid_sample_mode == "sparse_optim":
            return self.get_feats_sparse_grid_sample(
                voxcam_coords, img_feats, vox_valid, (bt, n, Npts)
            )
        else:
            raise NotImplementedError(
                f"grid_sample_mode {self.grid_sample_mode} not supported."
            )

    def get_feats_dense_grid_sample(self, voxcam_coords, img_feats, vox_valid):
        """Dense format grid sampling does not taking into account the fact that points are masked.

        Options: all calls torch.nn.functional.grid_sample
            Base: basic version.
            Opt mem: nested-padded tensor version.
            Triplane: tri-plane version.
        """
        # GS
        vox_feats = torch.nn.functional.grid_sample(
            img_feats, voxcam_coords, align_corners=False
        )
        vox_feats = vox_feats * vox_valid
        return vox_feats, vox_valid, None

    def get_feats_sparse_grid_sample(self, voxcam_coords, img_feats, vox_valid, shapes):
        """Optimized version of the grid_sampling taking into account that some points are masked.
        Calls the custom package.
        """
        # Unpack
        bt, n, Npts = shapes

        vox_valid_flat = rearrange(
            vox_valid, "(bt n) 1 zcam ycam xcam -> (bt n zcam ycam xcam)", bt=bt, n=n
        )
        index = torch.nonzero(vox_valid_flat).squeeze(1)
        index_cam = (index // prod(Npts)) % n
        index_batch = (index // (prod(Npts) * n)) % bt

        # GS
        select_vox = rearrange(
            voxcam_coords, "btn zcam ycam xcam i -> (btn zcam ycam xcam) i"
        )[index]

        sparse_vox_feats = sparsed_grid_sample(
            img_feats, select_vox, (index_batch * n + index_cam).to(torch.int16)
        )
        return sparse_vox_feats, None, index

    def reduce_cams(self, vox_feats, vox_valid, dim=1, eps=1e-6):
        """Filter zero-features, probably out of bound during grid-sampling.
        Then, compute mean over cameras."""

        mask_mems = vox_valid
        return torch.div(
            (vox_feats * mask_mems).sum(dim=dim),
            (mask_mems).sum(dim=dim) + eps,
        )

    def reduce_cams_sparse(self, sparse_vox_feats, index, shapes):
        """Reduce cameras using the sparse structure of the voxel features.

        Args:
            sparse_vox_feats (Tensor): voxel features. Shape (Npts, c).
            index (Tensor): Non empty voxel indices. Shape (bt*n*zcam*ycam*xcam).
            shapes (Tuple[int]): Information needed to decompose index.

        Returns:
            Tensor: voxel features reduced by cameras.
        """
        # Reduce cams: sum
        # Unpack
        device = sparse_vox_feats.device
        bt, n, Npts = shapes
        zcam, ycam, xcam = Npts
        c = sparse_vox_feats.shape[1]

        index_npts = index % prod(Npts)
        index_batch = (index // (prod(Npts) * n)) % bt
        index_bnpts = index_npts + index_batch * prod(Npts)

        # unique[inv_idx] = index_bnpts
        # Ex: [0,0,1,3,0] -> [0,1,3] / [0,0,1,2,0]
        unique_idx, inv_idx = torch.unique(
            index_bnpts, return_inverse=True, sorted=True
        )
        Nactiv_pts = unique_idx.numel()

        # All cameras points to the same output.
        out = torch.zeros((Nactiv_pts, c), device=device, dtype=sparse_vox_feats.dtype)
        sparse_vox_feats_sum = out.index_add_(
            dim=0, index=inv_idx, source=sparse_vox_feats
        )

        # mean
        cnt = torch.zeros((Nactiv_pts, c), device=device, dtype=sparse_vox_feats.dtype)
        cnt = cnt.index_add_(
            dim=0, index=inv_idx, source=torch.ones_like(sparse_vox_feats)
        )
        sparse_vox_feats_mean = sparse_vox_feats_sum / cnt

        # vox_feats[unique_idx] = sparse_vox_feats_mean
        vox_feats = torch.zeros(
            (bt, zcam, ycam, xcam, c), device=device, dtype=sparse_vox_feats.dtype
        ).view(-1, c)
        vox_feats[unique_idx] = sparse_vox_feats_mean
        vox_feats = rearrange(
            vox_feats,
            "(bt zcam ycam xcam) c -> bt c zcam ycam xcam",
            bt=bt,
            zcam=zcam,
            ycam=ycam,
            xcam=xcam,
            c=c,
        )
        return vox_feats

    def forward_cam(
        self, voxcam_coords, H_W, vox_feats, vox_valid, index=None, shapes=None
    ):
        """Camera processing (Optional) and camera reduction (Mandatory).

        Args:
            voxcam_coords (Tensor): Coordinates of the voxels. Shape: (bt, n, zcam, ycam, xcam, 3)
            H_W (Tuple[int]): Image width and height.
            vox_feats: voxel features.
                Dense: (bt, n, c, zcam, ycam, xcam)
                Sparse: (Npts, c)
            vox_valid: voxel mask.
                Dense: (bt, n, 1, zcam, ycam, xcam)
                Sparse: None
            index: Index of the non zero voxels.
                Dense: None
                Sparse: Tensor. (bt*n*zcam*ycam*xcam)

        Returns:
            vox_feats (Tensor): voxel features with camera reduction.
                Dense: (bt, c, zcam, ycam, xcam)
                Sparse: (Npts, c)
        """
        if not self.grid_sample_mode == "sparse_optim":
            vox_feats = self.reduce_cams(vox_feats, vox_valid)
        else:
            bt, n, *Npts, _ = voxcam_coords.shape
            vox_feats = self.reduce_cams_sparse(vox_feats, index, (bt, n, Npts))
        return vox_feats

    # Height compressor
    def _unpack_batch(self, vox_feats, b_t):
        """(B*t,C,Z,X)->(B,t,C,Z,X)"""
        # Alias
        b, t = b_t

        return rearrange(vox_feats, "(b t) c zcam xcam -> b t c zcam xcam", b=b, t=t)

    def _prepare_conv_compressor(self, vox_feats):
        """(B,C,Z,Y,X) -> (B,C*Y,Z,X)"""
        return rearrange(
            vox_feats,
            "bt c zcam ycam xcam -> bt (c ycam) zcam xcam",
            zcam=self.Z_cam,
            ycam=self.Y_cam,
            xcam=self.X_cam,
        )

    def forward_conv_compressor(self, vox_feats, b_t):
        vox_feats = self._prepare_conv_compressor(vox_feats)
        vox_feats = self.compressor(vox_feats)
        return self._unpack_batch(vox_feats, b_t)

    def forward_mlp_compressor(self, vox_feats, b_t):
        """Alias to forward_conv_compressor.
        MLP is treated as a 1x1 conv."""
        return self.forward_conv_compressor(vox_feats, b_t)

    def forward_height(self, vox_feats, b_t, vox_idx=None):
        if self.heightcompr_mode == "convn":
            vox_feats = self.forward_conv_compressor(vox_feats, b_t)
        elif self.heightcompr_mode == "mlp":
            vox_feats = self.forward_mlp_compressor(vox_feats, b_t)
        return vox_feats, vox_idx

    # Arrange sparse / dense outputs
    def _arrange_dense_outputs(self, vox_feats, vox_idx, b_t, Z_bins, X_bins):
        # Alias
        device = vox_feats.device
        bt = prod(b_t)
        c = vox_feats.size(1)

        # Output
        out = torch.zeros((bt, c, Z_bins, X_bins), device=device)
        mask = torch.zeros((bt, 1, Z_bins, X_bins), device=device, dtype=bool)

        out[vox_idx[:, 2], :, vox_idx[:, 0], vox_idx[:, 1]] = vox_feats
        mask[vox_idx[:, 2], :, vox_idx[:, 0], vox_idx[:, 1]] = 1
        b, t = b_t
        out = rearrange(out, "(b t) c zcam xcam -> b t c zcam xcam", b=b, t=t)
        mask = rearrange(mask, "(b t) 1 zcam xcam -> b t 1 zcam xcam", b=b, t=t)

        vox_feats, mask = [x.flip(-1, -2) for x in (out, mask)]
        return vox_feats, mask

    def _arrange_sparse_outputs(
        self, vox_idx, vox_feats, batch_indices, b_t, Z_bins, X_bins
    ):
        # Alias
        device = vox_feats.device
        bt = prod(b_t)
        b, t = b_t
        c = vox_feats.size(2)

        vox_feats = rearrange(vox_feats, "b t c zcam xcam -> (b t zcam xcam) c")

        # Z_cam, X_cam, Y_cam
        vox_idx = vox_idx[..., :, 0]
        Npts = prod(vox_idx.shape[-2:])
        vox_idx = rearrange(
            vox_idx, "b coords zcam xcam -> (b zcam xcam) coords", coords=3
        )
        vox_idx = vox_idx[..., [0, 1]].long()

        if batch_indices is not None:
            batch_idx = batch_indices.unsqueeze(-1)
        else:
            batch_idx = torch.arange(bt, device=device, dtype=torch.long)
            batch_idx = repeat(batch_idx, "b -> (b N) 1", N=Npts)
        vox_idx = torch.cat((vox_idx, batch_idx), dim=-1)

        # Return sparse elements.
        if self.return_sparse:
            # Spconv needs int32 as indices and batch as zero dim.
            indices = torch.cat([vox_idx[..., -1:], vox_idx[..., :2]], dim=-1).int()
            # Flip
            indices[..., 1:] = (
                torch.tensor([Z_bins - 1, X_bins - 1], device=device) - indices[..., 1:]
            )
            # Get mask from indices.
            mask_ = spconv.SparseConvTensor(
                torch.ones_like(indices[..., :1]), indices, [Z_bins, X_bins], bt
            )
            mask_ = mask_.dense().bool()
            # torch.equal(mask, mask_)
            mask_ = rearrange(mask_, "(b t) 1 h w -> b t 1 h w", b=b, t=t)
            return vox_feats, mask_, indices

        # New
        out = torch.zeros((bt * Z_bins * X_bins, c), device=device)
        mask = torch.zeros((bt * Z_bins * X_bins, 1), device=device)
        indices = (
            vox_idx[:, 0] * bt * X_bins + vox_idx[:, 1] * bt + vox_idx[:, 2]
        ).unsqueeze(1)
        out = torch.scatter(out, 0, indices.expand(indices.shape[0], c), vox_feats)
        out = rearrange(out, "(h w b) c -> b c h w", b=bt, h=Z_bins, w=X_bins)
        mask = torch.scatter(
            mask, 0, indices, torch.ones_like(indices, dtype=torch.float32)
        ).bool()
        mask = rearrange(mask, "(h w b) c -> b c h w", b=bt, h=Z_bins, w=X_bins)

        out = rearrange(out, "(b t) c zcam xcam -> b t c zcam xcam", b=b, t=t)
        mask = rearrange(mask, "(b t) 1 zcam xcam -> b t 1 zcam xcam", b=b, t=t)

        vox_feats, mask = [x.flip(-1, -2) for x in (out, mask)]
        return vox_feats, mask, None

    def _arrange_outputs(
        self,
        vox_idx,
        vox_feats,
        batch_indices=None,
        b_t=1,
    ):
        # Ouptut
        mask, indices = None, None
        Z_bins, X_bins = self.voxel_shape[:2]

        # Not sparse
        if not self.input_sparse:
            return vox_feats, mask, indices

        # Sparse
        if self.heightcompr_mode in ["quick_mean", "quick_sum"]:
            vox_feats, mask = self._arrange_dense_outputs(
                vox_feats, vox_idx, b_t, Z_bins, X_bins
            )
        else:
            vox_feats, mask, indices = self._arrange_sparse_outputs(
                vox_idx, vox_feats, batch_indices, b_t, Z_bins, X_bins
            )
        return vox_feats, mask, indices

    def _add_coords_embd(self, vox_feats, vox_coords):
        """Add coordinate embedding, voxel coordinates should be normalized and in the seqaug reference frame."""
        coordembd = self.coordembd(vox_coords)
        coordembd = rearrange(
            coordembd, "b t zcam ycam xcam c -> (b t) c zcam ycam xcam"
        )
        return vox_feats + coordembd

    # Forward
    def forward(self, img_feats, dict_vox, model_3D_processor=None):
        # Unpack
        voxcam_coords, vox_valid, vox_idx, vox_coords, batch_indices = (
            dict_vox["voxcam_coords"],
            dict_vox["vox_valid"],
            dict_vox.get("vox_idx", None),
            dict_vox.get("vox_coords", None),
            dict_vox.get("batch_indices", None),
        )

        # Alias
        (b, t, n, *_) = voxcam_coords.shape

        self._set_axis(voxcam_coords)

        # Prepare
        voxcam_coords = rearrange(
            voxcam_coords, "b t n zcam ycam xcam i -> (b t) n zcam ycam xcam i", i=3
        )
        vox_valid = rearrange(
            vox_valid, "b t n zcam ycam xcam i -> (b t) n zcam ycam xcam i", i=1
        )

        # GS
        vox_feats, vox_valid, index = self.get_feats_from_grid_sample(
            voxcam_coords, img_feats, vox_valid
        )

        # Cam-processing
        H_W = img_feats.shape[-2:]
        vox_feats = self.forward_cam(voxcam_coords, H_W, vox_feats, vox_valid, index)

        # Embedding
        vox_feats = self._add_coords_embd(vox_feats, vox_coords)

        # Height processing
        bev_feats, vox_idx = self.forward_height(vox_feats, (b, t), vox_idx)

        # Sparse or dense output format.
        bev_feats, mask, indices = self._arrange_outputs(
            vox_idx, bev_feats, batch_indices, (b, t)
        )
        return bev_feats, mask, indices
