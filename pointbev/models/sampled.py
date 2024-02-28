import math
from math import prod
from typing import Dict, Optional, Tuple

import spconv.pytorch as spconv
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import Tensor, nn

from pointbev.models.common import CoordSelector, Network
from pointbev.utils import list_dict_to_dict_list


class PointBeV(Network):
    def __init__(
        self,
        # Modules
        backbone=None,
        neck=None,
        projector=None,
        view_transform=None,
        autoencoder: Optional[nn.Module] = None,
        temporal=None,
        heads=None,
        # Configs
        in_c={},
        out_c={},
        in_shape={},
        voxel_ref="spatial",
        sampled_kwargs={},
    ):
        super().__init__(
            backbone=backbone,
            neck=neck,
            projector=projector,
            view_transform=view_transform,
            autoencoder=autoencoder,
            temporal=temporal,
            heads=heads,
            in_c=in_c,
            out_c=out_c,
            in_shape=in_shape,
            voxel_ref=voxel_ref,
            init_coordselec=False,
        )

        # Selector
        if sampled_kwargs:
            self.with_fine = sampled_kwargs.get("with_fine", False)
            self.valid_fine = sampled_kwargs.get("valid_fine", False)
            self.temp_thresh = sampled_kwargs.get("temp_thresh", False)
        self.coord_selector = SampledCoordSelector(in_shape, voxel_ref, sampled_kwargs)

        return

    # Sampled
    def forward_sampled(
        self,
        dict_img,
        dict_mat,
        dict_vox,
        dict_shape,
        decoder=None,
    ):
        # Unpack
        img_feats = dict_img["img_feats"]

        # Alias
        b, t = dict_shape["b"], dict_shape["t"]

        dict_vox.update(self.projector(dict_mat, dict_shape, dict_vox))

        bev_feats, mask, vox_idx = self.view_transform(
            img_feats,
            dict_vox,
        )

        kwargs = {
            "feats": bev_feats,
            "indices": vox_idx,
            "spatial_shape": self.coord_selector.spatial_range[:2],
            "batch_size": b * t,
            "from_dense": False,
        }
        bev_feats = decoder(**kwargs)

        bev_feats = self.forward_temporal(bev_feats, dict_shape)

        # Heads
        dict_out = self.forward_heads(bev_feats, dict_shape)
        mask_dict = {k: mask for k in dict_out.keys()}
        return dict_out, mask_dict, bev_feats

    def forward_coarse(self, dict_img, dict_mat, dict_vox, dict_shape):
        return self.forward_sampled(
            dict_img, dict_mat, dict_vox, dict_shape, self.decoder
        )

    def forward_fine(self, dict_img, dict_mat, dict_vox, dict_shape):
        return self.forward_sampled(
            dict_img, dict_mat, dict_vox, dict_shape, self.decoder
        )

    def _process_listdict(self, list_dict, list_masks, dict_shape):
        # Alias
        b = dict_shape["b"]
        dict_out = list_dict_to_dict_list(list_dict)
        dict_masks = list_dict_to_dict_list(list_masks)
        assert dict_out.keys() == dict_masks.keys()

        # Densify & Shape
        for d in [dict_out, dict_masks]:
            for k in d.keys():
                for i, elem in enumerate(d[k]):
                    if not isinstance(elem, torch.Tensor):
                        elem = elem.dense()

                    if len(elem.shape) == 4:
                        d[k][i] = rearrange(elem, "(b nq) c h w -> b nq c h w", b=b)
                    else:
                        d[k][i] = elem
                    # Keep only present.
                    d[k][i] = d[k][i][:, -1:]

        # Gather
        for k in dict_out.keys():
            elem = torch.stack(dict_out[k], dim=0).sum(0)
            temp = torch.stack(dict_masks[k], dim=0)
            sum_masks = temp.float().sum(0)
            union = temp.sum(0).bool()
            non_union = (~union).float()
            dict_out[k] = elem / (sum_masks + non_union)
            dict_masks[k] = union

        return dict_out, dict_masks

    def forward_coarse_and_fine(
        self,
        dict_img,
        dict_mat,
        dict_shape,
        dict_vox,
        sampling_imgs={},
    ):
        dict_vox.update(
            self.coord_selector._get_vox_coords_and_idx(
                dict_shape, dict_mat, sampling_imgs
            )
        )

        out_coarse, masks_coarse, feats_coarse = self.forward_coarse(
            dict_img, dict_mat, dict_vox, dict_shape
        )

        list_masks = [masks_coarse]
        list_out = [out_coarse]
        # Track number of points
        if "binimg" in masks_coarse.keys():
            n_coarse = masks_coarse["binimg"].sum() / masks_coarse["binimg"].flatten(
                0, 1
            ).size(0)
        else:
            n_coarse = -1  # hdmap training
        n_fine = torch.tensor([0.0], device="cuda")

        bool_skip = False
        if self.with_fine and (
            (self.training and not self.temporal) or (self.valid_fine)
        ):
            dict_vox = self.coord_selector._get_sampled_fine_coords(
                out_coarse,
                masks_coarse,
            )
            if dict_vox["vox_coords"].size(2) == 0:
                bool_skip = True

            if not bool_skip:
                out_fine, masks_fine, _ = self.forward_fine(
                    dict_img, dict_mat, dict_vox, dict_shape
                )
                list_out.append(out_fine)
                list_masks.append(masks_fine)
                # Track number of points
                if "binimg" in masks_coarse.keys():
                    n_fine = masks_fine["binimg"].sum() / masks_fine["binimg"].flatten(
                        0, 1
                    ).size(0)

        if not bool_skip:
            out, masks = self._process_listdict(list_out, list_masks, dict_shape)
        else:
            out = out_coarse
            masks = masks_coarse

        if self.with_fine and self.valid_fine:
            # ! Hard negative value on non sampled points.
            if "binimg" in out.keys():
                key = "binimg"
            elif "hdmap" in out.keys():
                key = "hdmap"
            out[key] = out[key] + ((1 - masks[key].float()) * -10000)

        tracks = {"N_coarse": n_coarse, "N_fine": n_fine}
        return out, masks, tracks

    # Heads
    def _prepare_heads(self, bev_feats):
        return bev_feats

    def _arrange_heads(self, dict_out, dict_shape):
        """Shape: Sparse -> (B*nq,C,H,W) -> (B,nq,C,H,W)"""
        b = dict_shape["b"]
        for k in dict_out.keys():
            temp = dict_out[k]
            # Densify
            if not isinstance(temp, torch.Tensor):
                temp = temp.dense()
            dict_out[k] = rearrange(temp, "(b nq) c h w -> b nq c h w", b=b)
        return dict_out

    def forward_heads(self, bev_feats, dict_shape):
        bev_feats = self._prepare_heads(bev_feats)
        dict_out = self.heads(bev_feats)
        return self._arrange_heads(dict_out, dict_shape)

    # Temporal
    def forward_temporal(self, bev_feats, dict_shape):
        if not self.temporal:
            return bev_feats
        else:
            b, t = (dict_shape["b"], dict_shape["t"])

            if not self.training:
                # Alias
                X, Y = bev_feats.spatial_shape
                device = bev_feats.features.device

                # For the past only: get important elements.
                dict_out = self.forward_heads(bev_feats, dict_shape)
                index_b, index_t, index_x, index_y = torch.where(
                    dict_out["binimg"][:, :-1].squeeze(2) > self.temp_thresh
                )
                past_indices = (
                    (index_b * dict_shape["t"] + index_t) * X * Y
                    + index_x * Y
                    + index_y
                )
                present_indices_b = torch.arange(b, device=device).repeat_interleave(
                    X * Y
                ) * t + (t - 1)
                present_indices_xy = (
                    torch.arange(0, X * Y, device=device).repeat(b, 1).flatten(0, 1)
                )
                present_indices = present_indices_b * X * Y + present_indices_xy
                indices = torch.cat([past_indices, present_indices], dim=0)

                bev_feats = spconv.SparseConvTensor(
                    torch.index_select(bev_feats.features, 0, indices),
                    torch.index_select(bev_feats.indices, 0, indices),
                    [X, Y],
                    bev_feats.batch_size,
                )
            return self.temporal(bev_feats, (b, t))

    # Forward
    def forward(self, imgs, rots, trans, intrins, bev_aug, egoTin_to_seq, **kwargs):
        (
            dict_shape,
            dict_vox,
            dict_img,
            dict_mat,
        ) = self._common_init_backneck_prepare_vt(
            imgs, rots, trans, intrins, bev_aug, egoTin_to_seq
        )

        sampling_imgs = {
            "lidar": kwargs.get("lidar_img", None),
            "hdmap": kwargs.get("hdmap", None),
        }
        out, masks, tracks = self.forward_coarse_and_fine(
            dict_img,
            dict_mat,
            dict_shape,
            dict_vox,
            sampling_imgs,
        )
        dict_out = {"bev": out}
        dict_out["masks"] = {"bev": masks}
        dict_out["tracks"] = tracks

        return dict_out


class SampledCoordSelector(CoordSelector):
    def __init__(self, spatial_kwargs, voxel_ref, coordselec_kwargs={}):
        super().__init__(spatial_kwargs, voxel_ref, init_buffer=False)

        # Init
        self._init_status(coordselec_kwargs)
        self._init_buffer(self.mode, self.val_mode)
        return

    def _init_buffer(self, mode, val_mode):
        self._set_cache_dense_coords()
        X, Y, Z = self.spatial_range
        self._set_cache_grid(mode, val_mode, self.N_coarse, X, Y)

    def _init_status(self, sampled_kwargs):
        # Coarse pass
        self.mode = sampled_kwargs["mode"]
        assert self.mode in [
            # Dense sampling
            "dense",
            # Pillar sampling
            "rnd_pillars",
            "regular_pillars",
            "rnd_patch_pillars",
        ], NotImplementedError("Unsupported mode")
        self.val_mode = sampled_kwargs.get("val_mode", "dense")

        # Coarse
        self.N_coarse = sampled_kwargs["N_coarse"]
        self.patch_size = sampled_kwargs["patch_size"]

        # Fine pass
        self.N_fine = sampled_kwargs["N_fine"]
        self.N_anchor = sampled_kwargs["N_anchor"]
        self.fine_patch_size = sampled_kwargs["fine_patch_size"]
        self.fine_thresh = sampled_kwargs["fine_thresh"]
        return

    # Get voxels.
    def _get_vox_coords_and_idx(
        self, dict_shape, dict_mat, sampling_imgs={}
    ) -> Dict[str, Tensor]:
        # Alias
        rots = dict_mat["rots"]
        bt = rots.size(0)
        device = rots.device

        # Prepare out
        dict_vox = {
            "vox_coords": None,
            "vox_idx": None,
        }

        # During training
        if self.training:
            if self.mode == "dense":
                dict_out = self._get_dense_coords(bt)
            else:
                dict_out = self._get_sparse_coords(
                    bt,
                    device,
                    self.mode,
                )
        # During validation
        else:
            if self.val_mode == "dense":
                dict_out = self._get_dense_coords(bt)
            elif (sampling_imgs["lidar"] is not None) and (self.val_mode == "lidar"):
                dict_out = self._get_img_pts(sampling_imgs["lidar"], device)
            elif (sampling_imgs["hdmap"] is not None) and (self.val_mode == "hdmap"):
                dict_out = self._get_img_pts(sampling_imgs["hdmap"][:, :, :1], device)
            else:
                dict_out = self._get_sparse_coords(bt, device, self.val_mode)
        dict_vox.update(dict_out)

        return dict_vox

    def _get_img_pts(self, img, device, subsample=True):
        """
        Get point coordinates using an image, for instance a lidar map (projection of lidar points) or an hdmap.
        """
        # Alias
        X, Y, Z = self.spatial_range
        sb = self.spatial_bounds
        assert img.size(0) == 1, "img evaluation only support val_bs=1"

        # From lidar img to xyz
        vox_idx = torch.nonzero(img.squeeze(0).squeeze(0).squeeze(0))[:, -2:]

        # Subsample
        if subsample:
            N_pts = min(self.N_coarse, vox_idx.size(0))
            rnd = torch.randperm(vox_idx.size(0))[:N_pts].to(device)
            vox_idx = torch.index_select(vox_idx, dim=0, index=rnd)

        vox_idx = torch.tensor([X - 1, Y - 1], device=device) - vox_idx
        vox_idx = repeat(vox_idx, "N c -> (N Z) c", Z=Z)
        vox_idx = torch.cat(
            [
                vox_idx,
                repeat(
                    torch.arange(Z, device=device).view(-1, 1),
                    "Z c -> (N Z) c",
                    N=vox_idx.size(0) // Z,
                ),
            ],
            dim=-1,
        )
        scale = torch.tensor(
            [sb[1] - sb[0], sb[3] - sb[2], sb[5] - sb[4]], device=device
        )
        dist = torch.tensor([abs(sb[0]), abs(sb[2]), abs(sb[4])], device=device)
        xyz = torch.tensor([X - 1, Y - 1, Z - 1], device=device)
        vox_coords = (vox_idx / xyz * scale) - dist
        return {
            "vox_coords": repeat(vox_coords, "(xy z) c -> b c xy 1 z", b=1, z=Z),
            "vox_idx": repeat(vox_idx, "(xy z) c -> b c xy 1 z", b=1, z=Z),
        }

    def _set_cache_grid(
        self, mode: str, val_mode: str, N_pts: int, X: int, Y: int
    ) -> None:
        """Get, and / or, set index grid.
        Avoid creating grid at each forward pass.

        Mode:
            - regular_pillars: Regular grid of size (sqrt_Npts,sqrt_Npts) in range [0,1].

            - rnd_pillars, test_*: Regular grid of size (X,Y) in range [0,1].

            - rnd_patch_pillars: Regular grid of size (X,Y) in range [X,Y].
        """
        for m, name in zip([mode, val_mode], ["grid_buffer", "grid_buffer_val"]):
            if m == "regular_pillars":
                sqrtN = int(math.sqrt(N_pts))
                grid = torch.stack(
                    torch.meshgrid(
                        torch.linspace(0, 1, sqrtN),
                        torch.linspace(0, 1, sqrtN),
                        indexing="ij",
                    ),
                    dim=-1,
                )
            elif m in [
                "rnd_pillars",
            ]:
                grid = torch.stack(
                    torch.meshgrid(
                        torch.linspace(0, 1, X), torch.linspace(0, 1, Y), indexing="ij"
                    ),
                    dim=-1,
                )
            elif m in ["rnd_patch_pillars"]:
                grid = torch.stack(
                    torch.meshgrid(
                        torch.arange(0, X), torch.arange(0, Y), indexing="ij"
                    ),
                    dim=-1,
                )
            else:
                grid = None
            self.register_buffer(name, grid)
        return grid

    def _get_sparse_coords(
        self,
        bt: int,
        device: str,
        mode: str = "rnd_pillars",
    ) -> Dict[str, Tensor]:
        """Sample points or pillars in 3D space between 3D spatial bounds.

        Args:
            mode: Select either :
                - "random_pillar" to sample pillars in 3D space.

                - "random_grouped_points" to sample points grouped in window units in 3D space.
                - "random_grouped_pillar" to sample pillars grouped in window units in 3D space.
        """
        # Alias
        X, Y, Z = self.spatial_range
        sb = self.spatial_bounds
        N_coarse = self.N_coarse
        patch_size = self.patch_size
        grid = self.grid_buffer if self.training else self.grid_buffer_val

        # Points:
        if mode == "regular_pillars":
            pillars = repeat(grid, "x y c -> b (x y) c", b=bt)
        elif mode == "rnd_pillars":
            pillars = repeat(grid, "x y c -> b (x y) c", b=bt)
            rnd = torch.randperm(X * Y)[:N_coarse].to(device)
            pillars = torch.index_select(pillars, dim=1, index=rnd)
        elif mode == "rnd_patch_pillars":
            # Get # anchors.
            N_anchors = N_coarse // (patch_size**2)
            perm = torch.randperm(X * Y)[:N_anchors].to(device)
            pillars = repeat(grid, "x y c -> b (x y) c", b=bt)
            pillars_anchor = torch.index_select(pillars, dim=1, index=perm)

            if patch_size != 1:
                flat_idx = pillars_anchor[..., 1] * Y + pillars_anchor[..., 0]
                # Densify
                mask = torch.zeros((bt, X * Y), device=device)
                mask = torch.scatter(mask, 1, flat_idx, 1)
                mask = mask.view(bt, 1, X, Y)
                mask = self._densify_mask(mask, patch_size)
                # Fill remaining points
                xy_vox_idx = self._select_idx_to_keep(mask, N_coarse, (X, Y))
            else:
                xy_vox_idx = pillars_anchor
            div = torch.tensor([X - 1, Y - 1], device=device).view(1, 1, -1)
            pillars = xy_vox_idx / div
        else:
            # Print error
            raise NotImplementedError(f"Unsupported mode: {mode}")

        Nxy = pillars.size(1)
        pillars = repeat(pillars, "bt xy c -> bt c (xy z)", z=Z)

        # -> Regular Z points
        pillar_heights = torch.linspace(0, 1, Z, device=device)
        pillar_heights = repeat(pillar_heights, "z -> bt 1 (xy z)", bt=bt, xy=Nxy)

        # Pillar pts: [0,1]
        pillar_pts = torch.cat([pillars, pillar_heights], dim=1)

        # Voxel coordinates: [-BoundMin, BoundMax]
        scale = torch.tensor(
            [sb[1] - sb[0], sb[3] - sb[2], sb[5] - sb[4]], device=device
        ).view(1, 3, 1)
        dist = torch.tensor([abs(sb[0]), abs(sb[2]), abs(sb[4])], device=device).view(
            1, 3, 1
        )
        vox_coords = pillar_pts * scale - dist

        # Voxel indices: [0,X-1]
        xyz = torch.tensor([X - 1, Y - 1, Z - 1], device=device).view(1, 3, 1)
        vox_idx = (pillar_pts * xyz).round().to(torch.int32)

        # Out
        dict_vox = {
            "vox_coords": rearrange(vox_coords, "b c (xy z) -> b c xy 1 z", z=Z),
            "vox_idx": rearrange(vox_idx, "b c (xy z) -> b c xy 1 z", z=Z),
        }
        return dict_vox

    def _get_flat_idx(self, bt, hw, device):
        flat_idx = torch.stack(
            torch.meshgrid(
                torch.arange(0, bt, device=device),
                torch.arange(0, hw, device=device),
                indexing="ij",
            )
        ).to(dtype=torch.int32)
        return flat_idx

    def _densify_mask(
        self,
        mask: Tensor,
        patch_size: int,
    ) -> Tensor:
        """Augment the mask by convolving it with a kernel of size patch_size. The larger
        the kernel, the more points are considered activated.

        Force: torch.float64 to use nonzero to get indices, otherwise values are nearly zero.
        """
        # Alias
        device = mask.device
        kernel = torch.ones(
            (1, 1, patch_size, patch_size), dtype=torch.float64, device=device
        )
        augm_mask = F.conv2d(
            mask.to(torch.float64), kernel, padding=(patch_size - 1) // 2
        )
        augm_mask = augm_mask.bool()
        augm_mask = rearrange(augm_mask, "bt 1 X Y -> bt (X Y)")
        return augm_mask

    def _select_idx_to_keep(self, mask: Tensor, N_pts: int, X_Y: Tuple[int]) -> Tensor:
        """Select final points to keep.
        Either we keep Nfine points ordered by their importance or we reinject random points when points are
        predicted as not important, otherwise we will have an artefact at the bottom due to the selection
        on uniform null points.
        """
        # Alias
        bt = mask.size(0)
        device = mask.device
        X, Y = X_Y

        out_idx = []
        if N_pts == "dyna":
            for i in range(bt):
                # Numbers of activated elements
                activ_idx = torch.nonzero(mask[i]).squeeze(1)
                out_idx.append(activ_idx)
        else:
            # Reinject random points in batches
            for i in range(bt):
                # Numbers of activated elements
                activ_idx = torch.nonzero(mask[i]).squeeze(1)
                # How many points are not activated.
                n_activ = activ_idx.size(0)
                idle = N_pts - n_activ

                # Less detected points than N_pts
                if idle > 0:
                    # Random selection
                    allowed_idx = torch.nonzero(mask[i] == 0).squeeze(1)
                    perm = torch.randperm(allowed_idx.size(0))
                    augm_idx = allowed_idx[perm[:idle]]
                else:
                    augm_idx = torch.empty([0], device=device, dtype=torch.int64)
                    activ_idx = activ_idx[:N_pts]

                out_idx.append(torch.cat([activ_idx, augm_idx]))

        out_idx = torch.stack(out_idx)
        xy_vox_idx = torch.stack([((out_idx // Y) % X), out_idx % Y], dim=-1)
        return xy_vox_idx

    @torch.no_grad()
    def _get_sampled_fine_coords(self, out, masks) -> Dict[str, Tensor]:
        """Select points according to the coarse pass logit output.
        Args:
            - N_anchor: Number of anchor points to select most relevant locations (highest logits).
            - Patch_size: Size of the patch to select around the anchor points.
        """
        # Alias
        sb = self.spatial_bounds
        N_anchor: str | int = self.N_anchor
        patch_size = self.fine_patch_size
        X, Y, Z = self.spatial_range
        N_fine = self.N_fine

        # Extract anchor points
        # Flip: (0,0) from top left to bottom right
        if "binimg" in out.keys():
            key = "binimg"
        elif "hdmap" in out.keys():
            key = "hdmap"
        else:
            raise KeyError(f"Unexpected key: {list(out.keys())}")
        out_score = (out[key].sigmoid() * masks[key]).flip(-2, -1)

        b, t, _, h, w = out_score.shape
        device = out_score.device
        if key == "binimg":
            out_score = out_score[:, :, 0]
        elif key == "hdmap":
            out_score = out_score.max(2).values
        out_score = rearrange(out_score, "b t h w -> (b t) (h w)")

        # Indices stop gradients, i.e no gradient backpropagation.
        flat_idx = self._get_flat_idx(b * t, h * w, device)
        flat_idx = rearrange(flat_idx, "c bt hw -> bt hw c", c=2)
        mask = self._get_fine_mask(out_score, flat_idx, (X, Y), N_anchor)
        mask = self._densify_mask(mask, patch_size)
        xy_vox_idx = self._select_idx_to_keep(mask, N_fine, (X, Y))

        xy_vox_idx = repeat(xy_vox_idx, "bt N_fine c -> bt N_fine z c", z=Z, c=2)
        z_vox_idx = torch.arange(Z, device=device)
        z_vox_idx = repeat(
            z_vox_idx, "z -> bt N_fine z 1", N_fine=xy_vox_idx.size(1), bt=b * t
        )
        vox_idx = torch.cat([xy_vox_idx, z_vox_idx], dim=-1).to(dtype=torch.int32)
        vox_idx = rearrange(vox_idx, "bt N_fine z c -> bt c (N_fine z)", c=3)

        # Corresponding points
        vox_coords = vox_idx / torch.tensor([X - 1, Y - 1, Z - 1], device=device).view(
            1, 3, 1
        )
        scale = torch.tensor(
            [sb[1] - sb[0], sb[3] - sb[2], sb[5] - sb[4]], device=device
        ).view(1, 3, 1)
        dist = torch.tensor([abs(sb[0]), abs(sb[2]), abs(sb[4])], device=device).view(
            1, 3, 1
        )
        vox_coords = vox_coords * scale - dist

        # Out
        dict_vox = {
            "vox_coords": rearrange(vox_coords, "b c (xy z) -> b c xy 1 z", z=Z),
            "vox_idx": rearrange(vox_idx, "b c (xy z) -> b c xy 1 z", z=Z),
        }
        return dict_vox

    def _get_fine_mask(
        self,
        out_score,
        flat_idx,
        X_Y: Tuple[int, int],
        N_anchor: str | int,
    ):
        """Initialize mask indexes as either top N_anchor or all points above a threshold.

        Args:
            - N_anchor: can be either a number indicating the number of anchor points or a tag indicating we keep
            points above 0.5 corresponding to the positive class in BCE.
        """
        # Alias
        device = out_score.device
        X, Y = X_Y
        bt, hw, _ = flat_idx.shape

        # Keep all important points.
        if N_anchor == "dyna":
            out_score_flat = torch.nonzero(out_score > self.fine_thresh)
            indices = out_score_flat[:, 0] * X * Y + out_score_flat[:, 1]

            mask = torch.zeros((bt * X * Y), device=device)
            mask = torch.scatter(mask, 0, indices.long(), 1)
            mask = mask.view(bt, X, Y)

        # Keep top N_anchor points.
        else:
            out_idx = out_score.topk(k=N_anchor, dim=1, largest=True).indices
            out_idx = rearrange(out_idx, "bt N -> (bt N)")
            batch_idx = torch.arange(bt, device=device).repeat_interleave(N_anchor)

            # Offset correction
            out_idx = batch_idx * prod(X_Y) + out_idx

            A, B = flat_idx, out_idx
            A = rearrange(A, "bt N c -> (bt N) c", c=2)
            xy_vox_idx = torch.index_select(A, 0, B)
            xy_vox_idx = rearrange(xy_vox_idx, "(bt N) c -> bt N c", c=2, bt=bt)

            # Convolutional kernel
            mask = torch.zeros((bt, X * Y), device=device)
            mask = torch.scatter(mask, 1, xy_vox_idx[..., 1].long(), 1)
            mask = mask.view(bt, X, Y)

        mask = rearrange(mask, "bt X Y -> bt 1 X Y")
        return mask
