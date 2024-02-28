from typing import Dict, Optional

import torch
from einops import rearrange, repeat
from torch import Tensor, nn

from pointbev.models.projector import CamProjector
from pointbev.utils.imgs import update_intrinsics


class Network(nn.Module):
    def __init__(
        self,
        # Modules
        backbone=None,
        neck=None,
        projector: CamProjector = None,
        view_transform=None,
        autoencoder: Optional[nn.Module] = None,
        temporal=None,
        heads=None,
        # Configs
        in_c={},
        out_c={},
        in_shape={},
        voxel_ref="spatial",
        init_coordselec=True,
    ):
        super().__init__()
        self.backbone = backbone
        self.downsample = self.backbone.downsample

        self.neck = neck
        self.projector = projector
        self.view_transform = view_transform
        self.decoder = autoencoder
        self.temporal = temporal
        self.heads = heads

        self.num_pass = 1
        if init_coordselec:
            self.coord_selector = CoordSelector(in_shape, voxel_ref)
        else:
            self.coord_selector = False
        return

    def _init_dict_shape(self, imgs, egoTin_to_seq):
        """Prepare a dictionary containing indicators of the shape of the input.

        Keys:
            - b: batch size
            - t: time index
            - n: number of cameras
            - nq: number of different BEV
            - Hinit: initial image height
            - Winit: initial image width
        """
        b, t, n, _, h, w = imgs.shape
        nq = egoTin_to_seq.size(1)
        assert egoTin_to_seq.size(0) == b
        assert egoTin_to_seq.size(2) == 4
        assert egoTin_to_seq.size(3) == 4
        dict_shape = {
            "b": b,
            "t": t,
            "n": n,
            "nq": nq,
            "Hinit": h,
            "Winit": w,
            "Hfeats": None,
            "Wfeats": None,
        }
        return dict_shape

    def _init_dict_mat(self, rots, trans, intrins, bev_aug, egoTin_to_seq):
        """Prepare a dictionary containing transformation matrices.

        Keys:
            - rots: rotation matrices from cam to ego.
            - trans: translation vectors from cam to ego.
            - intrins: intrinsic matrices from cam to image.
            - bev_aug: augmentation matrices to move the BEV around the ego position.
        """
        return {
            "rots": rots,
            "trans": trans,
            "intrins": intrins,
            "bev_aug": bev_aug,
            "egoTin_to_seq": egoTin_to_seq,
        }

    def _init_dict_vox(self):
        """Prepare a dictionary containing info on the voxels.

        Keys:
            - vox_feats: voxel features
            - vox_valid: voxel valid mask.
            - vox_coords: voxel coordinates.
            - voxcam_coords: voxel camera coordinates.
            - vox_idx: voxel indices.
        """
        # https://stackoverflow.com/questions/73035588/python-dictionary-with-fixed-keys-contents-and-variable-arguments
        return {
            "vox_feats": None,
            "vox_valid": None,
            "vox_coords": None,
            "voxcam_coords": None,
            "vox_idx": None,
        }

    # Common begining
    def _common_init_backneck_prepare_vt(
        self, imgs, rots, trans, intrins, bev_aug, egoTin_to_seq
    ):
        """Avoid redundancy accross files and models.

        Shared among models:
        - dictionary initialization,
        - forward backbone and neck,
        - preparation of view transformation,
        - extension of the input for the temporal models.
        """
        # Dict shape and vox.
        dict_shape = self._init_dict_shape(imgs, egoTin_to_seq)
        dict_vox = self._init_dict_vox()

        # Extract image features.
        img_feats = self.forward_backneck(imgs)
        dict_shape["Hfeats"] = img_feats.size(-2)
        dict_shape["Wfeats"] = img_feats.size(-1)
        dict_img = {"img_feats": img_feats}

        # Prepare VT
        rots, trans, intrins = self._prepare_view_transform(rots, trans, intrins)
        intrins = update_intrinsics(intrins, 1 / self.downsample)
        dict_mat = self._init_dict_mat(rots, trans, intrins, bev_aug, egoTin_to_seq)

        return dict_shape, dict_vox, dict_img, dict_mat

    # Backneck
    def _prepare_backneck(self, imgs):
        """Fuse batch, time and camera dimensions."""
        imgs = rearrange(imgs, "b t n c h w -> (b t n) c h w")
        return imgs

    def _arrange_backneck(self, btn, img_feats):
        """Split camera dimension, keep batch and time."""
        b, t, n = btn
        img_feats = rearrange(
            img_feats, "(b t n) c h w -> (b t) n c h w", b=b, t=t, n=n
        )
        return img_feats

    def forward_backneck(self, imgs):
        # Backbone and Neck
        btn = imgs.shape[:3]
        imgs = self._prepare_backneck(imgs)
        imgs_feats = self.neck(self.backbone(imgs))
        imgs_feats = self._arrange_backneck(btn, imgs_feats)
        return imgs_feats

    # VT
    def _prepare_dict_vox(self, dict_vox, dict_shape):
        bt = dict_shape["b"] * dict_shape["t"]
        dict_vox.update(self.coord_selector._get_dense_coords(bt))
        return

    def _prepare_view_transform(self, rots, trans, intrins):
        """Fuse batch and time dimensions."""
        b, t, n, _, _ = rots.shape
        rots = rearrange(rots, "b t n i j -> (b t) n i j", i=3, j=3)
        trans = rearrange(trans, "b t n i 1 -> (b t) n i 1", b=b, t=t, n=n, i=3)
        intrins = rearrange(intrins, "b t n i j -> (b t) n i j", i=3, j=3)
        return rots, trans, intrins

    # Utils
    def _fuse_b_nq(self, bev_feats):
        """Fuse batch and nq dimensions.

        Shape: (b, nq, c, h, w) -> (b * nq, c, h, w)
        """
        b, nq, c, h, w = bev_feats.shape
        bev_feats = rearrange(bev_feats, "b nq c h w -> (b nq) c h w", b=b, nq=nq)
        return bev_feats

    def _split_b_nq(self, bev_feats, b_nq):
        """Split batch and nq dimensions.
        Shape: (b * nq, c, h, w) -> (b, nq, c, h, w)
        """
        b, nq = b_nq
        return rearrange(bev_feats, "(b nq) c h w -> b nq c h w", b=b, nq=nq)

    # Decoder
    def _prepare_decoder(self, bev_feats):
        return self._fuse_b_nq(bev_feats)

    def _arrange_decoder(self, bev_feats, b_nq):
        return self._split_b_nq(bev_feats, b_nq)

    def forward_decoder(self, bev_feats, mask: Optional[torch.Tensor] = None):
        # Decoder
        b_nq = bev_feats.shape[:2]
        bev_feats = self._prepare_decoder(bev_feats)

        if mask is not None:
            mask = self._prepare_decoder(mask)
            bev_feats = self.decoder(bev_feats, mask)
        else:
            bev_feats = self.decoder(bev_feats)

        return self._arrange_decoder(bev_feats, b_nq)

    # Temporal
    def _prepare_temporal(self, bev_feats):
        return bev_feats

    def _arrange_temporal(self, bev_feats, b_nq):
        return bev_feats

    def forward_temporal(self, bev_feats):
        if not self.temporal:
            return bev_feats

        # Alias
        b_nq = bev_feats.shape[:2]
        bev_feats = self._prepare_temporal(bev_feats)
        bev_feats = self.temporal(bev_feats)
        return self._arrange_temporal(bev_feats, b_nq)

    # Heads
    def _prepare_heads(self, bev_feats):
        return self._fuse_b_nq(bev_feats)

    def _arrange_heads(self, dict_out, b_nq):
        """Split batch and nq dimensions."""
        b, nq = b_nq
        for k in dict_out.keys():
            dict_out[k] = rearrange(
                dict_out[k], "(b nq) c h w -> b nq c h w", b=b, nq=nq
            )
        return dict_out

    def forward_heads(self, bev_feats):
        b, nq = bev_feats.shape[:2]
        bev_feats = self._prepare_heads(bev_feats)
        dict_out = self.heads(bev_feats)
        return self._arrange_heads(dict_out, (b, nq))

    def forward(self, imgs, rots, trans, intrins, bev_aug, egoTin_to_seq, **kwargs):
        (
            dict_shape,
            dict_vox,
            dict_img,
            dict_mat,
        ) = self._common_init_backneck_prepare_vt(
            imgs, rots, trans, intrins, bev_aug, egoTin_to_seq
        )

        self._prepare_dict_vox(dict_vox, dict_shape)

        dict_vox.update(self.projector(dict_mat, dict_shape, dict_vox))
        bev_feats, *_ = self.view_transform(dict_img["img_feats"], dict_vox)

        # Decoder
        bev_feats = self.forward_decoder(bev_feats)

        # Temporal
        bev_feats = self.forward_temporal(bev_feats)

        # Heads
        dict_out = self.forward_heads(bev_feats)
        return {"bev": dict_out}


class CoordSelector(nn.Module):
    def __init__(self, spatial_kwargs, voxel_ref, init_buffer=True):
        super().__init__()
        self.spatial_bounds = spatial_kwargs["spatial_bounds"]
        self.spatial_range = spatial_kwargs["projector"]

        assert voxel_ref in ["spatial", "camera"]
        self.voxel_ref = voxel_ref

        self._init_buffer() if init_buffer else None
        return

    def _init_buffer(self):
        self._set_cache_dense_coords()

    def _set_cache_dense_coords(
        self,
    ):
        """Get, and / or, set dense coordinates used during training and validation."""
        # Alias
        X, Y, Z = self.spatial_range
        XMIN, XMAX, YMIN, YMAX, ZMIN, ZMAX = self.spatial_bounds

        # Coordinates
        if self.voxel_ref == "spatial":
            # (3, rX, rY, Z), r for reverse order.
            dense_vox_coords = torch.stack(
                torch.meshgrid(
                    torch.linspace(XMIN, XMAX, X, dtype=torch.float64),
                    torch.linspace(YMIN, YMAX, Y, dtype=torch.float64),
                    torch.linspace(ZMIN, ZMAX, Z, dtype=torch.float64),
                    indexing="ij",
                )
            ).flip(1, 2)
        self.register_buffer("dense_vox_coords", dense_vox_coords.float())

        # Indices
        dense_vox_idx = torch.stack(
            torch.meshgrid(
                torch.arange(X), torch.arange(Y), torch.arange(Z), indexing="ij"
            )
        ).flip(1, 2)
        self.register_buffer("dense_vox_idx", dense_vox_idx.int())

        return

    def _get_dense_coords(self, bt: int) -> Dict[str, Tensor]:
        """Regular space division of pillars

        Returns:
            vox_coords: 3D voxels coordinates. Voxels can be grouped as regular pillars.
            vox_idx: Corresponding voxels indices.
        """
        vox_coords, vox_idx = self.dense_vox_coords, self.dense_vox_idx
        vox_coords = repeat(vox_coords, "c x y z -> bt c x y z", bt=bt)
        vox_idx = repeat(vox_idx, "c x y z -> bt c x y z", bt=bt)

        return dict(
            {
                "vox_coords": vox_coords,
                "vox_idx": vox_idx,
            }
        )
