""" 
Author: Loick Chambon

Project points from 3D points to 2D images.
"""

from typing import Dict, List

import torch
from einops import rearrange, repeat
from torch import nn

from pointbev.utils.debug import debug_hook


class CamProjector(nn.Module):
    def __init__(
        self,
        spatial_bounds: List[float] = [-49.75, 49.75, -49.75, 49.75, -3.375, 5.375],
        voxel_ref="spatial",
        z_value_mode: str = "zero",
    ):
        super().__init__()
        self.register_forward_hook(debug_hook)
        assert z_value_mode in ["zero", "contract", "affine", None]
        self.z_value_mode = z_value_mode

        self.spatial_bounds = spatial_bounds

    def _stats(self, voxels):
        print(f"Max: {voxels.max().item():.4f}")
        print(f"Min: {voxels.min().item():.4f}")
        print(f"Mean: {voxels.mean().item():.4f}")
        print(f"Shape: {voxels.shape}")

    def _set_axis(self, vox_coords):
        """Deduce axis parameters: spatial (X,Y,Z) and camera (X_cam,Y_cam,Z_cam)."""
        X, Y, Z = vox_coords.shape[-3:]
        self.X, self.Y, self.Z = X, Y, Z
        self.X_cam, self.Y_cam, self.Z_cam = Y, Z, X

    # Voxel to cams
    def from_voxel_ref_to_cams(self, vox_coords, rots, trans, bev_aug, egoTin_to_seq):
        """Project points from voxel reference to camera reference.
        Args:
            - rots, trans: map points from cameras to ego. In Nuscenes, extrinsics
            are inverted compared to standard conventions. They map sensors to ego.

        Returns:
            - Voxel camera coordinates: coordinates of the voxels in the camera reference frame.
            - Voxel coordinates: coordinates of the voxels in the ego (sequence and augmentation) reference frame.
        """
        vox_coords = self.from_spatial_to_seqaug(vox_coords, bev_aug, egoTin_to_seq)
        voxcam_coords = self.from_spatial_to_cams(vox_coords, rots, trans)
        return voxcam_coords, vox_coords

    def from_spatial_to_seqaug(self, vox_coords, bev_aug, egoTin_to_seq):
        """Map points from spatial reference frame to augmented reference frame.

        Decomposition:
            - ego to egoseq: (R0, T0)
            - egoseq to bevaug: (R1, T1)
        """
        # Prepare vox_coords
        vox_coords = rearrange(vox_coords, "bt i x y z -> bt i (x y z)", i=3)

        # Apply: egoTin_to_seq.
        egoTin_to_seq = torch.linalg.inv(
            repeat(egoTin_to_seq, "b t i j -> (b t) i j", i=4, j=4)
        )

        # Apply: bev_aug.
        bev_aug = repeat(bev_aug, "b t i j -> (b t) i j", i=4, j=4)

        vox_coords = torch.cat([vox_coords, torch.ones_like(vox_coords[:, :1])], dim=1)
        vox_coords_aug = torch.bmm(bev_aug, torch.bmm(egoTin_to_seq, vox_coords))
        return vox_coords_aug[:, :3]

    def from_spatial_to_cams(self, vox_coords, rots, trans):
        """
        Map points from augmented reference frame to camera reference frame.

        Decomposition:
            - ego to cameras: (R2^-1, -R2^-1 @ T2)

        Formula: spatial to cameras:
            - Rotation: R2^-1 @ R1
            - Translation: R2^-1 @ (T1 - T2)
        """
        # Alias
        bt, n, *_ = rots.shape

        # Prepare: from cameras to ego.
        homog_mat = torch.eye(4, device=rots.device).repeat(bt * n, 1, 1)
        homog_mat[:, :3, :3] = rots.flatten(0, 1)
        homog_mat[:, :3, -1:] = trans.flatten(0, 1)
        homog_mat = torch.linalg.inv(homog_mat)

        # Apply: camera transformations.
        vox_coords = torch.cat([vox_coords, torch.ones_like(vox_coords[:, :1])], dim=1)
        vox_coords = repeat(vox_coords, "bt i Npts -> (bt n) i Npts", n=n, i=4)
        voxcam_coords = torch.bmm(homog_mat, vox_coords)[:, :3]
        return rearrange(voxcam_coords, "(bt n) i Npts -> bt n i Npts", bt=bt, n=n, i=3)

    # Cams to pixels
    def from_cameras_to_pixels(self, voxels, intrins):
        """Transform points from camera reference frame to image reference frame."""
        # Alias
        bt, n, i, j = intrins.shape

        intrins = rearrange(intrins, "bt n i j -> (bt n) i j", bt=bt, n=n, i=i, j=j)
        voxels = rearrange(voxels, "bt n i Npts -> (bt n) i Npts", bt=bt, n=n, i=3)
        voxels = torch.bmm(intrins, voxels)
        return rearrange(voxels, "(bt n) i Npts -> bt n i Npts", bt=bt, n=n, i=3)

    # Valid points
    def normalize_z_cam(self, voxels, eps=1e-6):
        """By convention, the Z_cam-coordinate on image references is equal to 1, so we rescale X,Y such
        that their Z equals 1."""
        normalizer = voxels[..., 2:3, :].clip(min=eps)
        return voxels / normalizer

    def valid_points_in_pixels(self, voxels, img_res):
        """Since we will interpolate with align corner = False, we consider only points
        inside empty circle in https://discuss.pytorch.org/t/what-we-should-use-align-corners-false/22663)

        Args:
            - Voxels: in image reference frame. (B,T,N,3,N_pts) where N_pts = X_cam*Y_cam*Z_cam.
        """
        # Alias
        H, W = img_res

        x_valid = (voxels[..., 0, :] > +0.5) & (voxels[..., 0, :] < W - 0.5)
        y_valid = (voxels[..., 1, :] > +0.5) & (voxels[..., 1, :] < H - 0.5)
        return x_valid, y_valid

    def valid_points_in_cam(self, voxels):
        """Points are valid in camera reference, if they are forward the Z_cam-axis."""
        return (voxels[..., -1, :] > 0.0).bool()

    # Prepare VT
    def normalize_vox(self, voxels, img_res, clamp_extreme=True):
        """
        Since we will interpolate with align corner = False, we need to map [0.5, W-0.5] to [-1,1].

        Note: z is supposed to be 1, after normalization, and the output should have a z equals to 0.
        """
        # Alias
        H, W = img_res
        device = voxels.device

        denom = rearrange(
            torch.tensor([W - 1, H - 1, 2], device=device), "i -> 1 1 i 1", i=3
        )
        add = rearrange(
            torch.tensor([(1 - W) / 2, (1 - H) / 2, 0], device=device),
            "i -> 1 1 i 1",
            i=3,
        )
        sub = rearrange(
            torch.tensor([1 / (W - 1), 1 / (H - 1), 0], device=device),
            "i -> 1 1 i 1",
            i=3,
        )
        voxels = 2.0 * ((voxels + add) / denom) - sub

        if clamp_extreme:
            voxels = voxels.clamp(-2, 2)
        return voxels

    def modify_z_value(self, voxels, z_before_norm, z_value_mode: bool = True):
        """Either set z to zero or adapt z to be in [-1,1] using the MIP-NeRF contraction."""
        if z_value_mode != "zero":
            zmin, zmax = self.spatial_bounds[:2]
            z_before_norm = z_before_norm / (max(abs(zmin), abs(zmax)) * 1.4142135)

        # Fix the last coordinates to zero.
        if z_value_mode == "zero":
            voxels = torch.cat(
                [voxels[..., :2, :], torch.zeros_like(voxels[..., :1, :])], dim=-2
            )

        # Contract z: [-1,1]
        elif z_value_mode == "contract":
            voxels[:, :, 2:3] = (
                torch.where(
                    z_before_norm.abs() <= 1,
                    z_before_norm,
                    (2 - 1 / z_before_norm.abs())
                    * (z_before_norm / z_before_norm.abs()),
                )
                - 1
            )

        # Affine transformation: [-1,1]
        elif z_value_mode == "affine":
            voxels[:, :, 2:3] = z_before_norm * 2 - 1
        return voxels

    def arange_voxels(self, voxcam_coords, vox_valid, vox_coords, b_t_n):
        """Arange shapes and normalize vox_coords in [-1,1]."""
        # Alias
        b, t, n = b_t_n

        list_out = []
        for _, i in zip([voxcam_coords, vox_valid], [3, 1]):
            list_out.append(
                rearrange(
                    _,
                    "(b t) n i (zcam xcam ycam) -> b t n zcam ycam xcam i",
                    b=b,
                    t=t,
                    n=n,
                    i=i,
                    zcam=self.Z_cam,
                    xcam=self.X_cam,
                    ycam=self.Y_cam,
                )
            )

        vox_coords = rearrange(
            vox_coords,
            "(b t) i (zcam xcam ycam) -> b t zcam ycam xcam i",
            b=b,
            t=t,
            zcam=self.Z_cam,
            xcam=self.X_cam,
            ycam=self.Y_cam,
            i=3,
        )

        # Normalize vox coords for GS embedding.
        # 1.2 usefull to avoid border effects when having seqaug matrix.
        XMIN, XMAX, YMIN, YMAX, ZMIN, ZMAX = self.spatial_bounds
        vox_coords = vox_coords / torch.tensor(
            [
                1.2 * max(abs(XMIN), abs(XMAX)),
                1.2 * max(abs(YMIN), abs(YMAX)),
                1.2 * max(abs(ZMIN), abs(ZMAX)),
            ],
            device=vox_coords.device,
            dtype=vox_coords.dtype,
        )
        list_out.append(vox_coords)

        return list_out

    # Forward
    def forward(self, dict_mat, dict_shape, dict_vox) -> Dict[str, torch.Tensor]:
        # Unpack
        rots, trans, intrins, bev_aug, egoTin_to_seq = (
            dict_mat["rots"],
            dict_mat["trans"],
            dict_mat["intrins"],
            dict_mat["bev_aug"],
            dict_mat["egoTin_to_seq"],
        )
        vox_coords = dict_vox.get("vox_coords", None)

        # Alias
        (b, n, t) = [dict_shape[k] for k in ["b", "n", "t"]]
        img_feats_res = (dict_shape["Hfeats"], dict_shape["Wfeats"])

        # Set axis range.
        self._set_axis(vox_coords)

        # Ego to cams.
        voxcam_coords, vox_coords = self.from_voxel_ref_to_cams(
            vox_coords,
            rots,
            trans,
            bev_aug,
            egoTin_to_seq,
        )
        z_valid = self.valid_points_in_cam(voxcam_coords)

        # Cams to pixels.
        voxcam_coords = self.from_cameras_to_pixels(voxcam_coords, intrins)
        if self.z_value_mode != "zero":
            z_before_norm = voxcam_coords[:, :, 2:3]  # Get z before normalization.
        else:
            z_before_norm = None
        voxcam_coords = self.normalize_z_cam(voxcam_coords)
        x_valid, y_valid = self.valid_points_in_pixels(voxcam_coords, img_feats_res)

        # # Filter valid points.
        vox_valid = (x_valid & y_valid & z_valid).unsqueeze(-2)
        voxcam_coords = self.normalize_vox(voxcam_coords, img_feats_res)

        # Adapt z:
        voxcam_coords = self.modify_z_value(
            voxcam_coords, z_before_norm, self.z_value_mode
        )

        # Get features in img space.
        voxcam_coords, vox_valid, vox_coords = self.arange_voxels(
            voxcam_coords, vox_valid, vox_coords, (b, t, n)
        )
        return dict(
            {
                "voxcam_coords": voxcam_coords,
                "vox_valid": vox_valid,
                "vox_coords": vox_coords,
            }
        )
