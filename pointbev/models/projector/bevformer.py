from typing import List

import torch
from einops import rearrange

from .common import CamProjector


class BevFormerProjector(CamProjector):
    def __init__(
        self,
        spatial_bounds: List[float] = [-49.75, 49.75, -49.75, 49.75, -3.375, 5.375],
    ):
        super().__init__(spatial_bounds, voxel_ref="spatial", z_value_mode="zero")

    def valid_points_in_pixels(self, voxels, img_res):
        """
        Since we do not interpolate with gridsample, we do not need to consider border effects.
        We are in raster coordinate system so points are inside if they are in pixel [0,W-1]x[0,H-1]
        corresponding to coordinates [0,W]x[0,H].
        """
        # Alias
        H, W = img_res

        x_valid = (voxels[..., 0, :] >= 0.0) & (voxels[..., 0, :] < W).bool()
        y_valid = (voxels[..., 1, :] >= 0.0) & (voxels[..., 1, :] < H).bool()
        return x_valid, y_valid

    def normalize_vox(self, voxels, img_res):
        """
        Since we do not interpolate with gridsample, we do not need to consider border effects.
        """
        # Alias
        H, W = img_res
        bt, n, _, Npts = voxels.shape

        denom = rearrange(
            torch.tensor([W, H, 1], device=voxels.device), "i -> 1 1 i 1", i=3
        )
        # In [0,1]
        voxels = voxels / denom
        return voxels

    def forward(self, dict_mat, dict_shape, dict_vox={}):
        return super().forward(dict_mat, dict_shape, dict_vox)
