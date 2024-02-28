"""
Geometric related utils.
Author: Loick Chambon

Adapted from:
Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE at https://github.com/nv-tlabs/lift-splat-shoot.
Authors: Jonah Philion and Sanja Fidler
"""

import numpy as np
import torch
from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation as R


def gen_dx_bx(xbound, ybound, zbound):
    dx = torch.Tensor([row[2] for row in [xbound, ybound, zbound]])
    bx = torch.Tensor([row[0] + row[2] / 2.0 for row in [xbound, ybound, zbound]])

    nx = torch.LongTensor(
        [(row[1] - row[0]) / row[2] for row in [xbound, ybound, zbound]]
    )
    return dx, bx, nx


def invert_homogenous(mat):
    R = mat[:3, :3]
    t = mat[:3, 3]

    out = np.eye(4)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


def get_yawtransfmat_from_mat(mat):
    yaw = Quaternion._from_matrix(mat[:3, :3]).yaw_pitch_roll[0]
    rot = Quaternion(
        scalar=np.cos(yaw / 2), vector=[0, 0, np.sin(yaw / 2)]
    ).rotation_matrix
    trans = mat[:3, -1]

    mat_yaw = np.eye(4)
    mat_yaw[:3, :3] = rot
    mat_yaw[:3, -1] = trans
    return mat_yaw


def from_corners_to_chw(bbox):
    center = np.mean(bbox, axis=0)
    len1 = np.linalg.norm(bbox[0, :] - bbox[1, :])
    len2 = np.linalg.norm(bbox[1, :] - bbox[2, :])
    return (center, len1, len2)


def get_random_ref_matrix(coeffs):
    """
    Use scipy to create a random reference transformation matrix.
    """
    coeffs = coeffs["trans_rot"]
    trans_coeff, rot_coeff = coeffs[:3], coeffs[3:]

    # Initialize in homogeneous coordinates.
    mat = np.eye(4, dtype=np.float64)

    # Translate
    mat[:3, 3] = (np.random.random((3)).astype(np.float32) * 2 - 1) * np.array(
        trans_coeff
    )

    # Rotate
    random_zyx = (np.random.random((3)).astype(np.float32) * 2 - 1) * np.array(
        rot_coeff
    )
    mat[:3, :3] = R.from_euler("zyx", random_zyx, degrees=True).as_matrix()

    return mat


class GeomScaler:
    def __init__(self, grid, as_tensor=False):
        """Class containing scaling functions from:
        - spatial -> spatial scaled scaling : [-50,50]m -> [-1,1]
        - spatial -> image scaling          : [-50,50]m -> [0,200]px
        - image   -> spatial scaling        : [0,200]px -> [-50,50]m
        - scaled  -> image scaling          : [-1,1]    -> [0,200]px

        Args:
            grid (Dict[str, List[int]]): grid parameters.
        """
        dx, bx, nx = gen_dx_bx(grid["xbound"], grid["ybound"], grid["zbound"])
        if not as_tensor:
            dx, bx, nx = dx.numpy(), bx.numpy(), nx.numpy()
        self.dx, self.bx, self.nx = dx, bx, nx
        return

    def _to_device_(self, device):
        self.dx, self.bx, self.nx = [x.to(device) for x in [self.dx, self.bx, self.nx]]

    def pts_from_spatial_to_scale(self, points):
        """x/50: [-50,50] -> [-1,1]"""
        return points / (-self.bx[:2] + self.dx[:2] / 2.0)

    def pts_from_spatial_to_img(self, points):
        """x+50)/0.5: [-50,50] -> [0,200]"""
        out = (points - self.bx[:2] + self.dx[:2] / 2.0) / self.dx[:2]
        return out
