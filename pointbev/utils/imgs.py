"""
Image related-utils.
Author: Loick Chambon

Adapted from:
Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE at https://github.com/nv-tlabs/lift-splat-shoot.
Authors: Jonah Philion and Sanja Fidler
"""

import cv2
import numpy as np
import torch
import torchvision
from PIL import Image
from PIL.ImageTransform import AffineTransform
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import affine, resize
from torchvision.utils import draw_keypoints


class NormalizeInverse(torchvision.transforms.Normalize):
    #  https://discuss.pytorch.org/t/simple-way-to-inverse-transform-normalization/4821/8
    def __init__(self, mean, std):
        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        std_inv = 1 / (std + 1e-7)
        mean_inv = -mean * std_inv
        super().__init__(mean=mean_inv, std=std_inv)

    def __call__(self, tensor):
        return super().__call__(tensor.clone())


DENORMALIZE_IMG = torchvision.transforms.Compose(
    (
        NormalizeInverse(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        torchvision.transforms.ToPILImage(),
    )
)

NORMALIZE_IMG = torchvision.transforms.Compose(
    (
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
    )
)

TO_TENSOR = torchvision.transforms.Compose((torchvision.transforms.ToTensor(),))


def get_affinity_matrix_from_augm(
    scale, crop_sky, crop_zoom, flip, rotate, final_dims, W_H=(1600, 900)
):
    res = list(W_H)

    affine_mat = np.eye(3)
    # Resize scaling factor.
    affine_mat[:2, :2] *= scale
    # Update res.
    res = [_ * scale for _ in res]

    # Centered crop zoom.
    w, h = final_dims
    affine_mat[0, :2] *= w / (crop_zoom[2] - crop_zoom[0])
    affine_mat[1, :2] *= h / (crop_zoom[3] - crop_zoom[1])
    affine_mat[0, 2] += (w - res[0] * w / (crop_zoom[2] - crop_zoom[0])) / 2
    affine_mat[1, 2] += (
        h - (res[1] + crop_sky) * h / (crop_zoom[3] - crop_zoom[1])
    ) / 2

    # Flip
    if flip:
        flip_mat = np.eye(3)
        flip_mat[0, 0] = -1
        flip_mat[0, 2] += w
        affine_mat = flip_mat @ affine_mat

    # Rotate
    theta = -rotate * np.pi / 180
    cos_theta, sin_theta = np.cos(theta), np.sin(theta)
    x, y = w / 2, h / 2
    rot_center_mat = np.array(
        [
            [cos_theta, -sin_theta, -x * cos_theta + y * sin_theta + x],
            [sin_theta, cos_theta, -x * sin_theta - y * cos_theta + y],
            [0, 0, 1],
        ]
    )
    affine_mat = rot_center_mat @ affine_mat
    return affine_mat


def update_intrinsics(intrins, ratio_scale):
    """
    Parameters
    ----------
        intrins: torch.Tensor (3, 3)

        | fx | 0  | cx |
        |  0 | fy | cy |
        |  0 | 0  | 1  |
    """
    new_intrins = intrins.clone()
    # Adjust intrinsics scale due to resizing
    new_intrins[..., 0, [0, 2]] *= ratio_scale
    new_intrins[..., 1, [1, 2]] *= ratio_scale

    return new_intrins


def prepare_img_axis(img, to_cam_ref=False):
    """To (X:up, Y:left)"""
    if not to_cam_ref:
        return img.transpose(-2, -1).flip(-2, -1)
    return img


def prepare_to_render_bbox_egopose(bbox_egopose, w_connect=False, size=(200, 200)):
    # Initialize a torch image as zero
    img = torch.zeros((3, *size)).to(torch.uint8)

    # Define the keypoints to draw
    keypoints = bbox_egopose
    # Only the center of the egocar.
    keypoints = keypoints.mean(dim=1, keepdim=True)

    if w_connect:
        connect = [(i, i + 1) for i in range(len(keypoints) - 1)]
    else:
        connect = None

    # Draw the keypoints on the image
    img_with_keypoints = draw_keypoints(
        img, keypoints.permute(1, 0, 2), connectivity=connect, colors="white", radius=1
    )

    # Convert to float and remove RGB.
    return img_with_keypoints[0] / 255.0


# -----------------------#
# Classes
# -----------------------#
class ImageLoader:
    """Load image using different libraries.
    PIL, opencv and turbojpeg are supported.
    """

    def __init__(self, mode="PIL", kwargs={}):
        self.mode = mode

        if mode == "turbojpeg":
            try:
                from turbojpeg import TurboJPEG

                turbojpeg = TurboJPEG()
                # Convert listconfig to dict in order to use tuples.
                kwargs = dict(kwargs)
                # hydra does not support tuples while turbojpeg uses frozenset.
                if "scaling_factor" in kwargs.keys():
                    kwargs["scaling_factor"] = tuple(kwargs["scaling_factor"])
                self.opener = lambda x: ImageLoader.turbojpeg_opener(
                    turbojpeg, x, kwargs
                )
            except:
                print('Fail to import "turbojpeg", switch to PIL.')
                self.opener = lambda x: Image.open(x)

        elif mode == "PIL":
            self.opener = lambda x: Image.open(x)

        elif mode == "PIL_optimized":
            self.opener = lambda x: ImageLoader.pil_optimized_opener(x, kwargs)

        elif mode == "opencv":
            self.opener = lambda x: ImageLoader.opencv_opener(x)
        else:
            raise NotImplementedError("Unsupported image loader mode: {}".format(mode))

    def from_array(self, array):
        if self.mode == "opencv":
            return cv2.cvtColor(array, cv2.COLOR_BGR2RGB)
        elif "PIL" in self.mode:
            return Image.fromarray(array)

    @staticmethod
    def pil_optimized_opener(filename, kwargs):
        img = Image.open(filename)
        img.draft("RGB", **kwargs)
        return img

    @staticmethod
    def turbojpeg_opener(turbojpeg, filename, kwargs):
        in_file = open(filename, "rb")
        img = turbojpeg.decode(in_file.read(), **kwargs)
        in_file.close()
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    @staticmethod
    def opencv_opener(filename):
        img = cv2.imread(str(filename), cv2.IMREAD_UNCHANGED)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def __call__(self, filename):
        return self.opener(filename)


class ImagePreProcessor:
    def __init__(self, mode="PIL"):
        if mode in ["PIL", "PIL_optimized"]:
            self.preprocess = self.pil_preprocess
            self.preprocess_affine = self.pil_preprocess_from_affine_mat
        elif mode in ["turbojpeg", "opencv"]:
            self.preprocess_affine = self.cv2_preprocess_from_affine_mat
        else:
            raise NotImplementedError(
                "Unsupported image preprocessor mode: {}".format(mode)
            )
        return

    def pil_preprocess(
        self, img, resize_dims, crop, flip, rotate, crop_zoom, final_dims
    ):
        # Resize scaling factor.
        img = img.resize(resize_dims)  # [270,480]
        # Remove the sky
        img = img.crop(crop)  # [224,480]
        # Zoom
        img = img.crop(crop_zoom)
        # Arange to final dim.
        img = img.resize(final_dims)
        # Flip
        if flip:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
        # Rotate
        img = img.rotate(rotate)
        return img

    def cv2_preprocess_from_affine_mat(self, img, affine_mat, final_dims):
        return cv2.warpAffine(img, affine_mat[:2, :3], final_dims)

    def pil_preprocess_from_affine_mat(self, img, affine_mat, final_dims):
        inv_mat = np.linalg.inv(affine_mat)
        img = img.transform(
            size=tuple(final_dims), method=AffineTransform(inv_mat[:2].ravel())
        )
        return img

    def __call__(self, img, from_affine=True, *args, **kwargs):
        if from_affine:
            return self.preprocess_affine(img, *args, **kwargs)
        else:
            return self.preprocess(img, *args, **kwargs)


# -----------------------#
# HDMaps
# -----------------------#
def get_patch_box_from_trans(trans_xy, margin):
    x_min = trans_xy[0] - margin
    x_max = trans_xy[0] + margin
    y_min = trans_xy[1] - margin
    y_max = trans_xy[1] + margin
    x_size = x_max - x_min
    y_size = y_max - y_min
    patch_box_ego = (
        int(x_min + 0.5 * (x_max - x_min)),
        int(y_min + 0.5 * (y_max - y_min)),
        int(y_size),
        int(x_size),
    )
    return patch_box_ego


def get_current_map_mask(
    map_mask, patch_angle, tw, th, orig_margin=150, final_shape=(200, 200)
):
    # Alias,
    H, W = map_mask.shape[:2]

    # Prepare
    tens_aff = torch.from_numpy(map_mask)
    tens_aff = tens_aff.permute(
        2, 0, 1
    )  # because we used np.transpose during dataset creation.

    # Translate: from orig to dest
    tens_aff = affine(
        img=tens_aff, angle=0, translate=(-tw, -th), scale=1.0, shear=[0.0, 0.0]
    )

    # Rotate
    # Note: translate corresponds to post-rotation. In our case, we want pretranslate.
    tens_aff = affine(
        img=tens_aff, angle=patch_angle, translate=(0, 0), scale=1.0, shear=[0.0, 0.0]
    )

    tens_aff = tens_aff.permute(1, 2, 0)
    tens_aff = tens_aff[
        (H - orig_margin) // 2 : orig_margin + (H - orig_margin) // 2,
        (W - orig_margin) // 2 : orig_margin + (W - orig_margin) // 2,
        :,
    ]

    tens_aff = resize(
        tens_aff.permute(2, 0, 1),
        final_shape,
        antialias=True,
        interpolation=InterpolationMode.NEAREST,
    )

    return tens_aff.permute(1, 2, 0)
