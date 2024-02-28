"""
Basic nuScenes dataloader.
Author: Loick Chambon

Adapted from:
Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE at https://github.com/nv-tlabs/lift-splat-shoot.
Authors: Jonah Philion and Sanja Fidler
"""

import json
import os
from copy import deepcopy
from math import prod
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
from einops import rearrange
from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import BoxVisibility, transform_matrix
from nuscenes.utils.splits import create_splits_scenes
from pyquaternion import Quaternion
from pytorch_lightning.utilities import rank_zero_only

from pointbev.utils.geom import (
    GeomScaler,
    from_corners_to_chw,
    gen_dx_bx,
    invert_homogenous,
)
from pointbev.utils.imgs import (
    NORMALIZE_IMG,
    TO_TENSOR,
    ImageLoader,
    ImagePreProcessor,
    get_affinity_matrix_from_augm,
    get_current_map_mask,
    get_patch_box_from_trans,
    prepare_img_axis,
)

from .lyft_common import TRAIN_LYFT_INDICES, VAL_LYFT_INDICES

# -----------------------#
# Global Parameters
# -----------------------#
IGNORE_INDEX = 255
MAP_DYNAMIC_TAG = {"parked": 0, "moving": 1, "stopped": 2, "other": 3}
VISIBILITY_TAG = {"0_40": 1, "40_60": 2, "60_80": 3, "80_100": 4, "Back": 255}
HDMAP_DICT = {
    k: i
    for i, k in enumerate(
        [
            "lane",
            "road_segment",
            "drivable_area",
            "road_divider",
            "lane_divider",
            "stop_line",
            "ped_crossing",
            "walkway",
        ]
    )
}
THRESHOLD_VALID_CENTERNESS = 0.1
CAMREF = 1
SIGMA = 3
DETECTION_CLS = {
    "movable_object.barrier",
    "vehicle.bicycle",
    "vehicle.bus.bendy",
    "vehicle.bus.rigid",
    "vehicle.car",
    "vehicle.construction",
    "vehicle.motorcycle",
    "human.pedestrian.adult",
    "human.pedestrian.child",
    "human.pedestrian.construction_worker",
    "human.pedestrian.police_officer",
    "movable_object.trafficcone",
    "vehicle.trailer",
    "vehicle.truck",
}


class NuScenesDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        nusc,
        # Lyft instead of nuscenes
        is_lyft: bool = False,
        # Mode
        is_train: bool = True,
        # Grid
        grid: Dict = {},
        # Images
        img_params: Dict = {},
        img_loader=ImageLoader(),
        normalize_img: bool = True,
        # Cameras
        to_cam_ref: bool = False,
        random_cam_ref: bool = False,
        force_camref: Optional[int] = None,
        # Lidar: found a bug using Lyft.
        keep_input_lidar: bool = False,
        # Augmentations
        coeffs={},
        # Filters
        only_object_center_in: bool = False,
        filters_cat: List[str] = [],
        plot_ego: bool = False,
        # Outputs
        hdmap_names: List[str] = [],
        keep_input_persp: bool = False,
        # Path
        hdmaproot: str = "",
    ):
        # Lyft dataset
        self.is_lyft = is_lyft
        self.nusc = nusc

        # Mode
        self.is_train = is_train

        if self.is_lyft:
            self.dataroot = self.nusc.data_path
        else:
            self.dataroot = self.nusc.dataroot
        self.scenes = self._get_scenes()
        self.ixes = self._prepro()

        # Outputs
        # -> Objects
        if not self.is_lyft:
            self.filters_cat = filters_cat
        else:
            filters_cat = [
                "bus",
                "car",
                "construction_vehicle",
                "trailer",
                "truck",
            ]
            self.filters_cat = filters_cat
        self.class_to_idx = self._init_class_mapping(filters_cat)
        # -> HDMaps
        self.hdmap_names = hdmap_names
        self.hdmap_radius = 150

        # Filters
        self.only_object_center_in = only_object_center_in
        self.plot_ego = plot_ego

        # Grid
        self.grid = grid
        *_, nx = gen_dx_bx(grid["xbound"], grid["ybound"], grid["zbound"])
        self.nx = nx.numpy()
        self.geomscaler = GeomScaler(grid)

        # Augmentations
        self.coeffs = coeffs

        # Images
        self.normalize_img = normalize_img
        self.img_params = img_params
        self.img_loader = img_loader
        self.img_processor = ImagePreProcessor(mode=img_loader.mode)

        # Cameras
        self.to_cam_ref = to_cam_ref
        self.random_cam_ref = random_cam_ref
        self.force_camref = force_camref

        # Lidar
        self.keep_input_lidar = keep_input_lidar

        # Dynamic dictionary
        self.inst_map = {}
        self.center_map = {}
        self.hdmap_map = {}

        # Dynamic tags
        self.map_dynamic_tag = MAP_DYNAMIC_TAG

        # Paths
        self.hdmaproot = hdmaproot

    @rank_zero_only
    def _print_desc(self):
        print()
        print(self)

    # Init
    def _init_class_mapping(self, filters_cat):
        """Creates a mapping from class name to class index with an additional filter to keep some classes."""
        # in category, one element corresponds to one class, so the mapping is trivial.
        list_elements = [d["name"] for d in self.nusc.category]
        list_elements = filter(
            lambda x: any([filter_c in x for filter_c in filters_cat]), list_elements
        )
        return {k: i for i, k in enumerate(list_elements)}

    def _get_scenes(self) -> List[str]:
        """Return validation or training scenes depending on the dataset mode."""
        if not self.is_lyft:
            # filter by scene split
            split = {
                "v1.0-trainval": {True: "train", False: "val"},
                "v1.0-mini": {True: "mini_train", False: "mini_val"},
            }[self.nusc.version][self.is_train]
            self.split = split
            scenes = create_splits_scenes()[split]
        else:
            scenes = [row["name"] for row in self.nusc.scene]
            indices = TRAIN_LYFT_INDICES if self.is_train else VAL_LYFT_INDICES
            scenes = [scenes[i] for i in indices]
        return scenes

    def _prepro(self):
        """Sort timestamps by scenes if they belong to the filtered dataset scene."""
        samples = [samp for samp in self.nusc.sample]

        # remove samples that aren't in this split
        samples = [
            samp
            for samp in samples
            if self.nusc.get("scene", samp["scene_token"])["name"] in self.scenes
        ]

        # sort by scene, timestamp (only to make chronological viz easier)
        samples.sort(key=lambda x: (x["scene_token"], x["timestamp"]))

        return samples

    # Inputs
    # -> Camera
    def get_camera_related_data(
        self,
        rec,
        cams: str,
        vis_level=BoxVisibility.ANY,
        keys_to_keep: List[str] = ["rots", "trans", "intrins", "imgs", "persp_imgs"],
        keep_double: bool = False,
    ):
        """Return image, and camera parameters for each camera.

        Args:
            - keep_double (bool): if True, returns rots and trans with float64 precision. It is needed
            for 'exact' matrix multiplication.
        """
        # Initialize
        imgs = []
        rots = []
        trans = []
        intrins = []
        persp_imgs = []
        keys_to_keep.sort()

        # Loop over cameras
        for cam in cams:
            cam_sample = self.nusc.get("sample_data", rec["data"][cam])

            # Parameters:
            # -> Intrinsics and extrinsics
            # Extrinsics are given with respect to the ego vehicle body frame.
            cs_cam = self.nusc.get(
                "calibrated_sensor", cam_sample["calibrated_sensor_token"]
            )
            intrin = np.array(cs_cam["camera_intrinsic"])
            rot_quat = Quaternion(cs_cam["rotation"])
            rot = torch.tensor(rot_quat.rotation_matrix, dtype=torch.float64)
            tran_np = np.array(cs_cam["translation"])
            tran = torch.tensor(tran_np, dtype=torch.float64).unsqueeze(-1)

            if keys_to_keep == ["rots", "trans"]:
                rots.append(rot)
                trans.append(tran)
                continue

            # -> Augmentations:
            (scale, _, crop, flip, rotate, crop_zoom, _) = self._sample_augmentation()

            # Image
            imgname = os.path.join(self.dataroot, cam_sample["filename"])
            img = self.img_loader(imgname)

            final_dims = list(self.img_params["final_dim"])[::-1]
            W, H = final_dims

            # -> Adjust image according to new parameter
            affine_mat = get_affinity_matrix_from_augm(
                scale, crop[1], crop_zoom, flip, rotate, final_dims, img.size
            )
            img = self.img_processor(img, True, affine_mat, final_dims)

            if self.normalize_img:
                img = NORMALIZE_IMG(img)
            else:
                img = TO_TENSOR(img)

            imgs.append(img)
            affine_intrin = torch.from_numpy(affine_mat @ intrin).float()
            intrins.append(affine_intrin)
            rots.append(rot)
            trans.append(tran)

            # Perspective segmentations
            persp_img = np.zeros((H, W), dtype=np.uint8)
            boxes = self.nusc.get_boxes(cam_sample["token"])

            pose_record = self.nusc.get(
                "ego_pose",
                self.nusc.get("sample_data", cam_sample["token"])["ego_pose_token"],
            )
            for box in boxes:
                if not any([cat in box.name for cat in self.filters_cat]):
                    continue

                # Move box to ego vehicle coord system.
                box.translate(-np.array(pose_record["translation"]))
                box.rotate(Quaternion(pose_record["rotation"]).inverse)

                #  Move box to sensor coord system.
                box.translate(-tran_np)
                box.rotate(rot_quat.inverse)
                corners_3d = box.corners()

                # Move box to image coord system.
                viewpad = np.eye(4)
                viewpad[: affine_intrin.shape[0], : affine_intrin.shape[1]] = (
                    affine_intrin
                )

                nbr_points = corners_3d.shape[1]
                points = np.concatenate((corners_3d, np.ones((1, nbr_points))))
                points = np.dot(viewpad, points)
                points = points[:3, :]

                # Normalize
                points = points / points[2:3, :].repeat(3, 0).reshape(3, nbr_points)
                corners_img = points[:2, :]

                visible = np.logical_and(corners_img[0, :] > 0, corners_img[0, :] < W)
                visible = np.logical_and(visible, corners_img[1, :] < H)
                visible = np.logical_and(visible, corners_img[1, :] > 0)
                visible = np.logical_and(visible, corners_3d[2, :] > 1)

                in_front = corners_3d[2, :] > 0.1
                if vis_level == BoxVisibility.ALL:
                    box_in_img = all(visible) and all(in_front)
                elif vis_level == BoxVisibility.ANY:
                    box_in_img = any(visible) and all(in_front)
                elif vis_level == BoxVisibility.NONE:
                    box_in_img = True
                else:
                    box_in_img = False

                if box_in_img:
                    corners = corners_img[:2, :]
                    corners = corners.T.astype(np.int32)

                    for idx in [
                        [0, 1, 2, 3],
                        [-4, -3, -2, -1],
                        [2, 3, 7, 6],
                        [0, 1, 4, 5],
                        [1, 5, 6, 2],
                        [0, 4, 7, 3],
                    ]:
                        # front, back, bottom, up, side, side
                        cv2.fillConvexPoly(persp_img, corners[idx], 1)

            persp_img = np.array(persp_img)
            persp_imgs.append(torch.from_numpy(persp_img))

        # Prepare
        if rots:
            rots = torch.stack(rots).to(
                torch.float32 if not keep_double else torch.float64
            )
        if trans:
            trans = torch.stack(trans).to(
                torch.float32 if not keep_double else torch.float64
            )
        if imgs:
            imgs = torch.stack(imgs)
        if intrins:
            intrins = torch.stack(intrins)
        if persp_imgs:
            persp_imgs = torch.stack(persp_imgs)

        # Keys
        out_dict = {}
        for k, v in zip(
            ["rots", "trans", "imgs", "intrins", "persp_imgs"],
            [rots, trans, imgs, intrins, persp_imgs],
        ):
            if k not in keys_to_keep:
                continue
            else:
                out_dict[k] = v
        return out_dict

    def _sample_augmentation(self):
        """Corresponds to get_resizing_and_cropping_parameters in the original code with some improvements.
        Available transformations:
            - scale
            - crop sky
            - crop zoom
            - final scale
            - flip
            - rotate.

        Ex: [1600,900] -> scale: 0.5 [800,450] -> crop sky: 10 [800,440] -> ...
        """
        # Specify the input image dimensions
        H, W = self.img_params["H"], self.img_params["W"]

        # During training
        if self.is_train:
            # Randomly choose a resize factor, e.g: 0.3.
            scale = np.random.uniform(*self.img_params["scale"])

            # Resize images, e.g: [270,480]
            newW, newH = int(W * scale), int(H * scale)

            # Resize.
            resize_dims = (newW, newH)

            # Crop the sky.
            crop_h = int(
                (1 - np.random.uniform(*self.img_params["crop_up_pct"])) * newH
            )
            crop = (0, crop_h, newW, newH)

            # Zoom in, zoom out: neutral=1, e.g: [0.95,1.05]
            zoom = np.random.uniform(*self.img_params["zoom_lim"])
            crop_zoomh, crop_zoomw = (
                ((newH - crop_h) * (1 - zoom)) // 2,
                (newW * (1 - zoom)) // 2,
            )
            crop_zoom = (
                -crop_zoomw,
                -crop_zoomh,
                crop_zoomw + newW,
                crop_zoomh + newH - crop_h,
            )

            # Allow flip and rotate during training.
            flip = False
            if self.img_params["rand_flip"] and np.random.choice([0, 1]):  # False
                flip = True
            rotate = np.random.uniform(*self.img_params["rot_lim"])  # ~U(0,0)
        else:
            # Randomly choose a resize factor, e.g: 0.3.
            # Images: [900,1600]
            scale = np.mean(self.img_params["scale"])

            # Resize images, e.g: [270,480]
            newW, newH = int(W * scale), int(H * scale)  # 480, 270

            # Resize.
            resize_dims = (newW, newH)

            # Remove the sky.
            crop_h = int((1 - np.mean(self.img_params["crop_up_pct"])) * newH)
            crop = (0, crop_h, newW, newH)

            # Zoom inside image.
            zoom = 1.0
            crop_zoom = (0, 0, newW, newH - crop_h)

            # Flip and rotate
            flip = False
            rotate = 0

        return scale, resize_dims, crop, flip, rotate, crop_zoom, zoom

    def get_lidar_data(self, rec, egoPout_to_global, bev_aug):
        # Alias
        h, w = self.nx[0], self.nx[1]

        # Lidar img
        lidar_img, lidar_img_aug = np.zeros((2, h, w), dtype=np.uint8)

        # LiDar
        lidar_sample = self.nusc.get("sample_data", rec["data"]["LIDAR_TOP"])

        # Lidar PC: sensor reference frame
        lidar_pts = np.fromfile(
            os.path.join(self.dataroot, lidar_sample["filename"]), dtype=np.float32
        ).reshape(-1, 5)[:, :3]
        homog_points = np.concatenate(
            [lidar_pts, np.ones_like(lidar_pts[:, :1])], 1
        ).transpose(1, 0)

        # From sensor reference frame to ego.
        calibration = self.nusc.get(
            "calibrated_sensor", lidar_sample["calibrated_sensor_token"]
        )
        sensor_to_ego = transform_matrix(
            calibration["translation"], Quaternion(calibration["rotation"])
        )
        homog_points = sensor_to_ego @ homog_points

        # From ego to global
        ref_pose = self.nusc.get("ego_pose", lidar_sample["ego_pose_token"])
        ego_to_global = transform_matrix(
            ref_pose["translation"], Quaternion(ref_pose["rotation"])
        )
        homog_points = ego_to_global @ homog_points

        # From global to ego_ref, from ego_ref to ego.
        homog_points = (invert_homogenous(egoPout_to_global)) @ homog_points
        pts_img = self._prepare_points_to_gtimg(np.eye(4), homog_points)
        pts_img_aug = self._prepare_points_to_gtimg(bev_aug, homog_points)

        for pts, img in zip([pts_img, pts_img_aug], [lidar_img, lidar_img_aug]):
            poly_region_img_rd = self.geomscaler.pts_from_spatial_to_img(pts)
            pts = (np.round(poly_region_img_rd)).astype(np.int32)

            pts = pts[
                np.where(
                    (pts[:, 0] < h)
                    & (pts[:, 1] < w)
                    & (pts[:, 0] > 0)
                    & (pts[:, 1] > 0)
                )[0]
            ]
            img[pts[:, 0], pts[:, 1]] = 1
        return lidar_img.transpose(-1, -2), lidar_img_aug.transpose(-1, -2)

    # -> BEV
    def get_bev_related_data(
        self,
        rec,
        egoPout_to_global,
        bev_aug,
    ):
        """Return BEV related data.

        Outputs:
            - binimg: (Tensor[torch.uint8]) contains bev segmentation.
            - visibility: (Tensor[torch.uint8]) contains segmentation per visibility level.
            - offsets: (Tensor[torch.float32]) contains distance of objects to the center.
            - centerness: (Tensor[torch.float32]) contains density center map of annotations.
            - bboxes: (Tensor[torch.float32]) contains bounding boxes represented as ordered polygons.
            - binimg_aug: (Tensor[torch.uint8]) contains augmented bev segmentation.
            - classes: (Tensor[torch.uint8]) contains annotated classes.
            - centers: (Tensor[torch.float32]) contains center coordinates.
        """
        # Alias
        h, w = self.nx[0], self.nx[1]

        # Initialize
        # -> Classes
        classes, classes_aug = [], []

        # -> Visibility
        visibility, visibility_aug = np.full((2, h, w), 255, dtype=np.uint8)

        # -> Mobile masks: 0: parked, 1: mobile, 2: stopped, 3: unknown
        mobility, mobility_aug = np.zeros((2, h, w), dtype=np.uint8)
        unrecognized_tag = []

        # -> Offsets
        instance, instance_aug = np.zeros((2, h, w), dtype=np.int32)
        offsets, offsets_aug = torch.full(
            (2, 2, h, w), fill_value=255.0, dtype=torch.float32
        )
        valid_centerness, valid_centerness_aug = np.ones((2, h, w), dtype=np.bool_)

        # -> Offset map
        center_bbox_on_img, center_bbox_on_img_aug = [], []

        x, y = torch.meshgrid(
            torch.arange(h, dtype=torch.float),
            torch.arange(w, dtype=torch.float),
            indexing="xy",
        )

        # -> Centerness
        centerness, centerness_aug = torch.zeros(2, 1, h, w)
        centers, centers_aug = [], []

        # -> Bounding box attributes
        bbox_attr, bbox_attr_aug = [], []

        # -> Bounding boxes
        bboxes, bboxes_aug = {}, {}
        visible_bbox = []

        # Are augmentations activated ?
        bool_aug_activated = not np.allclose(bev_aug, np.eye(4))

        egopose_token = self.nusc.get("sample_data", rec["data"]["LIDAR_TOP"])[
            f"ego_pose_token"
        ]
        inst_egopose = self.nusc.get("ego_pose", egopose_token)
        # https://forum.nuscenes.org/t/dimensions-of-the-ego-vehicle-used-to-gather-data/550
        inst_egopose["size"] = [1.73, 4.084, 1.562]
        inst_egopose["visibility_token"] = 4
        inst_egopose["dynamic_tag"] = 3
        inst_egopose["category_name"] = "vehicle.car"
        inst_egopose["attribute_tokens"] = []
        inst_egopose["instance_token"] = "ego"

        anns = rec["anns"]
        if self.plot_ego:
            anns = anns + [inst_egopose]

        min_vis = self.img_params["min_visibility"]

        # Loop over annotations
        for i, tok in enumerate(anns):
            # Given w.r.t the global coordinate system.
            is_ego = i == len(anns) - 1 and self.plot_ego
            if is_ego:
                inst = tok
            else:
                inst = self.nusc.get("sample_annotation", tok)

            # NuScenesDataset filter:
            if not any([cat in inst["category_name"] for cat in self.filters_cat]):
                continue
            if not self.is_lyft:
                # Visibility token, used for detection.
                is_visible = int(inst["visibility_token"]) >= min_vis
                visible_bbox.append(is_visible)
            else:
                is_visible = True
                visible_bbox.append(True)

            # Dynamic tag
            if len(inst["attribute_tokens"]) > 0 and (not self.is_lyft):
                assert len(inst["attribute_tokens"]) == 1
                dynamic_tag = self.nusc.get("attribute", inst["attribute_tokens"][0])[
                    "name"
                ]
                dynamic_tag = dynamic_tag.split(".")[-1]
            else:
                dynamic_tag = "other"

            if dynamic_tag in self.map_dynamic_tag.keys():
                inst["dynamic_tag"] = self.map_dynamic_tag[dynamic_tag]
            else:
                if dynamic_tag not in unrecognized_tag:
                    unrecognized_tag.append(dynamic_tag)
                    # print("Unrognized dynamic tag: ", dynamic_tag)
                inst["dynamic_tag"] = self.map_dynamic_tag["other"]

            # Update instance map
            if inst["instance_token"] not in self.inst_map.keys():
                assert (
                    len(self.inst_map) + 1 <= np.iinfo(instance.dtype).max
                ), "Can not encode more instances simultaneously due to precision."
                self.inst_map[inst["instance_token"]] = (
                    len(self.inst_map) + 1
                )  # starts at 1.

            # Bounding boxes:
            (bbox, bbox_aug, bbox_img, bbox_aug_img) = self._get_bbox_region_in_image(
                inst, egoPout_to_global, bev_aug
            )
            # fmt: off
            bbox, (center, bbox_h, bbox_w), offsets = self._process_bbox_region(
                bbox,bbox_img,visibility,inst,instance,x,y,centerness,SIGMA,offsets,
                mobility,center_bbox_on_img,is_visible, valid_centerness,
            )
            if bool_aug_activated:
                (
                    bbox_aug,(center_aug, bbox_h_aug, bbox_w_aug),offsets_aug,
                ) = self._process_bbox_region(
                    bbox_aug,bbox_aug_img,visibility_aug,inst,instance_aug,x,y,
                    centerness_aug,SIGMA,offsets_aug,mobility_aug,center_bbox_on_img_aug,
                    is_visible, valid_centerness_aug,
                )
            # fmt: on

            if is_ego:
                continue

            # Update
            # Objects: only objects that appear inside the image.
            if inst["category_name"] in DETECTION_CLS:
                if self.only_object_center_in:
                    if centers.min() >= -1 and centers.max() <= 1:
                        classes.append(self.class_to_idx[inst["category_name"]])
                        centers.append(center)
                        bbox_attr.append([bbox_h, bbox_w])
                else:
                    classes.append(self.class_to_idx[inst["category_name"]])
                    centers.append(center)
                    bbox_attr.append([bbox_h, bbox_w])

            bboxes[tok] = bbox
            if bool_aug_activated:
                if self.only_object_center_in:
                    if centers_aug.min() >= -1 and centers_aug.max() <= 1:
                        classes_aug.append(self.class_to_idx[inst["category_name"]])
                        centers_aug.append(center_aug)
                        bbox_attr_aug.append([bbox_h_aug, bbox_w_aug])
                else:
                    classes_aug.append(self.class_to_idx[inst["category_name"]])
                    centers_aug.append(center_aug)
                    bbox_attr_aug.append([bbox_h_aug, bbox_w_aug])
                bboxes_aug[tok] = bbox_aug

        # Add egopose bounding box
        (*_, bbox_egopose_img, bbox_egopose_aug_img) = self._get_bbox_region_in_image(
            inst_egopose, egoPout_to_global, bev_aug
        )
        bbox_egopose_img = self.geomscaler.pts_from_spatial_to_img(bbox_egopose_img)
        if bool_aug_activated:
            bbox_egopose_aug_img = self.geomscaler.pts_from_spatial_to_img(
                bbox_egopose_aug_img
            )
        else:
            bbox_egopose_aug_img = bbox_egopose_img

        if not bool_aug_activated:
            # List
            bboxes_aug = deepcopy(bboxes)
            classes_aug = classes.copy()
            center_bbox_on_img_aug = deepcopy(center_bbox_on_img)
            # Numpy
            visibility_aug = visibility.copy()
            mobility_aug = mobility.copy()
            centers_aug = centers.copy()
            bbox_attr_aug = bbox_attr.copy()
            valid_centerness_aug = valid_centerness.copy()
            # Torch
            centerness_aug = centerness.clone()
            offsets_aug = offsets.clone()

        # Can not stack empty list
        if len(centers) > 0:
            classes = torch.tensor(classes, dtype=torch.int64)
            centers = torch.from_numpy(np.stack(centers)).to(torch.float32)
            bbox_attr = torch.from_numpy(np.stack(bbox_attr)).to(torch.float32)

            classes_aug = torch.tensor(classes_aug, dtype=torch.int64)
            centers_aug = torch.from_numpy(np.stack(centers_aug)).to(torch.float32)
            bbox_attr_aug = torch.from_numpy(np.stack(bbox_attr_aug)).to(torch.float32)
        else:
            bbox_attr = torch.empty(0, dtype=torch.float32)
            centers = torch.empty(0, dtype=torch.float32)
            classes = torch.empty(0, dtype=torch.int64)

            bbox_attr_aug = torch.empty(0, dtype=torch.float32)
            centers_aug = torch.empty(0, dtype=torch.float32)
            classes_aug = torch.empty(0, dtype=torch.int64)

        # At least one element.
        if len(bboxes) > 0:
            bboxes = {
                k: torch.from_numpy(np.stack(v)).to(torch.float32)
                for k, v in bboxes.items()
            }
            bboxes_aug = {
                k: torch.from_numpy(np.stack(v)).to(torch.float32)
                for k, v in bboxes_aug.items()
            }

            # Process center_bbox_on_img: filter with visible_bbox
            center_bbox_on_img = torch.stack(center_bbox_on_img).to(torch.float32)
            center_bbox_on_img_aug = torch.stack(center_bbox_on_img_aug).to(
                torch.float32
            )
            offset_map, offset_map_aug = [
                self._get_offset_map_from_center_bbox(
                    torch.stack([x, y], dim=-1), bb[visible_bbox]
                ).permute(2, 0, 1)
                for bb in [center_bbox_on_img, center_bbox_on_img_aug]
            ]
        else:
            bboxes = {"": torch.empty(0, dtype=torch.float32)}
            bboxes_aug = {"": torch.empty(0, dtype=torch.float32)}
            center_bbox_on_img = torch.empty(0, dtype=torch.float32)
            center_bbox_on_img_aug = torch.empty(0, dtype=torch.float32)
            offset_map = torch.full([2, h, w], -1.0, dtype=torch.float32)
            offset_map_aug = torch.full([2, h, w], -1.0, dtype=torch.float32)

        # Ego pose bounding boxes
        bbox_egopose_img = torch.from_numpy(bbox_egopose_img).to(torch.float32)
        bbox_egopose_aug_img = torch.from_numpy(bbox_egopose_aug_img).to(torch.float32)

        # Lidar data
        if self.keep_input_lidar:
            # When using Lyft, some data are not divisible by 5. May be a bug in database.
            lidar_img, lidar_img_aug = self.get_lidar_data(
                rec, egoPout_to_global, bev_aug
            )
        else:
            lidar_img, lidar_img_aug = np.empty((h, w), dtype=np.int32), np.empty(
                (h, w), dtype=np.int32
            )

        # Prepare outputs
        (
            visibility,
            visibility_aug,
            mobility,
            mobility_aug,
            valid_centerness,
            valid_centerness_aug,
            lidar_img,
            lidar_img_aug,
        ) = [
            torch.from_numpy(x).unsqueeze(0)
            for x in [
                visibility,
                visibility_aug,
                mobility,
                mobility_aug,
                valid_centerness,
                valid_centerness_aug,
                lidar_img,
                lidar_img_aug,
            ]
        ]

        # Infer binimg from visibility
        binimg, binimg_aug = [
            torch.floor(1 - x // 255) for x in [visibility, visibility_aug]
        ]

        # BEV validity.
        valid_binimg = visibility >= min_vis
        valid_binimg_aug = visibility_aug >= min_vis
        valid_centerness = valid_centerness.bool()
        valid_centerness_aug = valid_centerness_aug.bool()

        # Change axes: space: (X: bottom, Y: right) -> image: (X: right, Y: bottom)
        [
            visibility,
            visibility_aug,
            mobility,
            mobility_aug,
            offsets,
            offsets_aug,
            centerness,
            centerness_aug,
            binimg,
            binimg_aug,
            valid_binimg,
            valid_binimg_aug,
            offset_map,
            offset_map_aug,
            valid_centerness,
            valid_centerness_aug,
            lidar_img,
            lidar_img_aug,
        ] = [
            prepare_img_axis(x, self.to_cam_ref)
            for x in [
                visibility,
                visibility_aug,
                mobility,
                mobility_aug,
                offsets,
                offsets_aug,
                centerness,
                centerness_aug,
                binimg,
                binimg_aug,
                valid_binimg,
                valid_binimg_aug,
                offset_map,
                offset_map_aug,
                valid_centerness,
                valid_centerness_aug,
                lidar_img,
                lidar_img_aug,
            ]
        ]

        return {
            "binimg": binimg,
            "binimg_aug": binimg_aug,
            "valid_binimg": valid_binimg,
            "valid_binimg_aug": valid_binimg_aug,
            "visibility": visibility,
            "visibility_aug": visibility_aug,
            "mobility": mobility,
            "mobility_aug": mobility_aug,
            "offsets": offsets,
            "offsets_aug": offsets_aug,
            "lidar_img": lidar_img,
            "lidar_img_aug": lidar_img_aug,
            "valid_centerness": valid_centerness,
            "valid_centerness_aug": valid_centerness_aug,
            "offsets_map": offset_map,
            "offsets_map_aug": offset_map_aug,
            "centerness": centerness,
            "centerness_aug": centerness_aug,
            "bboxes": bboxes,
            "bboxes_aug": bboxes_aug,
            "bbox_egopose": bbox_egopose_img,
            "bbox_egopose_aug": bbox_egopose_aug_img,
            "centers": centers,
            "centers_aug": centers_aug,
            "classes": classes,
            "classes_aug": classes_aug,
            "bbox_attr": bbox_attr,
            "bbox_attr_aug": bbox_attr_aug,
        }

    def _get_offset_map_from_center_bbox(self, grid, center_bbox):
        # Alias
        grid_res = grid.shape[:2]
        N_bbox = center_bbox.shape[0]

        # Shape
        grid = rearrange(grid, "h w c -> (h w) c")

        # Prevent empty center bbox
        if len(center_bbox) == 0:
            return torch.zeros(grid_res[0], grid_res[1], 2)

        dirs = torch.gather(
            (grid.unsqueeze(1) - center_bbox.unsqueeze(0)),
            1,
            torch.cdist(grid, center_bbox)
            .topk(1, largest=False)
            .indices.unsqueeze(-1)
            .expand(prod(grid_res), N_bbox, 2),
        )[:, 0, :].view(grid_res[0], grid_res[1], 2)
        # dirs = dirs / torch.tensor([grid_res[0], grid_res[1]]).view(1, 1, 2)
        return dirs

    def _get_bbox_region_in_image(self, inst, egoPout_to_global, bev_aug):
        # Global reference frame.
        box = Box(inst["translation"], inst["size"], Quaternion(inst["rotation"]))
        points = box.bottom_corners()

        homog_points = np.ones((4, 4))
        homog_points[:3, :] = points
        homog_points[-1, :] = 1

        # From global to ego_ref, from ego_ref to ego.
        homog_points = (invert_homogenous(egoPout_to_global)) @ homog_points

        # 3D
        # Image
        pts_aug_img = self._prepare_points_to_gtimg(bev_aug, homog_points)

        # 3D
        pts = homog_points[:2].T
        pts_aug = np.copy(pts)

        # Image
        pts_img = self._prepare_points_to_gtimg(np.eye(4), homog_points)

        return pts, pts_aug, pts_img, pts_aug_img

    def _prepare_points_to_gtimg(self, bev_aug, points):
        points_in = np.copy(points)

        Rquery = np.zeros((3, 3))
        # Inverse query aug:
        # Ex: when tx=10, the query is 10/res meters front,
        # so points are fictivelly 10/res meters back.
        Rquery[:3, :3] = bev_aug[:3, :3].T
        tquery = np.array([-1, -1, 1]) * bev_aug[:3, 3]
        tquery = tquery[:, None]

        # Rquery @ X + tquery
        if self.to_cam_ref:
            # Applying a camera transformation matrix change axis order.
            index = [0, 2]
        else:
            index = [0, 1]
        points_out = (Rquery @ (points_in[:3, :] + tquery))[index].T
        return points_out

    def _get_ego_to_global(self, rec_T):
        # Current time
        poserecord_T = self.nusc.get(
            "ego_pose",
            self.nusc.get("sample_data", rec_T["data"]["LIDAR_TOP"])[f"ego_pose_token"],
        )

        # Extract quaternion transf
        rot = Quaternion(poserecord_T["rotation"]).rotation_matrix
        trans = np.array(poserecord_T["translation"])

        mat = np.eye(4, dtype=np.float64)
        mat[:3, :3] = rot
        mat[:3, -1] = trans
        return mat

    def _process_bbox_region(
        # fmt: off
        self,bbox,bbox_img,visibility,inst,instance,x,y,centerness,sigma,offsets,
        mobility,center_bbox_on_img,is_visible:str=True, valid_centerness=None,
        # fmt: on
    ):
        # Alias
        h, w = self.nx[0], self.nx[1]

        (center, bbox_h, bbox_w) = from_corners_to_chw(bbox)
        center_img = np.mean(bbox_img, axis=0)

        # -> round
        poly_region_img_rd = self.geomscaler.pts_from_spatial_to_img(bbox_img)
        poly_region_img_rd = (np.round(poly_region_img_rd)).astype(np.int32)

        fill_func = lambda x, y, z: cv2.fillConvexPoly(x, y, z)

        # Update: Visibility
        if not self.is_lyft:
            inst_vis = int(inst["visibility_token"])
        else:
            inst_vis = 1
        fill_func(visibility, poly_region_img_rd, inst_vis)

        # Update mobile masks
        fill_func(mobility, poly_region_img_rd, int(inst["dynamic_tag"]))

        # -> local reference: [-50,50] -> [-1;1]
        bbox = self.geomscaler.pts_from_spatial_to_scale(bbox)

        # Update: Instance:
        inst_value = self.inst_map[inst["instance_token"]]
        fill_func(instance, poly_region_img_rd, inst_value)

        # Center and offsets
        # -> Offset
        # Returns coordinates of the center of the object on the image.
        xc, yc = self.geomscaler.pts_from_spatial_to_img(center_img)
        off_x = xc - x
        off_y = yc - y
        instance_mask = instance == inst_value

        # Multiply by -1 to have X axis pointing to the right.
        offsets[0, instance_mask] = -off_x[instance_mask].round()
        offsets[1, instance_mask] = -off_y[instance_mask].round()

        center_bbox_on_img.append(torch.tensor([xc, yc]))

        self.center_map[inst["instance_token"]] = (xc, yc)

        # -> Centerness:
        g = torch.exp(-(off_x**2 + off_y**2) / (2 * sigma**2))
        centerness[0] = torch.maximum(centerness[0], g)

        if not is_visible:
            valid_centerness[g >= THRESHOLD_VALID_CENTERNESS] = 0

        center = self.geomscaler.pts_from_spatial_to_scale(center)
        # -> Center: [-50,50] -> [-1;1]
        bbox_h, bbox_w = self.geomscaler.pts_from_spatial_to_scale(
            np.array([bbox_h, bbox_w])
        )

        return bbox, (center, bbox_h, bbox_w), offsets

    # -> Maps
    def get_map_related_data(self, rec, bev_aug=np.eye(4)):
        # Alias
        h, w = self.nx[0], self.nx[1]

        # Tokens
        scene_token = self.nusc.get("scene", rec["scene_token"])
        scene_name = scene_token["name"]
        egopose = self.nusc.get(
            "ego_pose",
            self.nusc.get("sample_data", rec["data"]["LIDAR_TOP"])["ego_pose_token"],
        )

        # Read pre-processed files
        map_mask = np.load(open(f"{self.hdmaproot}/map_0.1/{scene_name}.npy", "rb"))

        rec_info = json.load(
            open(f"{self.hdmaproot}/label/{self.split}/{scene_name}.json", "r")
        )[egopose["token"]]
        patch_box = np.loadtxt(
            open(f"{self.hdmaproot}/map_0.1/meta_{scene_name}.txt", "r")
        )

        # Extract info from files.
        margin = rec_info["margin"]
        patch_angle = float(rec_info["rot"]) * 180 / np.pi
        trans_xy = np.array([rec_info["trans_x"], rec_info["trans_y"]])

        # Extract image
        patch_box_ego = get_patch_box_from_trans(trans_xy, margin)
        trans_all_ego = np.array(patch_box_ego[:2]) - np.array(patch_box[:2])

        # Sub mask_map
        th, tw = trans_all_ego.astype(int)
        map_mask_ego = get_current_map_mask(map_mask, patch_angle, tw, th)

        # Prepare outputs
        keep_map = [i in self.hdmap_names for i in HDMAP_DICT.keys()]
        hdmap = map_mask_ego[:, :, keep_map]
        assert not self.to_cam_ref, "Not implemented with hdmap"
        hdmap = hdmap.to(torch.uint8).flip(0, 1).permute(2, 0, 1) // 255
        return {"hdmap": hdmap}

    # Other
    def choose_cams(self):
        if self.is_train and self.img_params["Ncams"] < len(self.img_params["cams"]):
            cams = np.random.choice(
                self.img_params["cams"], self.img_params["Ncams"], replace=False
            )
        else:
            cams = self.img_params["cams"]
        return cams

    def __str__(self):
        return f"""NuScenesDataset: {len(self)} samples. Split: {"train" if self.is_train else "val"}."""

    def __len__(self):
        return len(self.ixes)
