""" 
Author: Loick Chambon

Datamodule for the Temporal-NuScenes dataset.
"""

import os
from functools import partial
from pathlib import Path
from typing import List, Optional

import numpy as np
import pytorch_lightning as pl
import torch
from einops import rearrange
from nuscenes.nuscenes import NuScenes
from torch import Tensor

from pointbev.data.dataset import TemporalNuScenesDataset

try:
    from lyft_dataset_sdk.lyftdataset import LyftDataset
except:
    print("LYFT not installed")


class NuScenesDatamodule(pl.LightningDataModule):
    def __init__(
        self,
        # Nuscenes
        version,
        dataroot,
        hdmaproot,
        is_lyft,
        # Grid
        grid,
        # Images
        img_loader,
        img_params,
        # Coefficients
        coeffs={},
        # Dataloader
        cls_tag="TemporalNuScenesDataset",
        collate_fn="custom",
        batch_size=1,
        valid_batch_size=None,
        num_workers=10,
        pin_memory=True,
        prefetch_factor=2,
        train_drop_last=True,
        train_shuffle=False,
        val_shuffle=False,
        # Inputs
        normalize_img=True,
        keep_input_binimg: bool = True,
        keep_input_centr_offs: bool = False,
        keep_input_detection: bool = False,
        keep_input_hdmap: bool = False,
        hdmap_names: List[str] = ["drivable_area"],
        keep_input_persp: bool = False,
        keep_input_sampling: bool = False,
        keep_input_offsets_map: bool = False,
        keep_input_lidar: bool = False,
        save_folder: str = "",
        visualise_mode: bool = False,
        # BEV aug
        apply_valid_bev_aug: bool = True,
        # Multi-scale
        kernel_scales: List[int] = [1],
        # Temporal
        cam_T_P: List[List[int]] = [[0, 0]],
        bev_T_P: List[List[int]] = [[0, 0]],
        mode_ref_cam_T: str = "present",
        # Filters
        only_object_center_in: bool = False,
        filters_cat: List[str] = [],
        plot_ego: bool = False,
        # Cameras
        to_cam_ref: bool = False,
        random_cam_ref: bool = False,
        force_camref: Optional[int] = None,
    ):
        super().__init__()

        # Nuscenes
        self.version = version
        self.dataroot = dataroot
        self.is_lyft = is_lyft
        # Paths
        self.hdmaproot = hdmaproot
        # Grid
        self.grid = grid
        # Images
        self.img_loader = img_loader
        self.img_params = img_params
        # Coefficients
        self.coeffs = coeffs
        # Dataloader
        self.batch_size = int(batch_size)
        self.valid_batch_size = (
            int(valid_batch_size) if valid_batch_size is not None else int(batch_size)
        )
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.prefetch_factor = prefetch_factor
        self.train_drop_last = train_drop_last
        self.train_shuffle = train_shuffle
        self.val_shuffle = val_shuffle
        # Inputs
        self.normalize_img = normalize_img
        self.keep_input_binimg = keep_input_binimg
        self.keep_input_centr_offs = keep_input_centr_offs
        self.keep_input_detection = keep_input_detection
        self.keep_input_hdmap = keep_input_hdmap
        self.hdmap_names = hdmap_names
        self.keep_input_persp = keep_input_persp
        self.keep_input_sampling = keep_input_sampling
        self.keep_input_offsets_map = keep_input_offsets_map
        self.keep_input_lidar = keep_input_lidar
        self.save_folder = save_folder
        # Query aug
        self.apply_valid_bev_aug = apply_valid_bev_aug
        # Multi-scale
        self.kernel_scales = kernel_scales
        # Temporal
        self.cam_T_P = cam_T_P
        self.bev_T_P = bev_T_P
        self.mode_ref_cam_T = mode_ref_cam_T
        # Filters
        self.only_object_center_in = only_object_center_in
        self.filters_cat = filters_cat
        self.plot_ego = plot_ego
        # Cameras
        self.to_cam_ref = to_cam_ref
        self.random_cam_ref = random_cam_ref
        self.force_camref = force_camref

        self.cls = eval(cls_tag)

        self.collate_fn = collate_batch if collate_fn == "custom" else None

        # Debug
        self.visualise_mode = visualise_mode

    def setup(self, stage: Optional[str] = None):
        if not self.is_lyft:
            nusc = NuScenes(
                version="v1.0-{}".format(self.version),
                dataroot=self.dataroot,
                verbose=False,
            )
        else:
            dataroot = Path(self.dataroot).parent / "lyft"
            nusc = LyftDataset(
                data_path=dataroot,
                json_path=os.path.join(dataroot, "train_data"),
                verbose=True,
            )
        # Validation dataset
        partial_data = partial(
            self.cls,
            nusc=nusc,
            is_lyft=self.is_lyft,
            # Grid
            grid=self.grid,
            img_params=self.img_params,
            img_loader=self.img_loader,
            normalize_img=self.normalize_img,
            # Images
            to_cam_ref=self.to_cam_ref,
            random_cam_ref=self.random_cam_ref,
            force_camref=self.force_camref,
            # Augmentations
            coeffs=self.coeffs,
            # Filters
            only_object_center_in=self.only_object_center_in,
            filters_cat=self.filters_cat,
            plot_ego=self.plot_ego,
            # Temporal
            cam_T_P=self.cam_T_P,
            bev_T_P=self.bev_T_P,
            mode_ref_cam_T=self.mode_ref_cam_T,
            # Outputs
            keep_input_sampling=self.keep_input_sampling,
            keep_input_detection=self.keep_input_detection,
            keep_input_centr_offs=self.keep_input_centr_offs,
            keep_input_hdmap=self.keep_input_hdmap,
            hdmap_names=self.hdmap_names,
            keep_input_binimg=self.keep_input_binimg,
            keep_input_persp=self.keep_input_persp,
            keep_input_offsets_map=self.keep_input_offsets_map,
            keep_input_lidar=self.keep_input_lidar,
            save_folder=self.save_folder,
            # Paths
            hdmaproot=self.hdmaproot,
        )
        self.valdata = partial_data(
            # Mode
            is_train=False,
        )

        if stage == "only_val":
            return

        # Training dataset
        self.traindata = partial_data(
            # Mode
            is_train=True,
        )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.traindata,
            batch_size=self.batch_size,
            shuffle=self.train_shuffle,
            num_workers=self.num_workers,
            drop_last=self.train_drop_last,
            # worker_init_fn=worker_rnd_init,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.valdata,
            batch_size=self.valid_batch_size,
            shuffle=self.val_shuffle,
            drop_last=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor,
            collate_fn=self.collate_fn,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.valdata,
            batch_size=self.valid_batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor,
            collate_fn=self.collate_fn,
        )

    def on_after_batch_transfer(self, batch, dataloader_idx):
        for key in ["binimg", "binimg_aug"]:
            if key in batch.keys():
                # Some outputs are stored as int, but we need them as float for the loss.
                batch[key] = batch[key].float()

        # Object detection activated
        if self.keep_input_detection:
            batch["classes"] = [[elem for elem in b] for b in batch["classes"]]
            batch["classes_aug"] = [[elem for elem in b] for b in batch["classes_aug"]]

            batch["bbox_attr"] = [[elem for elem in b] for b in batch["bbox_attr"]]
            batch["bbox_attr_aug"] = [
                [elem for elem in b] for b in batch["bbox_attr_aug"]
            ]

            batch["centers"] = [[elem for elem in b] for b in batch["centers"]]
            batch["centers_aug"] = [[elem for elem in b] for b in batch["centers_aug"]]

        # HDMaps
        if self.keep_input_hdmap:
            batch["hdmap"] = batch["hdmap"].float()

        if self.keep_input_offsets_map:
            batch["offsets_map_dist"] = torch.sqrt(
                batch["offsets_map"][:, :, 0] ** 2 + batch["offsets_map"][:, :, 1] ** 2
            ).unsqueeze(2)
            batch["offsets_map_dist_aug"] = torch.sqrt(
                batch["offsets_map_aug"][:, :, 0] ** 2
                + batch["offsets_map_aug"][:, :, 1] ** 2
            ).unsqueeze(2)

        return batch


def worker_rnd_init(x):
    np.random.seed(13 + x)


def collate_batch(batch: List[Tensor]):
    key_as_list_of_tensor = [
        "classes",
        "classes_aug",
        "bbox_attr",
        "bbox_attr_aug",
        "centers",
        "centers_aug",
        "bboxes",
        "bboxes_aug",
        "bbox_egopose",
        "bbox_egopose_aug",
        "tokens",
    ]
    keys = batch[0].keys()
    out_dict = {
        k: torch.stack([b[k] for b in batch])
        for k in keys
        if k not in key_as_list_of_tensor
    }

    for k in key_as_list_of_tensor:
        if k in keys:
            out_dict.update({k: [b[k] for b in batch]})
    return out_dict
