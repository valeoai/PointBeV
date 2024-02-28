"""
Temporal extension of the nuScenes dataloader.
Author: Loick Chambon
"""

import pdb
from typing import Any, Dict, List, Tuple

import numpy as np
import numpy.typing as npt
import torch
from pytorch_lightning.utilities import rank_zero_only

from pointbev.data.dataset.nuscenes_common import CAMREF, NuScenesDataset
from pointbev.utils.geom import (
    get_random_ref_matrix,
    get_yawtransfmat_from_mat,
    invert_homogenous,
)


class TemporalNuScenesDataset(NuScenesDataset):
    def __init__(
        self,
        # Temporal
        cam_T_P: List[List[int]] = [[0, 0]],
        bev_T_P: List[List[int]] = [[0, 0]],
        mode_ref_cam_T: str = "present",
        # Inputs
        keep_input_sampling: bool = False,
        keep_input_detection: bool = False,
        keep_input_centr_offs: bool = False,
        keep_input_hdmap: bool = False,
        keep_input_binimg: bool = True,
        keep_input_offsets_map: bool = False,
        keep_input_lidar: bool = False,
        keep_input_persp: bool = False,
        save_folder: bool = "",
        *args,
        **kwargs,
    ):
        super().__init__(
            keep_input_persp=keep_input_persp,
            keep_input_lidar=keep_input_lidar,
            *args,
            **kwargs,
        )
        self._init_temporality(cam_T_P, bev_T_P)
        self._init_out_keys(
            keep_input_sampling,
            keep_input_detection,
            keep_input_centr_offs,
            keep_input_hdmap,
            keep_input_binimg,
            keep_input_offsets_map,
            keep_input_lidar,
            keep_input_persp,
        )
        self.save_folder = save_folder
        self._print_desc()
        assert mode_ref_cam_T in ["random", "present", "self"]
        self.mode_ref_cam_T = mode_ref_cam_T

    def _init_temporality(
        self,
        cam_T_P: List[List[int]] = [[0, 0]],
        bev_T_P: List[List[int]] = [[0, 0]],
    ) -> None:
        """Initialized self-temporality related arguments.

        Args:
            cam_T_P: List of tuple containing :
                - input times used to get camera related inputs;
                - pose used to query the grid for each camera;
            bev_T: List of tuple containing:
                - output times used to get bev related inputs;
                - pose used to query the grid for each output;
        """
        self.cam_T_P = np.array(cam_T_P)
        self.bev_T_P = np.array(bev_T_P)

        self.bev_T, self.bev_P = (
            self.bev_T_P[:, 0],
            self.bev_T_P[:, 1],
        )
        self.cam_T, self.cam_P = (
            self.cam_T_P[:, 0],
            self.cam_T_P[:, 1],
        )

        self.union_time = np.array(
            list(set(self.cam_T) | set(self.bev_T) | set(self.bev_P) | set(self.cam_P))
        )
        self.union_time.sort()

        if len(cam_T_P) > 0:
            self.cam_T_index = np.concatenate(
                [np.where(self.union_time == i)[0] for i in self.cam_T]
            )
            self.cam_P_index = np.concatenate(
                [np.where(self.union_time == i)[0] for i in self.cam_P]
            )
        else:
            self.cam_T_index = None
            self.cam_P_index = None

        if len(bev_T_P) > 0:
            self.bev_T_index = np.concatenate(
                [np.where(self.union_time == i)[0] for i in self.bev_T]
            )
            self.bev_P_index = np.concatenate(
                [np.where(self.union_time == i)[0] for i in self.bev_P]
            )
        else:
            self.bev_T_index = None
            self.bev_P_index = None

        self.past_index = np.where(self.union_time < 0)[0]
        self.present_index = np.where(self.union_time == 0)[0]
        self.future_index = np.where(self.union_time > 0)[0]

        # Horizon length
        horizon_length = max(self.union_time) - min(self.union_time) + 1
        self.indices = self.get_indices(horizon=horizon_length)

    def _init_out_keys(
        self,
        keep_input_sampling,
        keep_input_detection,
        keep_input_centr_offs,
        keep_input_hdmap,
        keep_input_binimg,
        keep_input_offsets_map,
        keep_input_lidar,
        keep_input_persp,
    ):
        """Init keys to keep in the output dictionary."""
        keys_to_keep = [
            "tokens",
            "imgs",
            "rots",
            "trans",
            "intrins",
            "mobility",
            "mobility_aug",
            "egoTin_to_seq",
            "bev_aug",
            "egoTout_to_seq",
        ]

        self.with_hdmap = keep_input_hdmap
        if keep_input_binimg:
            keys_to_keep.append("binimg")
            keys_to_keep.append("binimg_aug")
            keys_to_keep.append("visibility")
            keys_to_keep.append("visibility_aug")
            keys_to_keep.append("valid_binimg")
            keys_to_keep.append("valid_binimg_aug")

        if keep_input_persp:
            keys_to_keep.append("persp_imgs")

        if keep_input_offsets_map:
            keys_to_keep.append("offsets_map")
            keys_to_keep.append("offsets_map_aug")

        if keep_input_hdmap:
            keys_to_keep.append("hdmap")

        if keep_input_centr_offs:
            keys_to_keep.append("offsets")
            keys_to_keep.append("offsets_aug")
            keys_to_keep.append("valid_centerness")
            keys_to_keep.append("valid_centerness_aug")
            keys_to_keep.append("centerness")
            keys_to_keep.append("centerness_aug")

        if keep_input_sampling:
            keys_to_keep.append("bboxes")
            keys_to_keep.append("bboxes_aug")
            keys_to_keep.append("bbox_egopose")
            keys_to_keep.append("bbox_egopose_aug")

        if keep_input_detection:
            keys_to_keep.append("centers")
            keys_to_keep.append("centers_aug")
            keys_to_keep.append("classes")
            keys_to_keep.append("classes_aug")
            keys_to_keep.append("bbox_attr")
            keys_to_keep.append("bbox_attr_aug")

        if keep_input_lidar:
            keys_to_keep.append("lidar_img")
            keys_to_keep.append("lidar_img_aug")

        self.keys_to_keep = keys_to_keep

    def __len__(self):
        return len(self.indices)

    @rank_zero_only
    def _print_desc(self):
        print()
        print(self)
        print(f"Needed records: {self.union_time}")
        print(f"Cam. T value / index: {self.cam_T} / {self.cam_T_index}")
        print(f"Cam. P value / index : {self.cam_P} / {self.cam_P_index}")
        print(f"BEV. T value / index: {self.bev_T} / {self.bev_T_index}")
        print(f"BEV. P value / index: {self.bev_P} / {self.bev_P_index}")

    def get_indices(self, horizon):
        """Return horizon samples such as all data in a same sequence are from the same scene.

        E.j: [0,1,2,3,4,5] is a valid sample sequence if all corresponding data are from the same scene and
        if sequence length equals to 6.

        Performance: 25ms with timeit compared to 118ms for the original implementation.
        """
        # Find when to change scene
        brk_pts = [i for i in range(len(self.ixes)) if self.ixes[i]["prev"] == ""]
        brk_pts.append(len(self.ixes))
        # Rolling window over the break points
        indices = np.concatenate(
            [
                np.lib.stride_tricks.sliding_window_view(np.arange(start, end), horizon)
                for start, end in zip(brk_pts[:-1], brk_pts[1:])
            ]
        )

        return indices

    def _get_egoTin_to_egoTout(self, rec_Tin, rec_Tout):
        """
        Returns the matrix transformation to move from one fame (t=Tin) reference frame to
        a common reference frame (t=Tout).
        """
        mat = np.eye(4)
        if rec_Tin["token"] == rec_Tout["token"]:
            return mat

        # Input time
        mat_in = self._get_ego_to_global(rec_Tin)

        # Final time
        mat_out = self._get_ego_to_global(rec_Tout)
        return invert_homogenous(mat_out) @ (mat_in)

    def change_cam_ref(self, data_cams, camref):
        for d in data_cams:
            for k in d.keys():
                d[k][[CAMREF, camref]] = d[k][[camref, CAMREF]]
        return data_cams

    # Input - Output
    def _get_inputs_cam(
        self, cam_records: List[Dict[str, Any]]
    ) -> Tuple[Dict[str, torch.Tensor], List[str], str]:
        """Get camera inputs.

        Args:
            cam_records (Dict[str,Any]): List of input records from which we get the camera inputs.

        Returns:
            out_cams: Dict[str,torch.Tensor]: Dictionary containing the camera related inputs.
            cams: List[str]: List of camera names.
            camref: str: Camera reference name.
        """
        cams = self.choose_cams()
        data_cams = [self.get_camera_related_data(rec, cams) for rec in cam_records]

        # Change camera reference if needed.
        if self.force_camref is not None:
            camref = self.force_camref
        else:
            if self.is_train:
                if self.random_cam_ref:
                    camref = np.random.randint(6)
                else:
                    camref = CAMREF
            else:
                camref = CAMREF

        # Change input data accordingly to camref.
        data_cams = self.change_cam_ref(data_cams, camref)

        if len(cam_records) > 0:
            out_cams = {
                k: torch.stack([data_t[k] for data_t in data_cams])
                for k in data_cams[0].keys()
            }
        else:
            out_cams = {}
        return out_cams, cams, camref

    def _get_outputs_bev(
        self,
        bev_records_T: List[Tuple[int, int]],
        egoPout_to_global: npt.NDArray,
        bev_aug: npt.NDArray,
    ) -> List[Dict[str, Any]]:
        """Get the BEV related outputs.

        Args:
            bev_records_T (List[Tuple[int, int]]): List of output time and pose.
            egoPout_to_global (npt.NDArray): Matrix from world to recorded ego reference frame.
            egoTout_to_seq (npt.NDArray): Matrix from one ego (time) to another ego (pose) reference frame.
            seq_aug (npt.NDArray): Augmentation matrix moving the sequence. Does not impact the BEV.
            bev_aug (npt.NDArray): Augmentation matrix moving the bev. Impacts the BEV.

        Returns:
            List[Dict[str, Any]]: List of BEV related outputs.
        """
        data_bev = []
        tokens = []

        for i, rec in enumerate(bev_records_T):
            tokens.append(rec["token"])
            out_bev_dict = self.get_bev_related_data(
                rec=rec,
                egoPout_to_global=egoPout_to_global[i],
                bev_aug=bev_aug[i],  # from query to query aug.
            )

            out_bev_dict.update({"tokens": tokens})

            if self.with_hdmap:
                out_bev_dict.update(self.get_map_related_data(rec, bev_aug[i]))

            data_bev.append(out_bev_dict)

        # Reset instance mapping.
        self.inst_map = {}
        self.center_map = {}

        return data_bev

    # Matrix related
    def _save_bev_aug(self, out_dict, bev_aug):
        in_bev_aug = []
        for i in self.cam_P_index:
            index = np.where(self.bev_P_index == i)[0]
            if len(index) > 0:
                in_bev_aug.append(bev_aug[index[0]])
            else:
                in_bev_aug.append(np.eye(4))
        out_dict.update({"bev_aug": torch.from_numpy(np.stack(in_bev_aug)).float()})
        return

    def _save_egoTin_to_seq(self, out_dict, cam_records, present_record) -> None:
        """Saves the following transformation: ref_in -> ref_out -> augmented ref out.
        Where:
            - ref_in is the chosen reference frame (either random, present or self);
            - ref_out is the final reference frame (the one of the camera record).

        Explanations: By default, the car is always centered at [0,0], this matrix moves this
        center according to its position in the sequence and to an augmentation matrix.
        Used when a model takes into account the global coordinates of the ego car.
        Affects the whole sequence.

        Note: It does not impact the BEV.

        Args:
            cam_records: List of input camera records. Used to set the reference
            frame as the origin.
            present_record: Present camera record. Used to set the reference frame as the origin.
        """
        # -> Ego motion: reference ego car frame.
        # Can be chosen among available timesteps.
        if self.mode_ref_cam_T == "random":
            # Could be a random frame of the sequence.
            rec_reference_in = [np.random.choice(cam_records)] * len(cam_records)
        elif self.mode_ref_cam_T == "present":
            # Select the present as the last available frame.
            rec_reference_in = [present_record] * len(cam_records)
        elif self.mode_ref_cam_T == "self":
            # Consider the ego position of each frame, i.e work in local coordinates.
            rec_reference_in = cam_records

        # --> Apply the same random matrices.
        # In nuscenes, extrinsics are inverted, so the ego should be.
        # extrinsics: from world to ego.
        egoTin_to_seq = np.stack(
            [
                self._get_egoTin_to_egoTout(rec, rec_reference_in[i])
                for i, rec in enumerate(cam_records)
            ]
        )

        # Update
        out_dict["egoTin_to_seq"] = torch.from_numpy(egoTin_to_seq).float()
        return

    def _get_inputs_bevaug(
        self,
        bev_records_P: List[Dict[str, Any]],
    ) -> npt.NDArray:
        """Returns the bev augmentation matrix.

        Explanations:
        It is used to move the query BEV around the coordinates of the car at the given timestep.
        By default, BEV are centered at the location of the car of the current timestep, but we may want to
        look at the BEV from a different location. This matrix moves the coordinate of the query BEV.

        Args:
            override_bev_aug (npt.NDArray): Specified query augmentation matrix. Defaults to None.
            bev_records_P (List[Dict[str, Any]]): List of pose output records.

        Returns:
            bev_aug (npt.NDArray): BEV augmentation matrix (Nq,4x4)
        """
        # -> Query: augmentation matrix applied on the BEV
        if self.is_train:
            # During training, could be a random matrix.
            bev_aug = np.stack(
                [
                    get_random_ref_matrix(coeffs=self.coeffs.bev_aug)
                    for _ in range(len(bev_records_P))
                ]
            )
        elif not self.is_train:
            # During validation, always identity.
            bev_aug = np.eye(4, dtype=np.float64)[None, ...].repeat(
                len(bev_records_P), 0
            )

        return bev_aug

    def _save_egoreftoego_out_dict(
        self,
        out_dict: Dict[str, Any],
        bev_records_T: List[Tuple[int, int]],
        bev_records_P: List[Tuple[int, int]],
    ) -> None:
        """Save the ego Tout reference frame to ego sequence reference frame matrix.
        It does not impact BEV. It takes into account the location of the car at the given timestep.
        """
        egoTout_to_seq = np.stack(
            [
                get_yawtransfmat_from_mat(self._get_egoTin_to_egoTout(rec_t, rec_p))
                for rec_t, rec_p in zip(bev_records_T, bev_records_P)
            ]
        )

        out_dict.update(
            {
                "egoTout_to_seq": torch.from_numpy(
                    np.stack([(egoTout_to_seq[i]) for i in range(len(bev_records_P))]),
                ).float()
            }
        )

    def merge_bbox_anns(
        self, in_dict: List[Dict[str, torch.Tensor]], ref_index: int = 0
    ) -> Dict[str, List[torch.Tensor]]:
        """Merge annotations of the same object across timesteps.

        Args:
            in_dict (List[Dict[str, torch.Tensor]]): Dictionary containing the annotations for each timesteps.
            ref_index (int): Integer initializing final keys.

        Returns:
            Dict[str, List[torch.Tensor]]: Dictionary containing as keys, the annotations in the first timestep and as values, the list of annotations across timesteps.
        """
        T = len(in_dict)
        bbox_anns = {k: [v] for k, v in in_dict[ref_index].items()}
        for ann_token in bbox_anns.keys():
            if ann_token != "":
                next_tok = self.nusc.get("sample_annotation", ann_token)["next"]
            else:
                continue
            for t in range(1, T):
                bboxes_t = in_dict[t]
                if next_tok in bboxes_t:
                    bbox_anns[ann_token].append(bboxes_t[next_tok])
                    if next_tok != "":
                        next_tok = self.nusc.get("sample_annotation", next_tok)["next"]
                    else:
                        continue
                else:
                    bbox_anns[ann_token].append(None)
        return bbox_anns

    # Format related
    def _prepare_out_dict(
        self,
        out_dict: Dict[str, Any],
        data_bev: Dict[str, Any],
        predict_keys: List[str],
    ) -> Dict[str, Any]:
        """Stack temporal data, update and filter the output dictionary."""
        out_dict.update(
            {
                k: (
                    torch.stack([data_t[k] for data_t in data_bev])
                    if k
                    not in [
                        "classes",
                        "classes_aug",
                        "bbox_attr",
                        "bbox_attr_aug",
                        "centers",
                        "centers_aug",
                        "bboxes",
                        "bboxes_aug",
                        "tokens",
                    ]
                    else [data_t[k] for data_t in data_bev]
                )
                for k in predict_keys
            }
        )

        # Merge annotations.
        out_dict["bboxes"] = self.merge_bbox_anns(out_dict["bboxes"])
        out_dict["bboxes_aug"] = self.merge_bbox_anns(out_dict["bboxes_aug"])

        # Filtered output dictionary.
        return {k: v for k, v in out_dict.items() if k in self.keys_to_keep}

    def __getitem__(self, index):
        # Records of the sequence
        records = np.array([self.ixes[i] for i in self.indices[index]])

        # Input records:
        cam_records = records[self.cam_T_index]
        present_record = records[self.present_index][0]

        # Get camera inputs
        out_dict, *_ = self._get_inputs_cam(cam_records)

        # Get sequence-aug transformation matrix
        self._save_egoTin_to_seq(out_dict, cam_records, present_record)

        # Bev GT:
        # Get BEV records:
        bev_records_T = records[self.bev_T_index]
        bev_records_P = records[self.bev_P_index]

        # Get query-aug transformation matrix
        bev_aug = self._get_inputs_bevaug(bev_records_P)
        self._save_bev_aug(out_dict, bev_aug)

        # -> Ego motion: output reference timestep.
        egoPout_to_global = np.stack(
            [
                get_yawtransfmat_from_mat(self._get_ego_to_global(rec_p))
                for rec_p in bev_records_P
            ]
        )

        self._save_egoreftoego_out_dict(out_dict, bev_records_T, bev_records_P)

        # Get bev outputs
        data_bev = self._get_outputs_bev(bev_records_T, egoPout_to_global, bev_aug)
        predict_keys = list(data_bev[0].keys())

        # Prepare
        out_dict = self._prepare_out_dict(out_dict, data_bev, predict_keys)
        return out_dict
