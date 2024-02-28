""" 
Author: Loick Chambon

Create maps for each scene in NuScenes.
Each scene contains several ego poses.
We collect all ego poses information before creating maps.

Ex:
scene-0001 : 10 ego_token -> We need to collect info
of each 10 samples to create the corresponding scene.
"""

import pdb
import warnings

from shapely.errors import ShapelyDeprecationWarning

warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)

import argparse
import json
import os
from itertools import chain
from typing import List

import cv2
import numpy as np
from nuscenes import NuScenes
from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.prediction import PredictHelper
from nuscenes.utils.splits import create_splits_scenes
from pyquaternion import Quaternion
from tqdm import tqdm

LAYER_NAMES = [
    "lane",
    "road_segment",
    "drivable_area",
    "road_divider",
    "lane_divider",
    "stop_line",
    "ped_crossing",
    "walkway",
]


def get_prediction_challenge_split(split: str, dataroot: str) -> List[str]:
    if split not in {"mini_train", "mini_val", "train", "val"}:
        raise ValueError("split must be one of (mini_train, mini_val, train, val)")
    split_name = split

    path_to_file = os.path.join(dataroot, "maps", "prediction_scenes.json")
    prediction_scenes = json.load(open(path_to_file, "r"))
    scenes = create_splits_scenes()
    scenes_for_split = scenes[split_name]

    token_list_for_scenes = map(
        lambda scene: prediction_scenes.get(scene, []), scenes_for_split
    )

    return (
        prediction_scenes,
        scenes_for_split,
        list(chain.from_iterable(token_list_for_scenes)),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_root",
        help="path to the original nuScenes dataset",
        default="data/nuScenes",
    )
    parser.add_argument(
        "--data_out",
        default="data/nuscenes_processed_map/",
        help="path where you save the processed data",
    )
    parser.add_argument(
        "--split", default=["mini_val"], help="NuScenes split to use", nargs="*"
    )
    parser.add_argument("--version", default="mini", help="NuScenes version")
    args = parser.parse_args()

    DATAROOT = args.data_root
    DATAOUT = args.data_out
    print("# -------------- #")
    SCALE = 1.0  # 3.0
    print("Scale: Lower than original to accelerate: speed vs quality tradeoff.")
    MARGIN = 150
    print(
        "Larger than original to catch more context because of rotation introducing padding."
    )
    print("# -------------- #")
    nuscenes = NuScenes(f"v1.0-{args.version}", dataroot=DATAROOT)
    map_version = "0.1"
    SPLITS = args.split

    for split in SPLITS:
        os.makedirs(f"{DATAOUT}/label/{split}", exist_ok=True)
        os.makedirs(f"{DATAOUT}/map_{map_version}", exist_ok=True)
        (prediction_scenes, split_scenes, split_data) = get_prediction_challenge_split(
            split, dataroot=DATAROOT
        )
        helper = PredictHelper(nuscenes)

        total_pred = 0
        for cnt, scene_name in enumerate(tqdm(split_scenes)):
            scene_token = nuscenes.field2token("scene", "name", scene_name)[0]
            scene = nuscenes.get("scene", scene_token)
            first_sample_token = scene["first_sample_token"]
            first_sample = nuscenes.get("sample", first_sample_token)

            frame_id = 0
            sample = first_sample

            cvt_data = {}
            ego_pos_token_list = []
            trans_xy = []
            while True:
                instances_in_frame = []

                # Get ego car location
                sensor = "CAM_FRONT"
                cam_front_data = nuscenes.get("sample_data", sample["data"][sensor])
                ego_pose_token = nuscenes.get(
                    "ego_pose",
                    nuscenes.get("sample_data", sample["data"]["LIDAR_TOP"])[
                        "ego_pose_token"
                    ],
                )
                if ego_pose_token["token"] not in ego_pos_token_list:
                    ego_pos_token_list.append(ego_pose_token["token"])
                    annotation = ego_pose_token

                    data = {}
                    data["frame_id"] = frame_id
                    data["margin"] = MARGIN
                    data["scale"] = SCALE
                    data["scene_name"] = scene_name
                    data["trans_x"] = annotation["translation"][0]
                    data["trans_z"] = annotation["translation"][2]
                    data["trans_y"] = annotation["translation"][1]
                    data["rot"] = Quaternion(annotation["rotation"]).yaw_pitch_roll[0]

                    # Update
                    trans_xy.append([data["trans_x"], data["trans_y"]])
                    cvt_data[ego_pose_token["token"]] = data

                frame_id += 1
                if sample["next"] != "":
                    sample = nuscenes.get("sample", sample["next"])
                else:
                    break

            # Generate Maps
            map_name = nuscenes.get("log", scene["log_token"])["location"]
            nusc_map = NuScenesMap(dataroot=DATAROOT, map_name=map_name)

            # Maps center and scale
            trans_xy = np.stack(trans_xy)
            xy = trans_xy.astype(np.float32)
            x_min = np.round(xy[:, 0].min() - MARGIN)
            x_max = np.round(xy[:, 0].max() + MARGIN)
            y_min = np.round(xy[:, 1].min() - MARGIN)
            y_max = np.round(xy[:, 1].max() + MARGIN)
            x_size = x_max - x_min
            y_size = y_max - y_min
            patch_box = (
                x_min + 0.5 * (x_max - x_min),
                y_min + 0.5 * (y_max - y_min),
                y_size,
                x_size,
            )
            patch_angle = 0
            canvas_size = (
                np.round(SCALE * y_size).astype(int),
                np.round(SCALE * x_size).astype(int),
            )

            map_mask = (
                nusc_map.get_map_mask(patch_box, patch_angle, LAYER_NAMES, canvas_size)
                * 255.0
            ).astype(np.uint8)
            map_mask = np.swapaxes(map_mask, 1, 2)  # h,w to w,h

            # Save: Patch box
            meta = np.array(patch_box)
            np.savetxt(
                f"{DATAOUT}/map_{map_version}/meta_{scene_name}.txt", meta, fmt="%.2f"
            )

            # Save: masks
            np.save(
                f"{DATAOUT}/map_{map_version}/{scene_name}",
                np.transpose(map_mask, (1, 2, 0)),
            )

            # Save: cvt_data
            json.dump(cvt_data, open(f"{DATAOUT}/label/{split}/{scene_name}.json", "w"))

            pred_num = len(cvt_data)
            total_pred += pred_num

            print(f"{scene_name} finished! map_shape {map_mask.shape}")

        print(f"{split}_len: {len(split_data)} total_pred: {total_pred}")
