"""
File to process images and tensors for logging / saving.
"""

import torch
from einops import rearrange, repeat
from pytorch_lightning.utilities import rank_zero_only
from torchvision.utils import draw_keypoints, make_grid

from pointbev.utils import pylogger

log = pylogger.get_pylogger(__name__)
cams = [
    "CAM_FRONT_LEFT",
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_LEFT",
    "CAM_BACK",
    "CAM_BACK_RIGHT",
]


def align_types_and_format(preds, targets):
    # Repeat 1 channel to 3 channels
    c_pred, c_target = preds.shape[-3], targets.shape[-3]
    if c_pred != targets.shape[-3]:
        if (c_pred == 1) and (c_target == 3):
            preds = preds.repeat(1, 1, 1, c_target, 1, 1)

    # Convert all to float
    preds = preds.float()
    targets = targets.float()
    return preds, targets


def prepare_to_log_hdmap(preds, targets, max_item=1):
    # Alias
    bs, t, c, h, w = targets["hdmap"].shape

    # Prepare images
    preds_hdmap = preds["bev"]["hdmap"].detach().cpu().unsqueeze(-3)
    targets_hdmap = targets["hdmap"].detach().cpu().unsqueeze(-3)

    # View
    views = preds_hdmap.shape[0]
    targets_hdmap = targets_hdmap.unsqueeze(0)

    # Define the display settings
    nelems = min(max_item, bs)

    targets_img = torch.cat([targets_hdmap], dim=0)
    preds_img = preds_hdmap
    preds_img, targets_img = align_types_and_format(preds_img, targets_img)

    # Grid
    grid = make_grid(
        tensor=torch.cat([targets_img[:, :nelems], preds_img[:, :nelems]], axis=0)
        .permute(2, 0, 1, 3, 4, 5, 6)
        .flatten(0, 3),
        padding=10,
        pad_value=1,
        nrow=len(targets_img) + views,
    )
    return grid


def prepare_to_log_binimg(preds, targets, max_item=1):
    # Alias
    bs, t, c, h, w = targets["binimg"].shape

    # Prepare images
    preds_bin = preds["bev"]["binimg"].detach().cpu()
    targets_bin = targets["binimg"].detach().cpu()

    # View
    views = preds_bin.shape[0]
    targets_bin = targets_bin.unsqueeze(0)

    # Define the display settings
    nelems = min(max_item, bs)

    targets_img = torch.cat([targets_bin], dim=0)
    preds_img = preds_bin
    preds_img, targets_img = align_types_and_format(preds_img, targets_img)

    # Grid
    grid = make_grid(
        tensor=torch.cat([targets_img[:, :nelems], preds_img[:, :nelems]], axis=0)
        .permute(2, 0, 1, 3, 4, 5)
        .flatten(0, 2),
        padding=10,
        pad_value=1,
        nrow=len(targets_img) + views,
    )
    return grid


@rank_zero_only
def log_hyperparameters(object_dict: dict) -> None:
    """Controls which config parts are saved by lightning loggers.

    Additionally saves:
    - Number of model parameters
    """

    hparams = {}

    cfg = object_dict["cfg"]
    model = object_dict["model"]
    trainer = object_dict["trainer"]

    if not trainer.logger:
        log.warning("Logger not found! Skipping hyperparameter logging...")
        return

    hparams["model"] = cfg["model"]

    # save number of model parameters
    hparams["model/params/total"] = sum(p.numel() for p in model.parameters())
    hparams["model/params/trainable"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    hparams["model/params/non_trainable"] = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )

    hparams["data"] = cfg["data"]
    hparams["trainer"] = cfg["trainer"]

    hparams["callbacks"] = cfg.get("callbacks")
    hparams["extras"] = cfg.get("extras")

    hparams["task_name"] = cfg.get("task_name")
    hparams["tags"] = cfg.get("tags")
    hparams["ckpt_path"] = cfg.get("ckpt_path")
    hparams["seed"] = cfg.get("seed")

    # send hparams to all loggers
    for logger in trainer.loggers:
        logger.log_hyperparams(hparams)
