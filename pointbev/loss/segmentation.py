import pdb
from functools import partial
from typing import Optional

import torch
from einops import rearrange
from torch import nn
from torchvision.ops import sigmoid_focal_loss

from pointbev.loss import LossInterface


def select_loss(loss_fn, pred, target, time_index, scale_index, channel_index):
    if time_index is not None:
        loss = loss_fn(pred[:, time_index], target[:, time_index]).unsqueeze(1)
    elif scale_index is not None:
        loss = loss_fn(pred[scale_index], target[scale_index]).unsqueeze(0)
    elif channel_index is not None:
        loss = loss_fn(
            pred[:, :, channel_index], target[:, :, channel_index]
        ).unsqueeze(2)
    else:
        loss = loss_fn(pred, target)
    return loss


# Binimg losses.
class BCELoss(LossInterface):
    def __init__(
        self,
        pos_weight,
        key="binimg",
        name="loss_binimg",
        time_index: Optional[int] = None,
        scale_index: Optional[int] = None,
        channel_index: Optional[int] = None,
        select_index: Optional[int] = False,
    ):
        """
        BCE(p) = -(y * log(p) + (1 - y) * log(1 - p))

        if y=0:
            BCE(p) = -log(1 - p)
            - if p ~ 0:
                well classified and BCE(p) ~ 0
            - if p ~ 1:
                badly classified and BCE(p) ~ inf

        if y=1:
            BCE(p) = -log(p)
            - if p ~ 0:
                badly classified and BCE(p) ~ inf
            - if p ~ 1:
                well classified and BCE(p) ~ 0
        """

        super().__init__(key=key, name=name)
        self.loss_fn = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight]), reduction="none"
        )
        self.time_index = time_index
        self.scale_index = scale_index
        self.channel_index = channel_index
        self.select_index = select_index
        assert not (
            int(self.time_index is not None)
            + int(self.scale_index is not None)
            + int(self.channel_index is not None)
            > 1
        )

    def forward(self, pred, target, mask=None, target_weights=None, eps=1e-6):
        loss = select_loss(
            self.loss_fn,
            pred,
            target,
            self.time_index,
            self.scale_index,
            self.channel_index,
        )

        if target_weights is not None:
            loss = loss * (target * target_weights + (1 - target))

        if mask is None:
            mask = torch.ones_like(loss, dtype=torch.bool)

        return (loss * mask).sum() / (mask.sum() + eps)


class SpatialLoss(LossInterface):
    def __init__(self, norm, key="offsets", name="loss_offsets", ignore_index=None):
        super().__init__(key=key, name=name)

        if norm == 1:
            self.loss_fn = torch.nn.functional.l1_loss
        elif norm == 2:
            self.loss_fn = torch.nn.functional.mse_loss
        else:
            raise NotImplementedError
        self.ignore_index = ignore_index

    def forward(self, pred, target, mask=None, eps=1e-6) -> torch.Tensor:
        # Alias
        b, t, c, h, w = pred.shape

        loss = self.loss_fn(pred, target, reduction="none")

        if self.ignore_index is not None:
            target_mask = target != self.ignore_index
        else:
            target_mask = torch.ones_like(loss, dtype=torch.bool)

        if mask is None:
            mask = torch.ones_like(loss, dtype=torch.bool)

        mask = target_mask & mask
        return (loss * mask).sum() / (mask.sum() + eps)
