from typing import Optional

import torch
from einops import rearrange
from torchmetrics.metric import Metric


class IoUMetric(Metric):
    def __init__(
        self,
        thresholds: float = 0.5,
        min_value_mask: Optional[int] = None,
        exact_value_mask: Optional[int] = None,
    ):
        """
        label_indices:
            transforms labels (c, h, w) to (len(labels), h, w)
            see config/experiment/* for examples
        min_value_mask:
            passing "None" will ignore the mask
            otherwise uses visibility values to ignore certain labels
        """
        super().__init__(dist_sync_on_step=False, compute_on_step=False)

        self.thresholds = thresholds
        self.exact_value_mask = exact_value_mask
        self.min_value_mask = min_value_mask

        self.add_state("tp", default=torch.zeros(1), dist_reduce_fx="sum")
        self.add_state("fp", default=torch.zeros(1), dist_reduce_fx="sum")
        self.add_state("fn", default=torch.zeros(1), dist_reduce_fx="sum")

    def _get_mask(self, mask, shape, device):
        if self.exact_value_mask is not None:
            assert mask is not None
            mask = mask == self.exact_value_mask

        elif self.min_value_mask is not None:
            assert mask is not None
            mask = mask >= self.min_value_mask

        elif mask is not None:
            mask = mask

        else:
            mask = torch.ones(shape, device=device, dtype=torch.float32)
        return mask

    def update(self, pred, label, mask=None):
        """Mask: 1 to keep, 0 to discard."""
        mask = self._get_mask(mask, pred.shape, pred.device)
        pred = pred.detach().view(-1, 1)
        label = label.detach().bool().view(-1, 1)
        mask = mask.detach().float().view(-1, 1)

        pred = pred >= self.thresholds

        self.tp += ((pred & label) * mask).sum(0)
        self.fp += ((pred & ~label) * mask).sum(0)
        self.fn += ((~pred & label) * mask).sum(0)

    def compute(self, eps=1e-7):
        return self.tp / (self.tp + self.fp + self.fn + eps)
