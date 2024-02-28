import torch
from torchmetrics.metric import Metric

from .segmentation import IoUMetric


class MeanMetric(Metric):
    def __init__(self):
        super().__init__(dist_sync_on_step=False, compute_on_step=False)
        self.add_state("metric", default=torch.zeros(1), dist_reduce_fx="sum")
        self.add_state("n_obs", default=torch.zeros(1), dist_reduce_fx="sum")
        return

    def update(self, metric):
        self.metric += metric.sum()
        self.n_obs += metric.shape[0]
        return

    def compute(self):
        return self.metric / self.n_obs
