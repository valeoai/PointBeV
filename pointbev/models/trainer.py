from collections import defaultdict
from functools import partial
from typing import Any, Dict

import hydra
import torch
import torch.nn as nn
from omegaconf import DictConfig
from psutil import virtual_memory
from pytorch_lightning.core import LightningModule
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.utilities import rank_zero_only
from torch import Tensor
from torch.cuda import max_memory_allocated, max_memory_reserved

from pointbev.data.dataset.nuscenes_common import (MAP_DYNAMIC_TAG,
                                                   VISIBILITY_TAG)
from pointbev.loss import BCELoss, SpatialLoss, Weighting
from pointbev.metric import IoUMetric, MeanMetric
from pointbev.utils import (GeomScaler, nested_dict_to_nested_module_dict,
                            prepare_to_log_binimg, prepare_to_log_hdmap,
                            print_nested_dict)


class BasicTrainer(LightningModule):
    def __init__(
        self,
        net,
        optimizer: torch.optim.Optimizer = None,
        scheduler: torch.optim.lr_scheduler = None,
        weights_kwargs={},
        train_kwargs={},
        val_kwargs={},
        loss_kwargs={},
        metric_kwargs={},
        temporal_kwargs={},
        grid={"xbound": [], "ybound": [], "zbound": []},
        name="",
    ):
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=["net"])

        self.net = net

        # Temporal
        self._init_temporal(temporal_kwargs)

        # Losses
        dict_losses = self._init_loss(loss_kwargs)
        self.weighting = Weighting(dict_losses, weights_kwargs)
        self.dict_losses = nested_dict_to_nested_module_dict(dict_losses)

        # Metrics
        dict_metrics = self._init_metric(metric_kwargs)

        # Args
        self._print_info(dict_losses, dict_metrics)
        self._init_val(val_kwargs)
        self._init_train(train_kwargs)

        self.cur_step_train = 0
        self.cur_step_val = 0

        self.geomscaler = GeomScaler(grid)

    def _init_train(self, train_kwargs):
        self.train_loss_frequency = train_kwargs["train_loss_frequency"]
        self.train_visu_frequency = train_kwargs["train_visu_frequency"]
        self.train_visu_imgs = train_kwargs["train_visu_imgs"]
        self.train_visu_epoch_frequency = train_kwargs["train_visu_epoch_frequency"]
        return

    def _init_val(self, val_kwargs):
        # Additional training kwargs
        self.val_visu_imgs = val_kwargs["val_visu_imgs"]
        self.val_calculate_losses = val_kwargs["val_calculate_losses"]
        self.val_visu_frequency = val_kwargs["val_visu_frequency"]
        return

    def _init_temporal(self, temporal_kwargs):
        self.cam_T_P = torch.tensor(temporal_kwargs.cam_T_P)
        # ! Trace only present, to be changed if you want to apply loss on several timesteps.
        self.bev_T_P = torch.tensor(temporal_kwargs.bev_T_P)[-1:]
        return

    def _init_metric(self, metric_kwargs):
        dict_metrics = {}

        # -> Binimg
        # -> Time-Pose-valid
        if self.with_binimg:
            dict_metrics.update(
                {
                    "metric_iou_Time_Pose": {
                        f"T{time}_P{pos}": IoUMetric() for (time, pos) in self.bev_T_P
                    }
                }
            )

            # Tracking metrics during the learning
            self.track_pts = metric_kwargs["track_pts"]
            if self.track_pts:
                dict_metrics.update({"metric_N_fine_pts": MeanMetric()})
                dict_metrics.update({"metric_N_coarse_pts": MeanMetric()})

            self.track_mem = metric_kwargs["track_mem"]
            if self.track_mem:
                dict_metrics.update({"metric_mem": MeanMetric()})
                dict_metrics.update({"metric_mem_r": MeanMetric()})

            self.track_pts_thresh = metric_kwargs["track_pts_thresh"]
            self.pts_thresh = metric_kwargs["pts_thresh"]
            if self.track_pts_thresh:
                dict_metrics.update({"metric_pts_thresh": MeanMetric()})

        # -> Per dynamic tag
        self.map_dynamic_tag = MAP_DYNAMIC_TAG
        self.with_dynamic_tag = metric_kwargs.get("with_dynamic_tag", False)
        if self.with_dynamic_tag:
            dict_metrics.update(
                {
                    "metric_iou_per_dynamic_tag": {
                        k: IoUMetric(exact_value_mask=v)
                        for k, v in self.map_dynamic_tag
                    }
                }
            )

        # -> Per visibility tag
        self.map_visibility_tag = VISIBILITY_TAG
        self.with_visibility = metric_kwargs.get("with_visibility", False)
        if self.with_visibility:
            dict_metrics.update(
                {
                    "metric_iou_per_visibility": {
                        k: IoUMetric(exact_value_mask=v)
                        for k, v in self.map_visibility_tag
                    }
                }
            )

        # -> HDMap
        if self.with_hdmap:
            dict_metrics.update(
                {"metric_iou_hdmap": {k: IoUMetric() for k in self.hdmap_names}}
            )

        # ---> Training and validation metrics
        if metric_kwargs.get("only_val", True):
            avail_mode = ["val"]
        else:
            avail_mode = ["train", "val"]
        for mode in avail_mode:
            for name, metric in dict_metrics.items():
                if isinstance(metric, dict):
                    setattr(
                        self,
                        "_".join([name, mode]),
                        nn.ModuleDict({k: v.clone() for k, v in metric.items()}),
                    )
                else:
                    setattr(self, "_".join([name, mode]), metric.clone())

        return dict_metrics

    def _init_loss(self, loss_kwargs):
        dict_losses = defaultdict(lambda: defaultdict(dict))

        cls_loss_segm = loss_kwargs.get("segm_type").get("cls")
        cls_loss_kwargs = loss_kwargs.get("segm_type").get("kwargs")
        loss_segm = partial(eval(cls_loss_segm), **cls_loss_kwargs)

        # -> BEV
        self.with_binimg = loss_kwargs.get("with_binimg", False)
        if self.with_binimg:
            for index, elem in enumerate(self.bev_T_P):
                # Filter by activated outputs
                dict_losses["bev"]["binimg"][f"T{elem[0]}_P{elem[1]}"] = loss_segm(
                    time_index=index
                )

        # -> HDMap
        self.with_hdmap = loss_kwargs.get("with_hdmap", False)
        self.hdmap_names = loss_kwargs.get("hdmap_names", [])
        if self.with_hdmap:
            dict_losses["bev"].update({"hdmap": {}})
            for v, k in enumerate(self.hdmap_names):
                dict_losses["bev"]["hdmap"].update({k: loss_segm(channel_index=v)})

        # -> Centerness, offsets.
        self.with_centr_offs = loss_kwargs.get("with_centr_offs", False)
        if self.with_centr_offs:
            dict_losses["bev"].update(
                {
                    "centerness": SpatialLoss(norm=2),
                    "offsets": SpatialLoss(norm=1, ignore_index=255.0),
                }
            )
        return dict_losses

    @rank_zero_only
    def _print_info(self, dict_losses, dict_metrics):
        print("# --------- Losses --------- #")
        print_nested_dict(dict_losses, 0)
        print()

        print("# --------- Metrics --------- #")
        print_nested_dict(dict_metrics, 0)
        return

    # Forward
    def forward(self, batch):
        preds = self.net(**batch)
        return preds

    # Prepare
    def _wrap_loggers(self, method, *args, **kwargs):
        """Apply method on all loggers.
        Useful when Trainer uses several loggers."""
        [getattr(logger, method)(*args, **kwargs) for logger in self.loggers]

    def _switch_orig_augm(self, batch):
        """Invert GT and augmented GT."""
        for k in batch:
            if k in ["bev_aug"]:
                continue
            if k[-4:] == "_aug":
                k_woaug = k.replace("_aug", "")
                batch[k_woaug], batch[k] = batch[k], batch[k_woaug]

    def on_train_start(self):
        self._wrap_loggers("log_hyperparams", self.hparams)

    # Process
    def common_step(self, batch, step, mode="train", batch_idx=None):
        """Common step: prepare inputs, forward pass, compute losses and metrics."""
        # Augmentations:
        # Change reference and consider the augmented BEV as GT.
        if mode == "train":
            self._switch_orig_augm(batch)

        preds = self(batch)
        out_dict = {}

        # Losses
        losses, loss = self._common_step_losses(preds, batch)
        out_dict.update({"loss": loss})

        # Metrics
        self._common_step_metrics(preds, batch, mode)

        # Logs
        if (mode == "train" and step % self.train_loss_frequency == 0) or (
            mode == "val" and self.val_calculate_losses
        ):
            self._wrap_loggers(
                "log_metrics", {f"{mode}/{k}": v for k, v in losses.items()}, step=step
            )
            self._wrap_loggers("log_metrics", {f"{mode}/loss": loss}, step=step)
            self.log(f"{mode}_loss", loss, prog_bar=True, logger=False)

        # Change GT and augmented.
        if mode == "train":
            self._switch_orig_augm(batch)

        return out_dict, preds

    def _get_masks(self, preds, batch):
        """Get masks to apply on losses and metrics."""

        def union(pred_mask, target_mask):
            if pred_mask is not None and target_mask is not None:
                final_mask = target_mask & pred_mask
            elif pred_mask is not None:
                final_mask = pred_mask.bool()
            elif target_mask is not None:
                final_mask = target_mask.bool()
            else:
                final_mask = None
            return final_mask

        # -> Rectangular: Binimg, offsets,
        # GT mask.
        if "valid_binimg" in batch.keys():
            tgt_mask = batch["valid_binimg"]
        else:
            tgt_mask = None
        # Pred mask.
        if "masks" in preds.keys() and (self.with_binimg):
            pred_mask = preds["masks"]["bev"]["binimg"]
        else:
            if tgt_mask is not None:
                pred_mask = torch.ones_like(tgt_mask)
            else:
                pred_mask = None
        # Union
        binimg_mask = union(pred_mask, tgt_mask)

        # -> Centerness
        if "valid_centerness" in batch.keys():
            tgt_mask = batch["valid_centerness"]
        else:
            tgt_mask = None
        if "masks" in preds.keys():
            if self.with_centr_offs:
                pred_mask = preds["masks"]["bev"]["centerness"]
            else:
                pred_mask = None
        else:
            if tgt_mask is not None:
                pred_mask = torch.ones_like(tgt_mask)
            else:
                pred_mask = None
        # Union
        centerness_mask = union(pred_mask, tgt_mask)

        # Statistics: ~elements to keep.
        if binimg_mask is not None:
            self.log(
                "mask_binimg", binimg_mask.float().mean(), prog_bar=True, logger=False
            )
        if centerness_mask is not None:
            self.log(
                "mask_centerness",
                centerness_mask.float().mean(),
                prog_bar=True,
                logger=False,
            )
        return {
            "binimg": binimg_mask,
            "centerness": centerness_mask,
        }

    def _common_step_losses(self, preds, batch):
        losses = {}
        total_loss = 0.0

        def _update_total_loss(total_loss, loss, name, weighting):
            (weight, uncertainty) = weighting(name)
            return total_loss + loss * weight + uncertainty

        update_total_loss = partial(_update_total_loss, weighting=self.weighting)

        # Pipeline losses
        keys = self.dict_losses.keys()
        bev_losses = self.dict_losses["bev"] if "bev" in keys else None

        # Masks: 0 to remove, 1 to keep.
        dict_masks = self._get_masks(preds, batch)

        # Single element:
        # -> Centerness, Offsets
        for l_dict, l_pip, l_key, pred_key, target_key, l_mask, l_bool in zip(
            [bev_losses, bev_losses],
            ["bev", "bev"],
            ["centerness", "offsets"],
            ["centerness", "offsets"],
            ["centerness", "offsets"],
            [dict_masks["centerness"], dict_masks["binimg"]],
            [self.with_centr_offs, self.with_centr_offs],
        ):
            if not l_bool:
                continue

            l_bev_loss = l_dict[l_key]
            l_pred = preds[l_pip][pred_key]
            # ! Trace only present
            l_target = batch[target_key][:, -1:]

            loss = l_bev_loss(l_pred, l_target, l_mask)
            name = f"{l_pip}/{l_key}"
            losses.update({name: loss})
            total_loss = update_total_loss(total_loss, loss, name)

        # -> Dictionaries:
        # Binimg, HDMap
        for l_key, pred_key, target_key, l_mask, l_bool in zip(
            ["binimg", "hdmap"],
            ["binimg", "hdmap"],
            ["binimg", "hdmap"],
            [dict_masks["binimg"], None],
            [self.with_binimg, self.with_hdmap],
        ):
            if not l_bool:
                continue
            l_bev_losses = bev_losses[l_key]
            l_preds = preds["bev"][pred_key]
            l_targets = batch[target_key]

            for k, l in l_bev_losses.items():
                loss = l(l_preds, l_targets, l_mask)
                name = f"bev/{l_key}/{k}"
                losses.update({name: loss})
                total_loss = update_total_loss(total_loss, loss, name)
        return losses, total_loss / len(losses)

    @torch.no_grad()
    def _common_step_metrics(self, preds, batch, mode):
        # Metrics
        if "binimg" in batch.keys():
            pred_binimg = preds["bev"]["binimg"].sigmoid()
            target_binimg = batch["binimg"]
            valid_binimg = batch["valid_binimg"]

            # Time-Pose-valid
            if hasattr(self, f"metric_iou_Time_Pose_{mode}"):
                metric_dict = getattr(self, "_".join(["metric_iou_Time_Pose", mode]))
                for index, (time, pose) in enumerate(self.bev_T_P):
                    metric = metric_dict[f"T{time}_P{pose}"]
                    # ! Trace only present
                    metric(
                        pred_binimg[:, index],
                        target_binimg[:, index][:, -1:],
                        valid_binimg[:, index],
                    )

            if hasattr(self, f"metric_N_coarse_pts_{mode}"):
                metric = getattr(self, "_".join(["metric_N_coarse_pts", mode]))
                metric(preds["tracks"]["N_coarse"].unsqueeze(0))

            if hasattr(self, f"metric_N_fine_pts_{mode}"):
                metric = getattr(self, "_".join(["metric_N_fine_pts", mode]))
                metric(preds["tracks"]["N_fine"].unsqueeze(0))

            if hasattr(self, f"metric_mem_{mode}"):
                metric = getattr(self, "_".join(["metric_mem", mode]))
                metric(torch.tensor([preds["tracks"]["mem"]]))

            if hasattr(self, f"metric_mem_r_{mode}"):
                metric = getattr(self, "_".join(["metric_mem_r", mode]))
                metric(torch.tensor([preds["tracks"]["mem_r"]]))

            if hasattr(self, f"metric_pts_thresh_{mode}"):
                metric = getattr(self, "_".join(["metric_pts_thresh", mode]))
                metric(
                    torch.tensor(
                        [
                            torch.where(preds["bev"]["binimg"] > self.pts_thresh)[
                                0
                            ].size(0)
                        ]
                    )
                )

            # -> Per dynamic tag
            if hasattr(self, f"metric_iou_per_dynamic_tag_{mode}"):
                metric_dict = getattr(
                    self, "_".join(["metric_iou_per_dynamic_tag", mode])
                )
                for metric in metric_dict.values():
                    metric(pred_binimg, target_binimg, batch["mobility"] * valid_binimg)

            # -> Per visibility tag
            if hasattr(self, f"metric_iou_per_visibility_{mode}"):
                metric_dict = getattr(
                    self, "_".join(["metric_iou_per_visibility", mode])
                )
                for metric in metric_dict.values():
                    metric(
                        pred_binimg, target_binimg, batch["visibility"] * valid_binimg
                    )

        if "hdmap" in batch.keys():
            # -> HDMap
            if hasattr(self, f"metric_iou_hdmap_{mode}"):
                pred_binimg_hdmap = preds["bev"]["hdmap"].sigmoid()
                metric_dict = getattr(self, "_".join(["metric_iou_hdmap", mode]))
                for v, k in enumerate(self.hdmap_names):
                    metric = metric_dict[k]
                    metric(
                        pred_binimg_hdmap[:, :, v : v + 1].contiguous(),
                        batch["hdmap"][:, :, v : v + 1].contiguous(),
                    )

    def _init_preds_dict_for_vis(self, preds):
        preds_dict = {"bev": {}, "masks": {}}

        # -> BEV
        if "binimg" in preds["bev"].keys():
            preds_dict["bev"]["binimg"] = [preds["bev"]["binimg"].sigmoid()]

        if "hdmap" in preds["bev"].keys():
            preds_dict["bev"]["hdmap"] = [preds["bev"]["hdmap"].sigmoid()]

        # -> Masks
        if "masks" in preds.keys():
            if "binimg" in preds["masks"]["bev"].keys():
                preds_dict["masks"]["binimg"] = [preds["masks"]["bev"]["binimg"]]
        return preds_dict

    def training_step(self, batch, batch_idx):
        out_dict, preds = self.common_step(
            batch, step=self.cur_step_train, mode="train", batch_idx=batch_idx
        )
        # Alias
        log_ = partial(self.log, prog_bar=True, logger=False)

        # Traces
        log_("ram_pct", virtual_memory().percent)
        log_("gpu_mem_res", (max_memory_reserved(device=self.device) / (2**30)))
        log_("gpu_mem_alloc", (max_memory_allocated(device=self.device) / (2**30)))

        # Outputs
        if "bev" in preds.keys():
            if "binimg" in preds["bev"].keys():
                log_("max_pred", preds["bev"]["binimg"].max())

        try:
            log_("lr", self.lr_schedulers().get_last_lr()[0])
        except AttributeError:
            pass

        # Loss weights
        if self.cur_step_train % self.train_loss_frequency == 0:
            for k in self.weighting.weight_dict.keys():
                weight, _ = self.weighting(k)
                self._wrap_loggers(
                    "log_metrics",
                    {f"weight_{k}": weight.item()},
                    step=self.cur_step_train,
                )
                log_(f"weight_{k}", weight.item())

        # Visualize
        if (
            (self.train_visu_imgs)
            and (batch_idx % self.train_visu_frequency == 0)
            and (self.current_epoch % self.train_visu_epoch_frequency == 0)
        ):
            preds_dict = self._init_preds_dict_for_vis(preds)
            self._group_additional_preds(preds_dict)
            self._process_img_step(preds_dict, batch, batch_idx, mode="train")

        self.cur_step_train += 1
        return out_dict

    def validation_step(self, batch, batch_idx):
        out_dict, preds = self.common_step(
            batch, step=self.cur_step_val, mode="val", batch_idx=batch_idx
        )

        preds_dict = self._init_preds_dict_for_vis(preds)

        # Only at some specific batches.
        if (
            (self.val_visu_frequency > 0)
            and (batch_idx % self.val_visu_frequency == 0)
            and self.val_visu_imgs
        ):
            self._group_additional_preds(preds_dict)
            self._process_img_step(preds_dict, batch, batch_idx, mode="val")

        self.cur_step_val += 1
        return out_dict

    def test_step(self, batch, batch_idx):
        out_dict, preds = self.common_step(
            batch, step=self.cur_step_val, mode="val", batch_idx=batch_idx
        )
        return out_dict

    # Process images
    @torch.no_grad()
    def _process_img_step(self, preds_dict, batch, batch_idx, mode):
        out_img, out_caption = [], []
        # Prepare images
        if "binimg" in batch.keys():
            out_img.append(prepare_to_log_binimg(preds_dict, batch))
            out_caption.append(f"N°: ({mode}) {batch_idx}")

        if "hdmap" in batch.keys():
            out_img.append(prepare_to_log_hdmap(preds_dict, batch))
            out_caption.append(f"N° hdmap: ({mode}) {batch_idx}")

        # Save to logger
        for logger in self.loggers:
            if isinstance(logger, WandbLogger):
                if len(out_img) > 0:
                    logger.log_image(
                        f"{mode}_epoch_{self.current_epoch}",
                        out_img,
                        step=batch_idx,
                        caption=out_caption,
                    )

    @torch.no_grad()
    def _group_additional_preds(
        self, preds_additional: Dict[str, Dict[str, Any]]
    ) -> None:
        for pip in preds_additional.keys():
            for sub in preds_additional[pip].keys():
                preds_additional[pip][sub] = torch.stack(
                    preds_additional[pip][sub], dim=0
                )

    # Epochs
    @torch.no_grad()
    def common_epoch_end(self, mode):
        log_dict = {}

        # Metrics
        for ref, name in zip(
            ["bev", "bev", "bev", "bev", "bev", "bev"],
            [
                "offsets",
                "N_coarse_pts",
                "N_fine_pts",
                "mem",
                "mem_r",
                "pts_thresh",
            ],
        ):
            if hasattr(self, f"metric_{name}_{mode}"):
                # Compute metric and reset
                metric = getattr(self, f"metric_{name}_{mode}")
                scores = metric.compute()
                log_dict[f"{mode}_{ref}_metric_{name}"] = scores.item()

                self._wrap_loggers(
                    "log_metrics",
                    {f"{mode}/{ref}/metric_{name}": scores},
                    step=self.current_epoch,
                )
                metric.reset()

        # Dict metrics:
        for ref, name in zip(
            [
                "bev",
                "bev",
                "bev",
                "bev",
            ],
            [
                "iou_Time_Pose",  # -> Time-Pose-valid
                "iou_per_dynamic_tag",  # -> Per dynamic tag
                "iou_per_visibility",  # -> Per visibility tag
                "iou_hdmap",  # -> HDMap
            ],
        ):
            if hasattr(self, f"metric_{name}_{mode}"):
                metric_dict = getattr(self, "_".join([f"metric_{name}", mode]))
                for subname, metric in metric_dict.items():
                    scores = metric.compute().item()
                    metric.reset()
                    log_dict[f"{mode}_{ref}_metric_{name}_{subname}"] = scores

                    self._wrap_loggers(
                        "log_metrics",
                        {f"{mode}/{ref}/metric_{name}_{subname}": scores},
                        step=self.current_epoch,
                    )

        log_dict["mean_metrics"] = sum([v for v in log_dict.values()]) / len(log_dict)
        # Reset cache
        torch.cuda.empty_cache()
        return log_dict

    def on_training_epoch_end(self):
        self.common_epoch_end(mode="train")
        return

    def on_validation_epoch_end(self):
        log_dict = self.common_epoch_end(mode="val")
        self.log_dict(
            log_dict, prog_bar=True, on_epoch=True, logger=False, sync_dist=True
        )
        return

    def on_test_epoch_end(self):
        log_dict = self.common_epoch_end(mode="val")
        self.log_dict(
            log_dict, prog_bar=True, on_epoch=True, logger=False, sync_dist=True
        )
        return

    # Optimizer
    def _update_one_cycle_lr(self, config, grad_steps_per_epoch, lr):
        config["total_steps"] = (
            grad_steps_per_epoch
            * self.trainer.max_epochs
            // (self.trainer.accumulate_grad_batches)
        )
        config["max_lr"] = lr
        return

    def configure_optimizers(self):
        optimizer = self.hparams.optimizer(params=self.parameters())

        if self.hparams.scheduler is None:
            return {"optimizer": optimizer}

        if isinstance(self.hparams.scheduler, (dict, DictConfig)):
            # Scheduler using dynamic parameters. (e.g number of epochs).
            interval = self.hparams.scheduler.pop("interval")

            if "OneCycleLR" in self.hparams.scheduler.classname:
                self.hparams.scheduler["_target_"] = self.hparams.scheduler.pop(
                    "classname"
                )
                self.trainer.estimated_stepping_batches
                grad_steps_per_epoch = len(self.trainer.train_dataloader)
                self._update_one_cycle_lr(
                    self.hparams.scheduler,
                    grad_steps_per_epoch,
                    optimizer.param_groups[0]["lr"],
                )

            scheduler = hydra.utils.instantiate(
                self.hparams.scheduler, optimizer=optimizer
            )

            lr_scheduler = {
                "scheduler": scheduler,
                "interval": interval,
                "frequency": 1,
                "name": "lr",
            }
        else:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            lr_scheduler = {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
                "name": "lr",
            }
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}

    # Saving
    def on_save_checkpoint(self, checkpoint):
        checkpoint["model_class_path"] = (
            self.__module__ + "." + self.__class__.__qualname__
        )
