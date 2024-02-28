import pdb
from typing import Dict, List, Union

import torch
from pytorch_lightning.utilities import rank_zero_only
from torch import nn

from pointbev.utils import get_element_from_nested_key, unpack_nested_dict


class LossInterface(nn.Module):
    def __init__(
        self,
        key: Union[str, List[str]],
        name: Union[str, List[str]],
    ):
        """Defines the expected interface of a loss function.

        key: corresponds to the dictionary key (or keys) used in preds and targets
        to compute the loss function.
        E.g:
            - key="binimg" will compute the loss function using preds["binimg"].
            - key=["binimg","classes"] will compute the loss function using
            {"binimg":preds["binimg"], "classes":preds["classes"]}.

        name: corresponds to the name (or names) of the dictionary after calculation.
        E.g.
            - "loss_centerness" will return {"loss_centerness":loss_value}.
            - ["loss_centerness","loss_classes"] will return {"loss_centerness":loss_value1,
            "loss_classes":loss_value2}.
        """
        super().__init__()
        self.key = key
        self.name = name

    def forward(self, preds, targets) -> Union[Dict[str, torch.Tensor], torch.Tensor]:
        raise NotImplementedError


class Weighting(nn.Module):
    def __init__(self, dict_losses, kwargs, isolated_loss: str = "bev/binimg/T0_P0"):
        super().__init__()
        weights_mode = kwargs["mode"]

        unpacked_dict_losses = unpack_nested_dict(dict_losses)
        loss_names = unpacked_dict_losses.keys()

        self.isolated_loss = isolated_loss
        if weights_mode == "default":
            weight_dict = Weighting.from_default(
                loss_names, coeff=kwargs.get("coeff", 1.0)
            )
        elif weights_mode in ["learned", "semi_learned"]:
            bool_ = weights_mode == "semi_learned"
            weight_dict = Weighting.from_learned(loss_names, bool_, isolated_loss)
        elif weights_mode == "manually":
            weight_dict = Weighting.from_manually(kwargs["weights"])
        else:
            raise NotImplementedError(f"weights_mode {weights_mode} not implemented.")

        weight_dict = {
            k: nn.Parameter(v, requires_grad=v.requires_grad)
            for k, v in weight_dict.items()
        }
        self.weight_dict = nn.ParameterDict(weight_dict)
        self.coeffs_dict = self._init_coeffs_dict(weights_mode)
        self.mode = weights_mode
        self._print_desc()

    @rank_zero_only
    def _print_desc(self):
        print("# --------- Weighting --------- #")
        print("Initialized weight: ")
        print("name / init / coeffs / requires_grad")
        for k, v in self.weight_dict.items():
            print(
                f"{k:<30} {v.data.item():<10.3f} {self.coeffs_dict[k]:<10.3f} {v.requires_grad}"
            )
        print()

    def _init_coeffs_dict(self, weights_mode):
        dict_coeffs = {k: 1.0 for k in self.weight_dict.keys()}

        if weights_mode in ["learned", "semi_learned"]:
            dict_coeffs = {k: 0.5 for k in self.weight_dict.keys()}
            if weights_mode == "learned":
                # ! Factor 10 as in SimpleBEV.
                if self.isolated_loss in self.weight_dict.keys():
                    dict_coeffs[self.isolated_loss] = 10.0
            if weights_mode == "semi_learned":
                if self.isolated_loss in self.weight_dict.keys():
                    dict_coeffs[self.isolated_loss] = 1.0
        return dict_coeffs

    @staticmethod
    def from_default(loss_names, coeff=1.0):
        weight_dict = {
            loss_name: nn.Parameter(torch.tensor([coeff]), requires_grad=False)
            for loss_name in loss_names
        }
        return weight_dict

    @staticmethod
    def from_manually(weights):
        weight_dict = {
            loss_name: nn.Parameter(torch.tensor([w]), requires_grad=False)
            for loss_name, w in weights.items()
        }
        return weight_dict

    @staticmethod
    def from_learned(loss_names, semi_learned=False, isolated_loss="bev/binimg/T0_P0"):
        weight_dict = {
            loss_name: nn.Parameter(torch.tensor(0.0), requires_grad=True)
            for loss_name in loss_names
        }

        if semi_learned:
            # Fix the binimg weight to 1.
            assert isolated_loss in weight_dict.keys()
            weight_dict[isolated_loss] = nn.Parameter(
                torch.tensor(1.0), requires_grad=False
            )
        return weight_dict

    def _get_uncertainty(self, key, weight):
        if self.mode in ["default", "manually"]:
            uncertainty = 0

        elif self.mode in ["learned", "semi_learned"]:
            if self.mode == "learned":
                uncertainty = 0.5 * weight

            elif self.mode == "semi_learned":
                uncertainty = 0.5 * weight if (key != self.isolated_loss) else 0

        return uncertainty

    def forward(self, key):
        weight = self.weight_dict[key]
        uncertainty = self._get_uncertainty(key, weight)
        coeff = self.coeffs_dict[key]

        if self.mode in ["learned", "semi_learned"]:
            weight = 1 / torch.exp(weight)
        return (weight * coeff, uncertainty)


class LossCollection(LossInterface):
    def __init__(self, losses, kwargs={"weights_mode": "default"}):
        super().__init__(key="", name="")
        self.pipelines = losses.keys()
        self.dict_losses = nn.ModuleDict(unpack_nested_dict(losses))

        loss_names = []
        for k, v in self.dict_losses.items():
            if isinstance(v.name, str):
                loss_names.append(k)
            else:
                loss_names += ["/".join([k, _]) for _ in v.name]

        self.weighting = Weighting(loss_names, kwargs)

    def forward(self, preds, targets, masks=None):
        dict_out = {}
        for key, v in self.dict_losses.items():
            pred = get_element_from_nested_key(preds, key)
            mask = get_element_from_nested_key(masks, key)
            target = get_element_from_nested_key(targets, v.key)
            out_l = self.dict_losses[key](pred, target, mask)
            if isinstance(out_l, dict):
                dict_l = {"/".join([key, k]): v for k, v in out_l.items()}
            else:
                dict_l = {key: out_l}
            dict_out.update(dict_l)

        weights = {k: self.weighting(k) for k in dict_out.keys()}
        loss = torch.stack(
            [l * weights[k][0] + weights[k][1] for k, l in dict_out.items()]
        ).mean()
        return dict_out, loss
