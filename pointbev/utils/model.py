from collections import defaultdict
from typing import List, Union

import torch


def load_state_model(
    model,
    ckpt,
    keys_to_freeze: Union[str, List[str]],
    keys_to_load: Union[str, List[str]],
    verbose=0,
):
    message = "\n# ------ Prepare model ------ #\n"

    if ckpt is not None:
        state_dict = ckpt["state_dict"]

        loaded_dict = defaultdict(int)
        if keys_to_load is not None:
            for n, p in model.state_dict().items():
                if keys_to_load == "all" or any(key in n for key in keys_to_load):
                    if n in state_dict:
                        loaded_dict[".".join(n.split(".")[:2])] += p.numel()
                        with torch.no_grad():
                            p.copy_(state_dict[n])
                    else:
                        message += f"Not loading {n}\n"
                else:
                    continue
        message += f"\nLoaded params: {dict(loaded_dict)}\n"
        message += f"Total loaded params: {sum(loaded_dict.values())}\n"

    # Freeze
    freezed_dict = defaultdict(int)
    if keys_to_freeze is not None:
        for gen in [model.named_buffers(), model.named_parameters()]:
            for n, p in gen:
                if keys_to_freeze == "all" or any(key in n for key in keys_to_freeze):
                    freezed_dict[".".join(n.split(".")[:2])] += p.numel()
                    p.requires_grad = False
                else:
                    continue

    # PL might change .train() after in the training so we force no runing stats.
    for n, m in model.named_modules():
        if keys_to_freeze is not None and (
            keys_to_freeze == "all" or any(key in n for key in keys_to_freeze)
        ):
            if isinstance(m, torch.nn.BatchNorm2d) or isinstance(
                m, torch.nn.BatchNorm1d
            ):
                if hasattr(m, "weight"):
                    m.weight.requires_grad_(False)
                if hasattr(m, "bias"):
                    m.bias.requires_grad_(False)
                m.track_running_stats = False
                m.eval()

    message += f"\nFreezed params: {dict(freezed_dict)}\n"
    message += f"Total freezed params: {sum(freezed_dict.values())}\n"
    if verbose > 0:
        print(message)
    return model
