import functools

import torch
from pytorch_lightning.utilities import rank_zero_only
from torch import nn


def unpack_nested_dict(dict_, symbol="/", mem=""):
    """Unpacks a nested dictionary into a flat dictionary."""
    out = {}
    for k, v in dict_.items():
        if v is not None:
            if isinstance(v, dict) or isinstance(v, list):
                new_mem = symbol.join([mem, str(k)]) if mem != "" else str(k)
                out.update(unpack_nested_dict(v, symbol, new_mem))
            else:
                out.update({symbol.join([mem, str(k)]) if mem != "" else str(k): v})
    return out


def get_element_from_nested_key(inputs, keys, symbol="/"):
    if inputs is None:
        return None

    if isinstance(keys, str):
        keys = [keys]

    out = {}

    for key in keys:
        inp = inputs

        # Check input is not a dictionary
        if not isinstance(inp, dict):
            return None

        # Check if separator is in key
        keep = False
        if symbol in key:
            while symbol in key:
                inp = inp[key.split(symbol)[0]]
                key = key.split(symbol)[1]
            if key in inp.keys():
                inp = inp[key]
                keep = True
        elif key in inp.keys():
            inp = inp[key]
            keep = True

        if keep:
            out[key] = inp

    if len(out) == 1:
        return list(out.values())[0]
    return out


def list_dict_to_dict_list(list_dict):
    """Convert a list of dictionnary to a dictionnary of list of values."""
    shared_dict = {}
    if isinstance(list_dict, list):
        for d in list_dict:
            for k in d.keys():
                if k not in shared_dict:
                    shared_dict[k] = []
                shared_dict[k].append(d[k])
    return shared_dict


def print_shape(x):
    if isinstance(x, tuple) or isinstance(x, list):
        for e in x:
            print_shape_singleton(e)
    elif isinstance(x, dict):
        print_shape_dict(x)
    else:
        print_shape_singleton(x)


def print_shape_dict(dict_):
    for k, v in dict_.items():
        if isinstance(v, dict):
            print_shape_dict(v)
        else:
            print_shape_singleton(v)


def print_shape_singleton(e):
    if isinstance(e, torch.Tensor):
        print(f"--> {e.shape}")


def nested_dict_to_nested_module_dict(dict_):
    out = {}
    for k, v in dict_.items():
        if isinstance(v, dict):
            out.update(nn.ModuleDict({k: nested_dict_to_nested_module_dict(v)}))
        else:
            out.update(nn.ModuleDict({str(k): v}))
    return nn.ModuleDict(out)


@rank_zero_only
def print_nested_dict(dict_, level):
    for k, v in dict_.items():
        if isinstance(v, dict):
            print(f"{'  '*level}{k}:")
            print_nested_dict(v, level + 1)
        else:
            print(f"{'  '*level}{k}:")
            print(f"{'  '*(level+1)}{v}")
