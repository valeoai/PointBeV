import functools

import torch
from pytorch_lightning.utilities import rank_zero_only

from .object import print_shape


def execute_once(is_hook=True):
    def decorator(f):
        dict_cnt = {}

        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            nonlocal dict_cnt
            if is_hook:
                module = args[0]
                name = module.__class__.__name__
            else:
                name = f.__qualname__
            if name not in dict_cnt.keys():
                dict_cnt[name] = 1
                return f(*args, **kwargs)

        return wrapper

    return decorator


@execute_once()
@rank_zero_only
def debug_hook(module, input, output):
    """Note: torch hooks do not work with kwargs passed as inputs."""
    print("Class:", module.__class__.__name__)

    print("Inputs:")
    print_shape(input)

    print("Outputs:")
    print_shape(output)

    torch.cuda.reset_peak_memory_stats()
    print()
