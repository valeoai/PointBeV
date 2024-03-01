import functools
import time

import torch
from einops import rearrange


def time_decorator(N):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal N
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            delta = 0
            for _ in range(N):
                start.record()
                result = func(*args, **kwargs)
                end.record()
                torch.cuda.synchronize()
                if _ <= 5:  # Warmup
                    pass
                else:
                    delta += start.elapsed_time(end) * 1e-3

            print(f"{func.__name__} took {delta:.4f} seconds")
            return result

        return wrapper

    return decorator


def memory_decorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        reset_mem()
        result = func(*args, **kwargs)
        print(f"Max mem {func.__name__}: {torch.cuda.memory_allocated()/(2**30):.4f}")
        return result

    return wrapper


def reset_mem():
    torch.cuda.torch.cuda.reset_peak_memory_stats("cuda")
    torch.cuda.torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()


def get_input(rand, grid, mask=None):
    """Arange input shape to be compatible with the gs."""
    # Inputs
    input = rand.flatten(0, 1).clone().requires_grad_()
    grid = grid.flatten(0, 1).requires_grad_()
    mask = mask.flatten(0, 1) if mask is not None else None
    return input, grid, mask


def set_inp_to_inf(inp, inp_mask, mask):
    NS, C, D, H, W = inp.shape

    inp = rearrange(inp, "ns c d h w -> (ns d h w) c")
    inp_mask = rearrange(inp_mask, "ns d h w -> (ns d h w)")
    inp = inp.detach()
    inp[~inp_mask.bool()] = torch.inf
    inp = rearrange(inp, "(ns d h w) c -> ns c d h w", ns=NS, c=C, d=D, h=H, w=W)
    # mask = torch.where(mask == 1, mask, torch.nan)
    mask = torch.where(mask == 1, mask, torch.inf)

    return inp.clone().requires_grad_(), mask


def set_inf_to_zero(out):
    out = torch.nan_to_num(out, 0.0)
    out = torch.where(out != torch.inf, out, 0.0)
    out = torch.where(out < 1e38, out, 0.0)
    out = torch.where(out > -1e38, out, 0.0)
    return out
