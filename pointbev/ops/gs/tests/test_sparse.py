import pdb
import time
from math import prod
from sys import path

import pytest
import torch

from tests.fixtures import *

from .utils import *

path.insert(0, "../")
from functions import sparsed_grid_sample, torch_grid_sample

N_REPEAT = 1000
W_MASK = False
W_MODULE = False


# @memory_decorator
@time_decorator(N_REPEAT)
def gs_torch_fw(is_2d, input, grid, mask):
    if W_MODULE:
        out = torch_grid_sample(input, grid, 0, 0, False)
    else:
        out = torch.nn.functional.grid_sample(input, grid, "bilinear", "zeros", False)
    if W_MASK:
        if is_2d:
            out = rearrange(out, "b c h w -> b h w c")[mask.bool()]
        else:
            out = rearrange(out, "b c d h w -> b d h w c")[mask.bool()]
    return out


# @memory_decorator
@time_decorator(N_REPEAT)
def gs_torch_bw(is_2d, input, grid, mask):
    out = torch.nn.functional.grid_sample(input, grid, "bilinear", "zeros", False)
    if W_MASK:
        if is_2d:
            out = rearrange(out, "b c h w -> b h w c")[mask.bool()]
        else:
            out = rearrange(out, "b c d h w -> b d h w c")[mask.bool()]
    out.mean().backward(retain_graph=True)
    return out


def gs_torch(is_2d, input, grid, mask):
    out_fw = gs_torch_fw(is_2d, input, grid, mask)
    out_bw = gs_torch_bw(is_2d, input, grid, mask)
    return out_fw, out_bw


# @memory_decorator
@time_decorator(N_REPEAT)
def gs_sparse_pckg_fw(input, grid, index_batch):
    out = sparsed_grid_sample(input, grid, index_batch, 0, 0, False)
    return out


# @memory_decorator
@time_decorator(N_REPEAT)
def gs_sparse_pckg_bw(input, grid, index_batch):
    out = sparsed_grid_sample(input, grid, index_batch, 0, 0, False)
    out.mean().backward(retain_graph=True)
    return out


def gs_sparse_pckg(input, grid, index_batch):
    out_pckg_fw = gs_sparse_pckg_fw(input, grid, index_batch)
    out_pckg_bw = gs_sparse_pckg_bw(input, grid, index_batch)
    return out_pckg_fw, out_pckg_bw


@pytest.mark.parametrize("is_2d, w_torch, w_pckg", [(True, True, True)])
def test_compare_sparse(input_data, is_2d, w_torch, w_pckg):
    rand, grid, mask, dict_ = input_data

    # Inputs
    if is_2d:
        rand = rand[:, :, :, 0].contiguous()
        grid = grid[..., 0, :, :2].contiguous()
        mask = mask[..., 0, :].contiguous()
    if w_torch:
        inp_torch, grid_torch, grid_mask_torch = get_input(rand, grid, mask)
    if w_pckg:
        inp_pckg, grid_pckg, grid_mask_pckg = get_input(rand, grid, mask)
        index_batch = torch.arange(
            inp_pckg.shape[0], device=inp_pckg.device, dtype=torch.int16
        ).repeat_interleave(prod(grid_pckg.shape[1:-1]))
        index_batch = index_batch[grid_mask_pckg.view(-1).bool()].contiguous()
        grid_pckg_ = grid_pckg.view(-1, 3 if not is_2d else 2)[
            grid_mask_pckg.view(-1).bool()
        ].contiguous()
        reset_mem()

    # fmt: off
    print(f"\nPct: {dict_['pct']}")
    if w_torch:
        if is_2d:
            assert not W_MODULE
        out_torch_fw, out_torch_bw = gs_torch(is_2d, inp_torch,grid_torch,grid_mask_torch)
    if w_pckg:
        out_sparse_fw, out_sparse_bw = gs_sparse_pckg(inp_pckg,grid_pckg_,index_batch)
    print('\n================================')
    # fmt: on

    if w_torch and w_pckg and W_MASK:
        assert out_torch_fw.shape == out_sparse_fw.shape
        assert torch.equal(out_torch_fw, out_sparse_fw)

        assert out_torch_bw.shape == out_sparse_bw.shape
        assert torch.equal(out_torch_bw, out_sparse_bw)

        assert (inp_torch.grad is not None) and (inp_pckg.grad is not None)
        assert torch.allclose(inp_torch.grad, inp_pckg.grad)

        assert (grid_torch.grad is not None) and (grid_pckg.grad is not None)
        assert torch.allclose(grid_torch.grad, grid_pckg.grad)

    return


@pytest.mark.parametrize(
    "dtype_inp_grid, dtype_index",
    [
        ("half", "half"),
        ("half", "int"),
        ("half", "long"),
        ("float", "half"),
        ("float", "int"),
        ("float", "long"),
    ],
)
def test_dtype(dtype_inp_grid, dtype_index):
    # N, C, D, H, W, Npts = 64, 128, 1, 28, 60, 18_000_000
    N, C, D, H, W, Npts = 1, 128, 1, 28, 60, 10

    if dtype_inp_grid == "float":
        dtype_inp_grid = torch.float32
    elif dtype_inp_grid == "double":
        dtype_inp_grid = torch.float64
    elif dtype_inp_grid == "half":
        dtype_inp_grid = torch.float16

    if dtype_index == "int":
        dtype_index = torch.int32
    elif dtype_index == "long":
        dtype_index = torch.int64
    elif dtype_index == "half":
        dtype_index = torch.int16

    inp_pckg = torch.randn(
        N, C, D, H, W, device="cuda", dtype=dtype_inp_grid, requires_grad=True
    )
    grid_pckg_ = torch.rand(Npts, C, device="cuda", dtype=dtype_inp_grid)
    index_batch = torch.randint(0, N, (Npts,), device="cuda", dtype=dtype_index)

    gs_sparse_pckg(inp_pckg, grid_pckg_, index_batch)
    del inp_pckg, grid_pckg_, index_batch
    reset_mem()
