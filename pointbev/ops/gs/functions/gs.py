import pdb

import torch

import sparse_gs  # isort: skip


class SparsedGridSampleFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        input,
        grid,
        index_batch,
        interpolation_mode,
        padding_mode,
        align_corners,
    ):
        if len(input.shape) == 5:
            func = sparse_gs.forward_sparse
        elif len(input.shape) == 4:
            func = sparse_gs.forward_2d_sparse
        out = func(
            input,
            grid,
            index_batch,
            interpolation_mode,
            padding_mode,
            align_corners,
        )
        ctx.save_for_backward(*[input, grid, index_batch])

        ctx.interpolation_mode = interpolation_mode
        ctx.padding_mode = padding_mode
        ctx.align_corners = align_corners
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, grid, index_batch = ctx.saved_tensors
        output_mask = (ctx.needs_input_grad[0], ctx.needs_input_grad[1])
        if len(input.shape) == 5:
            func = sparse_gs.backward_sparse
        elif len(input.shape) == 4:
            func = sparse_gs.backward_2d_sparse
        grad_input, grad_grid = func(
            grad_output.contiguous(),
            input,
            grid,
            index_batch,
            ctx.interpolation_mode,
            ctx.padding_mode,
            ctx.align_corners,
            output_mask,
        )
        return grad_input, grad_grid, None, None, None, None


class TorchGridSampleFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        input,
        grid,
        interpolation_mode,
        padding_mode,
        align_corners,
    ):
        out = sparse_gs.forward_torch(
            input,
            grid,
            interpolation_mode,
            padding_mode,
            align_corners,
        )
        ctx.save_for_backward(*[input, grid])

        ctx.interpolation_mode = interpolation_mode
        ctx.padding_mode = padding_mode
        ctx.align_corners = align_corners
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, grid = ctx.saved_tensors
        output_mask = (ctx.needs_input_grad[0], ctx.needs_input_grad[1])
        grad_input, grad_grid = sparse_gs.backward_sparse(
            grad_output.contiguous(),
            input,
            grid,
            ctx.interpolation_mode,
            ctx.padding_mode,
            ctx.align_corners,
            output_mask,
        )
        return grad_input, grad_grid, None, None, None


# Functions
def sparsed_grid_sample(
    input, grid, index_batch, interpolation_mode=0, padding_mode=0, align_corners=False
):
    return SparsedGridSampleFunction.apply(
        input,
        grid,
        index_batch,
        interpolation_mode,
        padding_mode,
        align_corners,
    )


def torch_grid_sample(
    input, grid, interpolation_mode=0, padding_mode=0, align_corners=False
):
    return TorchGridSampleFunction.apply(
        input,
        grid,
        interpolation_mode,
        padding_mode,
        align_corners,
    )
