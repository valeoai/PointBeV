#include "gs.h"
#include <torch/extension.h>

// Forward
torch::Tensor sparsed_gs_3d_fw_cuda(const torch::Tensor& input, const torch::Tensor& grid,
                                    const torch::Tensor& index_batch, int64_t interpolation_mode,
                                    int64_t padding_mode, bool align_corners)
{
  // N*S,C,D,H,W
  auto c = input.size(1);
  auto npts = index_batch.size(0);
  auto output = at::empty({npts, c}, input.options());
  launch_sparsed_gs_3d_fw_kernel(output, input, grid, index_batch, interpolation_mode, padding_mode,
                                 align_corners);
  return output;
}
torch::Tensor sparsed_gs_2d_fw_cuda(const torch::Tensor& input, const torch::Tensor& grid,
                                    const torch::Tensor& index_batch, int64_t interpolation_mode,
                                    int64_t padding_mode, bool align_corners)
{
  // N*S,C,H,W
  auto c = input.size(1);
  auto npts = index_batch.size(0);
  auto output = at::empty({npts, c}, input.options());
  launch_sparsed_gs_2d_fw_kernel(output, input, grid, index_batch, interpolation_mode, padding_mode,
                                 align_corners);
  return output;
}

torch::Tensor torch_gs_3d_fw_cuda(const torch::Tensor& input, const torch::Tensor& grid,
                                  int64_t interpolation_mode, int64_t padding_mode,
                                  bool align_corners)
{
  auto in_size = input.sizes();
  auto grid_size = grid.sizes();
  auto output = at::empty({in_size[0], in_size[1], grid_size[1], grid_size[2], grid_size[3]},
                          input.options());
  launch_torch_gs_3d_fw_kernel(output, input, grid, interpolation_mode, padding_mode,
                               align_corners);
  return output;
}

// Backward
std::tuple<torch::Tensor, torch::Tensor>
sparsed_gs_3d_bw_cuda(const torch::Tensor& grad_output, const torch::Tensor& input,
                      const torch::Tensor& grid, const torch::Tensor& index_batch,
                      int64_t interpolation_mode, int64_t padding_mode, bool align_corners,
                      std::array<bool, 2> output_mask)
{
  auto input_requires_grad = output_mask[0];
  torch::Tensor grad_input = ([&]() {
    if (input_requires_grad)
    {
      return at::zeros_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
    }
    else
    {
      return torch::Tensor();
    }
  })();
  auto grad_grid = at::empty_like(grid, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  launch_sparsed_gs_3d_bw_kernel(grad_input, grad_grid, grad_output, input, grid, index_batch,
                                 interpolation_mode, padding_mode, align_corners, output_mask);
  return std::make_tuple(grad_input, grad_grid);
}

std::tuple<torch::Tensor, torch::Tensor>
sparsed_gs_2d_bw_cuda(const torch::Tensor& grad_output, const torch::Tensor& input,
                      const torch::Tensor& grid, const torch::Tensor& index_batch,
                      int64_t interpolation_mode, int64_t padding_mode, bool align_corners,
                      std::array<bool, 2> output_mask)
{
  auto input_requires_grad = output_mask[0];
  torch::Tensor grad_input = ([&]() {
    if (input_requires_grad)
    {
      return at::zeros_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
    }
    else
    {
      return torch::Tensor();
    }
  })();
  auto grad_grid = at::empty_like(grid, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  launch_sparsed_gs_2d_bw_kernel(grad_input, grad_grid, grad_output, input, grid, index_batch,
                                 interpolation_mode, padding_mode, align_corners, output_mask);
  return std::make_tuple(grad_input, grad_grid);
}

std::tuple<torch::Tensor, torch::Tensor>
torch_gs_3d_bw_cuda(const torch::Tensor& grad_output, const torch::Tensor& input,
                    const torch::Tensor& grid, int64_t interpolation_mode, int64_t padding_mode,
                    bool align_corners, std::array<bool, 2> output_mask)
{
  auto input_requires_grad = output_mask[0];
  torch::Tensor grad_input = ([&]() {
    if (input_requires_grad)
    {
      return at::zeros_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
    }
    else
    {
      return torch::Tensor();
    }
  })();
  auto grad_grid = at::empty_like(grid, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  launch_torch_gs_3d_bw_kernel(grad_input, grad_grid, grad_output, input, grid, interpolation_mode,
                               padding_mode, align_corners, output_mask);
  return std::make_tuple(grad_input, grad_grid);
}
