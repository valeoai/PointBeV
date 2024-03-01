#pragma once
#include <ATen/native/GridSamplerUtils.h>
#include <torch/extension.h>

using at::native::detail::GridSamplerInterpolation;
using at::native::detail::GridSamplerPadding;

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x)                                                                             \
  CHECK_CUDA(x);                                                                                   \
  CHECK_CONTIGUOUS(x)

// ---------------------------------------
// Kernels
// ---------------------------------------
torch::Tensor sparsed_gs_3d_fw_cuda(const torch::Tensor& input, const torch::Tensor& grid,
                                    const torch::Tensor& index_batch, int64_t interpolation_mode,
                                    int64_t padding_mode, bool align_corners);

torch::Tensor sparsed_gs_2d_fw_cuda(const torch::Tensor& input, const torch::Tensor& grid,
                                    const torch::Tensor& index_batch, int64_t interpolation_mode,
                                    int64_t padding_mode, bool align_corners);

torch::Tensor torch_gs_3d_fw_cuda(const torch::Tensor& input, const torch::Tensor& grid,
                                  int64_t interpolation_mode, int64_t padding_mode,
                                  bool align_corners);

// Backward
std::tuple<torch::Tensor, torch::Tensor>
sparsed_gs_3d_bw_cuda(const torch::Tensor& grad_output, const torch::Tensor& input,
                      const torch::Tensor& grid, const torch::Tensor& index_batch,
                      int64_t interpolation_mode, int64_t padding_mode, bool align_corners,
                      std::array<bool, 2> output_mask);

std::tuple<torch::Tensor, torch::Tensor>
sparsed_gs_2d_bw_cuda(const torch::Tensor& grad_output, const torch::Tensor& input,
                      const torch::Tensor& grid, const torch::Tensor& index_batch,
                      int64_t interpolation_mode, int64_t padding_mode, bool align_corners,
                      std::array<bool, 2> output_mask);

std::tuple<torch::Tensor, torch::Tensor>
torch_gs_3d_bw_cuda(const torch::Tensor& grad_output, const torch::Tensor& input,
                    const torch::Tensor& grid, int64_t interpolation_mode, int64_t padding_mode,
                    bool align_corners, std::array<bool, 2> output_mask);

// ---------------------------------------
// Launchers
// ---------------------------------------
void launch_sparsed_gs_3d_fw_kernel(const at::TensorBase& output, const at::TensorBase& input,
                                    const at::TensorBase& grid, const torch::Tensor& index_batch,
                                    int64_t interpolation_mode, int64_t padding_mode,
                                    bool align_corners);

void launch_sparsed_gs_2d_fw_kernel(const at::TensorBase& output, const at::TensorBase& input,
                                    const at::TensorBase& grid, const torch::Tensor& index_batch,
                                    int64_t interpolation_mode, int64_t padding_mode,
                                    bool align_corners);

void launch_torch_gs_3d_fw_kernel(const at::TensorBase& output, const at::TensorBase& input,
                                  const at::TensorBase& grid, int64_t interpolation_mode,
                                  int64_t padding_mode, bool align_corners);

// Backward
void launch_sparsed_gs_3d_bw_kernel(const at::TensorBase& grad_input,
                                    const at::TensorBase& grad_grid,
                                    const at::TensorBase& grad_output, const at::TensorBase& input,
                                    const at::TensorBase& grid, const torch::Tensor& index_batch,
                                    int64_t interpolation_mode, int64_t padding_mode,
                                    bool align_corners, std::array<bool, 2> output_mask);
void launch_sparsed_gs_2d_bw_kernel(const at::TensorBase& grad_input,
                                    const at::TensorBase& grad_grid,
                                    const at::TensorBase& grad_output, const at::TensorBase& input,
                                    const at::TensorBase& grid, const torch::Tensor& index_batch,
                                    int64_t interpolation_mode, int64_t padding_mode,
                                    bool align_corners, std::array<bool, 2> output_mask);

void launch_torch_gs_3d_bw_kernel(const at::TensorBase& grad_input, const at::TensorBase& grad_grid,
                                  const at::TensorBase& grad_output, const at::TensorBase& input,
                                  const at::TensorBase& grid, int64_t interpolation_mode,
                                  int64_t padding_mode, bool align_corners,
                                  std::array<bool, 2> output_mask);