#include <ATen/OpMathType.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/cuda/detail/KernelUtils.h>
#include <ATen/cuda/detail/TensorInfo.cuh>
#include <ATen/native/GridSamplerUtils.h>
#include <ATen/native/cuda/GridSampler.cuh>
#include <ATen/native/cuda/GridSampler.h>

#include "check.h"

using namespace at::cuda::detail;
using at::native::detail::GridSamplerInterpolation;
using at::native::detail::GridSamplerPadding;

// ----------------------------------------------
// Kernels: sparsed
// ----------------------------------------------
template <typename scalar_t, typename index_t>
C10_LAUNCH_BOUNDS_1(512)
__global__ void sparsed_gs_3d_fw_kernel(
    const index_t nthreads, TensorInfo<scalar_t, index_t> input, TensorInfo<scalar_t, index_t> grid,
    const torch::PackedTensorAccessor<int16_t, 1, torch::RestrictPtrTraits, size_t> index_batch_ptr,
    TensorInfo<scalar_t, index_t> output, const GridSamplerInterpolation interpolation_mode,
    const GridSamplerPadding padding_mode, bool align_corners)
{
  const index_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= output.sizes[0])
    return;

  using opmath_t = at::opmath_type<scalar_t>;
  index_t C = input.sizes[1];
  index_t inp_D = input.sizes[2];
  index_t inp_H = input.sizes[3];
  index_t inp_W = input.sizes[4];
  index_t out_Npts = grid.sizes[0];
  index_t inp_sN = input.strides[0];
  index_t inp_sC = input.strides[1];
  index_t inp_sD = input.strides[2];
  index_t inp_sH = input.strides[3];
  index_t inp_sW = input.strides[4];
  index_t grid_sNpts = grid.strides[0];
  index_t grid_sCoor = grid.strides[1];
  index_t out_sNpts = output.strides[0];
  index_t out_sC = output.strides[1];

  const index_t grid_offset = index * grid_sNpts;

  // get the corresponding input x, y, z co-ordinates from grid
  opmath_t x = grid.data[grid_offset];
  opmath_t y = grid.data[grid_offset + grid_sCoor];
  opmath_t z = grid.data[grid_offset + 2 * grid_sCoor];
  index_t n = index_batch_ptr[index];

  opmath_t ix =
      at::native::grid_sampler_compute_source_index(x, inp_W, padding_mode, align_corners);
  opmath_t iy =
      at::native::grid_sampler_compute_source_index(y, inp_H, padding_mode, align_corners);
  opmath_t iz =
      at::native::grid_sampler_compute_source_index(z, inp_D, padding_mode, align_corners);

  if (interpolation_mode == GridSamplerInterpolation::Bilinear)
  {
    // get corner pixel values from (x, y, z)
    // for 4d, we used north-east-south-west
    // for 5d, we add top-bottom
    index_t ix_tnw = static_cast<index_t>(::floor(ix));
    index_t iy_tnw = static_cast<index_t>(::floor(iy));
    index_t iz_tnw = static_cast<index_t>(::floor(iz));

    index_t ix_tne = ix_tnw + 1;
    index_t iy_tne = iy_tnw;
    index_t iz_tne = iz_tnw;

    index_t ix_tsw = ix_tnw;
    index_t iy_tsw = iy_tnw + 1;
    index_t iz_tsw = iz_tnw;

    index_t ix_tse = ix_tnw + 1;
    index_t iy_tse = iy_tnw + 1;
    index_t iz_tse = iz_tnw;

    index_t ix_bnw = ix_tnw;
    index_t iy_bnw = iy_tnw;
    index_t iz_bnw = iz_tnw + 1;

    index_t ix_bne = ix_tnw + 1;
    index_t iy_bne = iy_tnw;
    index_t iz_bne = iz_tnw + 1;

    index_t ix_bsw = ix_tnw;
    index_t iy_bsw = iy_tnw + 1;
    index_t iz_bsw = iz_tnw + 1;

    index_t ix_bse = ix_tnw + 1;
    index_t iy_bse = iy_tnw + 1;
    index_t iz_bse = iz_tnw + 1;

    // get surfaces to each neighbor:
    opmath_t tnw = (ix_bse - ix) * (iy_bse - iy) * (iz_bse - iz);
    opmath_t tne = (ix - ix_bsw) * (iy_bsw - iy) * (iz_bsw - iz);
    opmath_t tsw = (ix_bne - ix) * (iy - iy_bne) * (iz_bne - iz);
    opmath_t tse = (ix - ix_bnw) * (iy - iy_bnw) * (iz_bnw - iz);
    opmath_t bnw = (ix_tse - ix) * (iy_tse - iy) * (iz - iz_tse);
    opmath_t bne = (ix - ix_tsw) * (iy_tsw - iy) * (iz - iz_tsw);
    opmath_t bsw = (ix_tne - ix) * (iy - iy_tne) * (iz - iz_tne);
    opmath_t bse = (ix - ix_tnw) * (iy - iy_tnw) * (iz - iz_tnw);

    auto inp_ptr_NC = input.data + n * inp_sN;
    auto out_ptr_NCDHW = output.data + index * out_sNpts;
    for (index_t c = 0; c < C; ++c, inp_ptr_NC += inp_sC, out_ptr_NCDHW += out_sC)
    {
      opmath_t out_acc = 0;
      if (at::native::within_bounds_3d(iz_tnw, iy_tnw, ix_tnw, inp_D, inp_H, inp_W))
      {
        out_acc += inp_ptr_NC[iz_tnw * inp_sD + iy_tnw * inp_sH + ix_tnw * inp_sW] * tnw;
      }
      if (at::native::within_bounds_3d(iz_tne, iy_tne, ix_tne, inp_D, inp_H, inp_W))
      {
        out_acc += inp_ptr_NC[iz_tne * inp_sD + iy_tne * inp_sH + ix_tne * inp_sW] * tne;
      }
      if (at::native::within_bounds_3d(iz_tsw, iy_tsw, ix_tsw, inp_D, inp_H, inp_W))
      {
        out_acc += inp_ptr_NC[iz_tsw * inp_sD + iy_tsw * inp_sH + ix_tsw * inp_sW] * tsw;
      }
      if (at::native::within_bounds_3d(iz_tse, iy_tse, ix_tse, inp_D, inp_H, inp_W))
      {
        out_acc += inp_ptr_NC[iz_tse * inp_sD + iy_tse * inp_sH + ix_tse * inp_sW] * tse;
      }
      if (at::native::within_bounds_3d(iz_bnw, iy_bnw, ix_bnw, inp_D, inp_H, inp_W))
      {
        out_acc += inp_ptr_NC[iz_bnw * inp_sD + iy_bnw * inp_sH + ix_bnw * inp_sW] * bnw;
      }
      if (at::native::within_bounds_3d(iz_bne, iy_bne, ix_bne, inp_D, inp_H, inp_W))
      {
        out_acc += inp_ptr_NC[iz_bne * inp_sD + iy_bne * inp_sH + ix_bne * inp_sW] * bne;
      }
      if (at::native::within_bounds_3d(iz_bsw, iy_bsw, ix_bsw, inp_D, inp_H, inp_W))
      {
        out_acc += inp_ptr_NC[iz_bsw * inp_sD + iy_bsw * inp_sH + ix_bsw * inp_sW] * bsw;
      }
      if (at::native::within_bounds_3d(iz_bse, iy_bse, ix_bse, inp_D, inp_H, inp_W))
      {
        out_acc += inp_ptr_NC[iz_bse * inp_sD + iy_bse * inp_sH + ix_bse * inp_sW] * bse;
      }
      *out_ptr_NCDHW = out_acc;
    }
  }
}

template <typename scalar_t, typename index_t>
C10_LAUNCH_BOUNDS_1(256)
__global__ void sparsed_gs_2d_fw_kernel(
    const index_t nthreads, TensorInfo<scalar_t, index_t> input, TensorInfo<scalar_t, index_t> grid,
    const torch::PackedTensorAccessor<int16_t, 1, torch::RestrictPtrTraits, size_t> index_batch_ptr,
    TensorInfo<scalar_t, index_t> output, const GridSamplerInterpolation interpolation_mode,
    const GridSamplerPadding padding_mode, bool align_corners)
{
  const index_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= output.sizes[0])
    return;

  using opmath_t = at::opmath_type<scalar_t>;
  index_t C = input.sizes[1];
  index_t inp_H = input.sizes[2];
  index_t inp_W = input.sizes[3];
  index_t out_Npts = grid.sizes[0];
  index_t inp_sN = input.strides[0];
  index_t inp_sC = input.strides[1];
  index_t inp_sH = input.strides[2];
  index_t inp_sW = input.strides[3];
  index_t grid_sNpts = grid.strides[0];
  index_t grid_sCoor = grid.strides[1];
  index_t out_sNpts = output.strides[0];
  index_t out_sC = output.strides[1];

  const index_t grid_offset = index * grid_sNpts;

  // get the corresponding input x, y, z co-ordinates from grid
  opmath_t x = grid.data[grid_offset];
  opmath_t y = grid.data[grid_offset + grid_sCoor];
  index_t n = index_batch_ptr[index];

  opmath_t ix =
      at::native::grid_sampler_compute_source_index(x, inp_W, padding_mode, align_corners);
  opmath_t iy =
      at::native::grid_sampler_compute_source_index(y, inp_H, padding_mode, align_corners);

  if (interpolation_mode == GridSamplerInterpolation::Bilinear)
  {
    index_t ix_nw = static_cast<index_t>(::floor(ix));
    index_t iy_nw = static_cast<index_t>(::floor(iy));
    index_t ix_ne = ix_nw + 1;
    index_t iy_ne = iy_nw;
    index_t ix_sw = ix_nw;
    index_t iy_sw = iy_nw + 1;
    index_t ix_se = ix_nw + 1;
    index_t iy_se = iy_nw + 1;

    // get surfaces to each neighbor:
    opmath_t nw = (ix_se - ix) * (iy_se - iy);
    opmath_t ne = (ix - ix_sw) * (iy_sw - iy);
    opmath_t sw = (ix_ne - ix) * (iy - iy_ne);
    opmath_t se = (ix - ix_nw) * (iy - iy_nw);

    auto inp_ptr_NC = input.data + n * inp_sN;
    auto out_ptr_NCHW = output.data + index * out_sNpts;
    for (index_t c = 0; c < C; ++c, inp_ptr_NC += inp_sC, out_ptr_NCHW += out_sC)
    {
      opmath_t out_acc = 0;
      if (at::native::within_bounds_2d(iy_nw, ix_nw, inp_H, inp_W))
      {
        out_acc += inp_ptr_NC[iy_nw * inp_sH + ix_nw * inp_sW] * nw;
      }
      if (at::native::within_bounds_2d(iy_ne, ix_ne, inp_H, inp_W))
      {
        out_acc += inp_ptr_NC[iy_ne * inp_sH + ix_ne * inp_sW] * ne;
      }
      if (at::native::within_bounds_2d(iy_sw, ix_sw, inp_H, inp_W))
      {
        out_acc += inp_ptr_NC[iy_sw * inp_sH + ix_sw * inp_sW] * sw;
      }
      if (at::native::within_bounds_2d(iy_se, ix_se, inp_H, inp_W))
      {
        out_acc += inp_ptr_NC[iy_se * inp_sH + ix_se * inp_sW] * se;
      }
      *out_ptr_NCHW = out_acc;
    }
  }
}

template <typename scalar_t, typename index_t>
C10_LAUNCH_BOUNDS_1(512)
__global__ void sparsed_gs_3d_bw_kernel(
    const index_t nthreads, TensorInfo<scalar_t, index_t> grad_output,
    TensorInfo<scalar_t, index_t> input, TensorInfo<scalar_t, index_t> grid,
    const torch::PackedTensorAccessor<int16_t, 1, torch::RestrictPtrTraits, size_t> index_batch_ptr,
    TensorInfo<scalar_t, index_t> grad_input, TensorInfo<scalar_t, index_t> grad_grid,
    const GridSamplerInterpolation interpolation_mode, const GridSamplerPadding padding_mode,
    bool align_corners, const index_t grad_input_memory_span, const bool input_requires_grad)
{
  const index_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= grad_output.sizes[0])
    return;

  index_t C = input.sizes[1];
  index_t inp_D = input.sizes[2];
  index_t inp_H = input.sizes[3];
  index_t inp_W = input.sizes[4];
  index_t inp_sN = input.strides[0];
  index_t inp_sC = input.strides[1];
  index_t inp_sD = input.strides[2];
  index_t inp_sH = input.strides[3];
  index_t inp_sW = input.strides[4];
  index_t grid_sNpts = grid.strides[0];
  index_t grid_sCoor = grid.strides[1];
  index_t gOut_sNpts = grad_output.strides[0];
  index_t gOut_sC = grad_output.strides[1];
  int64_t gInp_sN = 0;
  int64_t gInp_sC = 0;
  int64_t gInp_sD = 0;
  int64_t gInp_sH = 0;
  int64_t gInp_sW = 0;
  if (input_requires_grad)
  {
    gInp_sN = grad_input.strides[0];
    gInp_sC = grad_input.strides[1];
    gInp_sD = grad_input.strides[2];
    gInp_sH = grad_input.strides[3];
    gInp_sW = grad_input.strides[4];
  }
  index_t gGrid_sNpts = grad_grid.strides[0];

  const auto grid_offset = index * grid_sNpts;

  // get the corresponding input x, y, z co-ordinates from grid
  scalar_t ix = grid.data[grid_offset];
  scalar_t iy = grid.data[grid_offset + grid_sCoor];
  scalar_t iz = grid.data[grid_offset + 2 * grid_sCoor];

  // multipliers for gradients on ix, iy, and iz
  scalar_t gix_mult, giy_mult, giz_mult;
  ix = at::native::grid_sampler_compute_source_index_set_grad(ix, inp_W, padding_mode,
                                                              align_corners, &gix_mult);
  iy = at::native::grid_sampler_compute_source_index_set_grad(iy, inp_H, padding_mode,
                                                              align_corners, &giy_mult);
  iz = at::native::grid_sampler_compute_source_index_set_grad(iz, inp_D, padding_mode,
                                                              align_corners, &giz_mult);

  if (interpolation_mode == GridSamplerInterpolation::Bilinear)
  {
    // get corner pixel values from (x, y, z)
    // for 4d, we used north-east-south-west
    // for 5d, we add top-bottom
    index_t ix_tnw = static_cast<index_t>(std::floor(ix));
    index_t iy_tnw = static_cast<index_t>(std::floor(iy));
    index_t iz_tnw = static_cast<index_t>(std::floor(iz));

    index_t ix_tne = ix_tnw + 1;
    index_t iy_tne = iy_tnw;
    index_t iz_tne = iz_tnw;

    index_t ix_tsw = ix_tnw;
    index_t iy_tsw = iy_tnw + 1;
    index_t iz_tsw = iz_tnw;

    index_t ix_tse = ix_tnw + 1;
    index_t iy_tse = iy_tnw + 1;
    index_t iz_tse = iz_tnw;

    index_t ix_bnw = ix_tnw;
    index_t iy_bnw = iy_tnw;
    index_t iz_bnw = iz_tnw + 1;

    index_t ix_bne = ix_tnw + 1;
    index_t iy_bne = iy_tnw;
    index_t iz_bne = iz_tnw + 1;

    index_t ix_bsw = ix_tnw;
    index_t iy_bsw = iy_tnw + 1;
    index_t iz_bsw = iz_tnw + 1;

    index_t ix_bse = ix_tnw + 1;
    index_t iy_bse = iy_tnw + 1;
    index_t iz_bse = iz_tnw + 1;

    // get surfaces to each neighbor:
    scalar_t tnw = (ix_bse - ix) * (iy_bse - iy) * (iz_bse - iz);
    scalar_t tne = (ix - ix_bsw) * (iy_bsw - iy) * (iz_bsw - iz);
    scalar_t tsw = (ix_bne - ix) * (iy - iy_bne) * (iz_bne - iz);
    scalar_t tse = (ix - ix_bnw) * (iy - iy_bnw) * (iz_bnw - iz);
    scalar_t bnw = (ix_tse - ix) * (iy_tse - iy) * (iz - iz_tse);
    scalar_t bne = (ix - ix_tsw) * (iy_tsw - iy) * (iz - iz_tsw);
    scalar_t bsw = (ix_tne - ix) * (iy - iy_tne) * (iz - iz_tne);
    scalar_t bse = (ix - ix_tnw) * (iy - iy_tnw) * (iz - iz_tnw);

    scalar_t gix = static_cast<scalar_t>(0), giy = static_cast<scalar_t>(0),
             giz = static_cast<scalar_t>(0);

    const index_t n = index_batch_ptr[index];
    scalar_t* gOut_ptr_NCDHW = grad_output.data + index * gOut_sNpts;
    index_t NC_offset;
    if (input_requires_grad)
    {
      NC_offset = n * gInp_sN;
    }
    scalar_t* inp_ptr_NC = input.data + n * inp_sN;
    // calculate bilinear weighted pixel value and set output pixel
    for (index_t c = 0; c < C;
         ++c, gOut_ptr_NCDHW += gOut_sC, NC_offset += gInp_sC, inp_ptr_NC += inp_sC)
    {
      scalar_t gOut = *gOut_ptr_NCDHW;

      if (input_requires_grad)
      {
        at::native::safe_add_3d(grad_input.data, iz_tnw, iy_tnw, ix_tnw, gInp_sD, gInp_sH, gInp_sW,
                                inp_D, inp_H, inp_W, tnw * gOut, NC_offset, grad_input_memory_span);
        at::native::safe_add_3d(grad_input.data, iz_tne, iy_tne, ix_tne, gInp_sD, gInp_sH, gInp_sW,
                                inp_D, inp_H, inp_W, tne * gOut, NC_offset, grad_input_memory_span);
        at::native::safe_add_3d(grad_input.data, iz_tsw, iy_tsw, ix_tsw, gInp_sD, gInp_sH, gInp_sW,
                                inp_D, inp_H, inp_W, tsw * gOut, NC_offset, grad_input_memory_span);
        at::native::safe_add_3d(grad_input.data, iz_tse, iy_tse, ix_tse, gInp_sD, gInp_sH, gInp_sW,
                                inp_D, inp_H, inp_W, tse * gOut, NC_offset, grad_input_memory_span);
        at::native::safe_add_3d(grad_input.data, iz_bnw, iy_bnw, ix_bnw, gInp_sD, gInp_sH, gInp_sW,
                                inp_D, inp_H, inp_W, bnw * gOut, NC_offset, grad_input_memory_span);
        at::native::safe_add_3d(grad_input.data, iz_bne, iy_bne, ix_bne, gInp_sD, gInp_sH, gInp_sW,
                                inp_D, inp_H, inp_W, bne * gOut, NC_offset, grad_input_memory_span);
        at::native::safe_add_3d(grad_input.data, iz_bsw, iy_bsw, ix_bsw, gInp_sD, gInp_sH, gInp_sW,
                                inp_D, inp_H, inp_W, bsw * gOut, NC_offset, grad_input_memory_span);
        at::native::safe_add_3d(grad_input.data, iz_bse, iy_bse, ix_bse, gInp_sD, gInp_sH, gInp_sW,
                                inp_D, inp_H, inp_W, bse * gOut, NC_offset, grad_input_memory_span);
      }
      // calculate grad_grid
      if (at::native::within_bounds_3d(iz_tnw, iy_tnw, ix_tnw, inp_D, inp_H, inp_W))
      {
        scalar_t tnw_val = inp_ptr_NC[iz_tnw * inp_sD + iy_tnw * inp_sH + ix_tnw * inp_sW];
        gix -= tnw_val * (iy_bse - iy) * (iz_bse - iz) * gOut;
        giy -= tnw_val * (ix_bse - ix) * (iz_bse - iz) * gOut;
        giz -= tnw_val * (ix_bse - ix) * (iy_bse - iy) * gOut;
      }
      if (at::native::within_bounds_3d(iz_tne, iy_tne, ix_tne, inp_D, inp_H, inp_W))
      {
        scalar_t tne_val = inp_ptr_NC[iz_tne * inp_sD + iy_tne * inp_sH + ix_tne * inp_sW];
        gix += tne_val * (iy_bsw - iy) * (iz_bsw - iz) * gOut;
        giy -= tne_val * (ix - ix_bsw) * (iz_bsw - iz) * gOut;
        giz -= tne_val * (ix - ix_bsw) * (iy_bsw - iy) * gOut;
      }
      if (at::native::within_bounds_3d(iz_tsw, iy_tsw, ix_tsw, inp_D, inp_H, inp_W))
      {
        scalar_t tsw_val = inp_ptr_NC[iz_tsw * inp_sD + iy_tsw * inp_sH + ix_tsw * inp_sW];
        gix -= tsw_val * (iy - iy_bne) * (iz_bne - iz) * gOut;
        giy += tsw_val * (ix_bne - ix) * (iz_bne - iz) * gOut;
        giz -= tsw_val * (ix_bne - ix) * (iy - iy_bne) * gOut;
      }
      if (at::native::within_bounds_3d(iz_tse, iy_tse, ix_tse, inp_D, inp_H, inp_W))
      {
        scalar_t tse_val = inp_ptr_NC[iz_tse * inp_sD + iy_tse * inp_sH + ix_tse * inp_sW];
        gix += tse_val * (iy - iy_bnw) * (iz_bnw - iz) * gOut;
        giy += tse_val * (ix - ix_bnw) * (iz_bnw - iz) * gOut;
        giz -= tse_val * (ix - ix_bnw) * (iy - iy_bnw) * gOut;
      }
      if (at::native::within_bounds_3d(iz_bnw, iy_bnw, ix_bnw, inp_D, inp_H, inp_W))
      {
        scalar_t bnw_val = inp_ptr_NC[iz_bnw * inp_sD + iy_bnw * inp_sH + ix_bnw * inp_sW];
        gix -= bnw_val * (iy_tse - iy) * (iz - iz_tse) * gOut;
        giy -= bnw_val * (ix_tse - ix) * (iz - iz_tse) * gOut;
        giz += bnw_val * (ix_tse - ix) * (iy_tse - iy) * gOut;
      }
      if (at::native::within_bounds_3d(iz_bne, iy_bne, ix_bne, inp_D, inp_H, inp_W))
      {
        scalar_t bne_val = inp_ptr_NC[iz_bne * inp_sD + iy_bne * inp_sH + ix_bne * inp_sW];
        gix += bne_val * (iy_tsw - iy) * (iz - iz_tsw) * gOut;
        giy -= bne_val * (ix - ix_tsw) * (iz - iz_tsw) * gOut;
        giz += bne_val * (ix - ix_tsw) * (iy_tsw - iy) * gOut;
      }
      if (at::native::within_bounds_3d(iz_bsw, iy_bsw, ix_bsw, inp_D, inp_H, inp_W))
      {
        scalar_t bsw_val = inp_ptr_NC[iz_bsw * inp_sD + iy_bsw * inp_sH + ix_bsw * inp_sW];
        gix -= bsw_val * (iy - iy_tne) * (iz - iz_tne) * gOut;
        giy += bsw_val * (ix_tne - ix) * (iz - iz_tne) * gOut;
        giz += bsw_val * (ix_tne - ix) * (iy - iy_tne) * gOut;
      }
      if (at::native::within_bounds_3d(iz_bse, iy_bse, ix_bse, inp_D, inp_H, inp_W))
      {
        scalar_t bse_val = inp_ptr_NC[iz_bse * inp_sD + iy_bse * inp_sH + ix_bse * inp_sW];
        gix += bse_val * (iy - iy_tnw) * (iz - iz_tnw) * gOut;
        giy += bse_val * (ix - ix_tnw) * (iz - iz_tnw) * gOut;
        giz += bse_val * (ix - ix_tnw) * (iy - iy_tnw) * gOut;
      }
    }

    // assuming grad_grid is contiguous
    // thus we can
    //   1. use index with gGrid_sW to directly compute gGrid_ptr_NDHW
    //   2. directly assign to gGrid_ptr_NDHW[0], gGrid_ptr_NDHW[1], gGrid_ptr_NDHW[2]
    scalar_t* gGrid_ptr_NDHW = grad_grid.data + index * gGrid_sNpts;
    gGrid_ptr_NDHW[0] = gix_mult * gix;
    gGrid_ptr_NDHW[1] = giy_mult * giy;
    gGrid_ptr_NDHW[2] = giz_mult * giz;
  }
}

template <typename scalar_t, typename index_t>
C10_LAUNCH_BOUNDS_1(256)
__global__ void sparsed_gs_2d_bw_kernel(
    const index_t nthreads, TensorInfo<scalar_t, index_t> grad_output,
    TensorInfo<scalar_t, index_t> input, TensorInfo<scalar_t, index_t> grid,
    const torch::PackedTensorAccessor<int16_t, 1, torch::RestrictPtrTraits, size_t> index_batch_ptr,
    TensorInfo<scalar_t, index_t> grad_input, TensorInfo<scalar_t, index_t> grad_grid,
    const GridSamplerInterpolation interpolation_mode, const GridSamplerPadding padding_mode,
    bool align_corners, const index_t grad_input_memory_span, const bool input_requires_grad)
{
  const index_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= grad_output.sizes[0])
    return;

  index_t C = input.sizes[1];
  index_t inp_H = input.sizes[2];
  index_t inp_W = input.sizes[3];
  index_t inp_sN = input.strides[0];
  index_t inp_sC = input.strides[1];
  index_t inp_sH = input.strides[2];
  index_t inp_sW = input.strides[3];
  index_t grid_sNpts = grid.strides[0];
  index_t grid_sCoor = grid.strides[1];
  index_t gOut_sNpts = grad_output.strides[0];
  index_t gOut_sC = grad_output.strides[1];
  int64_t gInp_sN = 0;
  int64_t gInp_sC = 0;
  int64_t gInp_sH = 0;
  int64_t gInp_sW = 0;
  if (input_requires_grad)
  {
    gInp_sN = grad_input.strides[0];
    gInp_sC = grad_input.strides[1];
    gInp_sH = grad_input.strides[2];
    gInp_sW = grad_input.strides[3];
  }
  index_t gGrid_sNpts = grad_grid.strides[0];

  const auto grid_offset = index * grid_sNpts;

  // get the corresponding input x, y, z co-ordinates from grid
  scalar_t ix = grid.data[grid_offset];
  scalar_t iy = grid.data[grid_offset + grid_sCoor];

  // multipliers for gradients on ix, iy, and iz
  scalar_t gix_mult, giy_mult, giz_mult;
  ix = at::native::grid_sampler_compute_source_index_set_grad(ix, inp_W, padding_mode,
                                                              align_corners, &gix_mult);
  iy = at::native::grid_sampler_compute_source_index_set_grad(iy, inp_H, padding_mode,
                                                              align_corners, &giy_mult);

  if (interpolation_mode == GridSamplerInterpolation::Bilinear)
  {
    // get NE, NW, SE, SW pixel values from (x, y)
    index_t ix_nw = static_cast<index_t>(std::floor(ix));
    index_t iy_nw = static_cast<index_t>(std::floor(iy));
    index_t ix_ne = ix_nw + 1;
    index_t iy_ne = iy_nw;
    index_t ix_sw = ix_nw;
    index_t iy_sw = iy_nw + 1;
    index_t ix_se = ix_nw + 1;
    index_t iy_se = iy_nw + 1;

    // get surfaces to each neighbor:
    scalar_t nw = (ix_se - ix) * (iy_se - iy);
    scalar_t ne = (ix - ix_sw) * (iy_sw - iy);
    scalar_t sw = (ix_ne - ix) * (iy - iy_ne);
    scalar_t se = (ix - ix_nw) * (iy - iy_nw);

    scalar_t gix = static_cast<scalar_t>(0), giy = static_cast<scalar_t>(0);

    const index_t n = index_batch_ptr[index];
    scalar_t* gOut_ptr_NCHW = grad_output.data + index * gOut_sNpts;
    index_t NC_offset;
    if (input_requires_grad)
    {
      NC_offset = n * gInp_sN;
    }
    scalar_t* inp_ptr_NC = input.data + n * inp_sN;
    // calculate bilinear weighted pixel value and set output pixel
    for (index_t c = 0; c < C;
         ++c, gOut_ptr_NCHW += gOut_sC, NC_offset += gInp_sC, inp_ptr_NC += inp_sC)
    {
      scalar_t gOut = *gOut_ptr_NCHW;

      if (input_requires_grad)
      {
        // calculate and set grad_input. See Note [Passing pointer and offset to fastAtomicAdd].
        at::native::safe_add_2d(grad_input.data, iy_nw, ix_nw, gInp_sH, gInp_sW, inp_H, inp_W,
                                nw * gOut, NC_offset, grad_input_memory_span);
        at::native::safe_add_2d(grad_input.data, iy_ne, ix_ne, gInp_sH, gInp_sW, inp_H, inp_W,
                                ne * gOut, NC_offset, grad_input_memory_span);
        at::native::safe_add_2d(grad_input.data, iy_sw, ix_sw, gInp_sH, gInp_sW, inp_H, inp_W,
                                sw * gOut, NC_offset, grad_input_memory_span);
        at::native::safe_add_2d(grad_input.data, iy_se, ix_se, gInp_sH, gInp_sW, inp_H, inp_W,
                                se * gOut, NC_offset, grad_input_memory_span);
      }
      // calculate grad_grid
      if (at::native::within_bounds_2d(iy_nw, ix_nw, inp_H, inp_W))
      {
        scalar_t nw_val = inp_ptr_NC[iy_nw * inp_sH + ix_nw * inp_sW];
        gix -= nw_val * (iy_se - iy) * gOut;
        giy -= nw_val * (ix_se - ix) * gOut;
      }
      if (at::native::within_bounds_2d(iy_ne, ix_ne, inp_H, inp_W))
      {
        scalar_t ne_val = inp_ptr_NC[iy_ne * inp_sH + ix_ne * inp_sW];
        gix += ne_val * (iy_sw - iy) * gOut;
        giy -= ne_val * (ix - ix_sw) * gOut;
      }
      if (at::native::within_bounds_2d(iy_sw, ix_sw, inp_H, inp_W))
      {
        scalar_t sw_val = inp_ptr_NC[iy_sw * inp_sH + ix_sw * inp_sW];
        gix -= sw_val * (iy - iy_ne) * gOut;
        giy += sw_val * (ix_ne - ix) * gOut;
      }
      if (at::native::within_bounds_2d(iy_se, ix_se, inp_H, inp_W))
      {
        scalar_t se_val = inp_ptr_NC[iy_se * inp_sH + ix_se * inp_sW];
        gix += se_val * (iy - iy_nw) * gOut;
        giy += se_val * (ix - ix_nw) * gOut;
      }
    }

    // assuming grad_grid is contiguous
    // thus we can
    //   1. use index with gGrid_sW to directly compute gGrid_ptr_NDHW
    //   2. directly assign to gGrid_ptr_NDHW[0], gGrid_ptr_NDHW[1], gGrid_ptr_NDHW[2]
    scalar_t* gGrid_ptr_NDHW = grad_grid.data + index * gGrid_sNpts;
    gGrid_ptr_NDHW[0] = gix_mult * gix;
    gGrid_ptr_NDHW[1] = giy_mult * giy;
  }
}

// ----------------------------------------------
// Launchers
// ----------------------------------------------
void launch_sparsed_gs_3d_fw_kernel(const at::TensorBase& output, const at::TensorBase& input,
                                    const at::TensorBase& grid, const torch::Tensor& index_batch,
                                    int64_t interpolation_mode, int64_t padding_mode,
                                    bool align_corners)
{
  int64_t count = output.size(0);
  if (count > 0)
  {
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        input.scalar_type(), "sparsed_gs_fw_kernel",
        [&]
        {
          if (canUse32BitIndexMath(input) && canUse32BitIndexMath(grid) &&
              canUse32BitIndexMath(output))
          {
            sparsed_gs_3d_fw_kernel<scalar_t>
                <<<GET_BLOCKS(count, 512), 512, 0, at::cuda::getCurrentCUDAStream()>>>(
                    static_cast<int>(count), getTensorInfo<scalar_t, int>(input),
                    getTensorInfo<scalar_t, int>(grid),
                    index_batch.packed_accessor<int16_t, 1, torch::RestrictPtrTraits, size_t>(),
                    getTensorInfo<scalar_t, int>(output),
                    static_cast<GridSamplerInterpolation>(interpolation_mode),
                    static_cast<GridSamplerPadding>(padding_mode), align_corners);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
          }
          else
          {
            sparsed_gs_3d_fw_kernel<scalar_t>
                <<<GET_BLOCKS(count, 512), 512, 0, at::cuda::getCurrentCUDAStream()>>>(
                    count, getTensorInfo<scalar_t, int64_t>(input),
                    getTensorInfo<scalar_t, int64_t>(grid),
                    index_batch.packed_accessor<int16_t, 1, torch::RestrictPtrTraits, size_t>(),
                    getTensorInfo<scalar_t, int64_t>(output),
                    static_cast<GridSamplerInterpolation>(interpolation_mode),
                    static_cast<GridSamplerPadding>(padding_mode), align_corners);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
          }
        });
  }
}
void launch_sparsed_gs_2d_fw_kernel(const at::TensorBase& output, const at::TensorBase& input,
                                    const at::TensorBase& grid, const torch::Tensor& index_batch,
                                    int64_t interpolation_mode, int64_t padding_mode,
                                    bool align_corners)
{
  int64_t count = output.size(0);
  if (count > 0)
  {
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        input.scalar_type(), "sparsed_gs_2d_fw_kernel",
        [&]
        {
          if (canUse32BitIndexMath(input) && canUse32BitIndexMath(grid) &&
              canUse32BitIndexMath(output))
          {
            sparsed_gs_2d_fw_kernel<scalar_t>
                <<<GET_BLOCKS(count, 256), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
                    static_cast<int>(count), getTensorInfo<scalar_t, int>(input),
                    getTensorInfo<scalar_t, int>(grid),
                    index_batch.packed_accessor<int16_t, 1, torch::RestrictPtrTraits, size_t>(),
                    getTensorInfo<scalar_t, int>(output),
                    static_cast<GridSamplerInterpolation>(interpolation_mode),
                    static_cast<GridSamplerPadding>(padding_mode), align_corners);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
          }
          else
          {
            sparsed_gs_2d_fw_kernel<scalar_t>
                <<<GET_BLOCKS(count, 256), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
                    count, getTensorInfo<scalar_t, int64_t>(input),
                    getTensorInfo<scalar_t, int64_t>(grid),
                    index_batch.packed_accessor<int16_t, 1, torch::RestrictPtrTraits, size_t>(),
                    getTensorInfo<scalar_t, int64_t>(output),
                    static_cast<GridSamplerInterpolation>(interpolation_mode),
                    static_cast<GridSamplerPadding>(padding_mode), align_corners);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
          }
        });
  }
}

void launch_sparsed_gs_3d_bw_kernel(const at::TensorBase& grad_input,
                                    const at::TensorBase& grad_grid,
                                    const at::TensorBase& grad_output, const at::TensorBase& input,
                                    const at::TensorBase& grid, const torch::Tensor& index_batch,
                                    int64_t interpolation_mode, int64_t padding_mode,
                                    bool align_corners, std::array<bool, 2> output_mask)
{
  at::globalContext().alertNotDeterministic("sparsed_gs_3d_bw_kernel");

  // grid: Npts, Coord
  int64_t count = grid.size(0);
  auto input_requires_grad = output_mask[0];
  int16_t NUM_THREADS = 512;

  // clang-format off
    if (count > 0) {
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "sparsed_gs_3d_bw_kernel", [&] {
      if (
          at::native::canUse32BitIndexMath(input) && 
          at::native::canUse32BitIndexMath(grid) &&
          at::native::canUse32BitIndexMath(grad_output)) 
          {
        sparsed_gs_3d_bw_kernel<scalar_t>
          <<<at::cuda::detail::GET_BLOCKS(count, NUM_THREADS), NUM_THREADS, 0, 
          at::cuda::getCurrentCUDAStream()>>>(
            static_cast<int>(count),
            getTensorInfo<scalar_t, int>(grad_output),
            getTensorInfo<scalar_t, int>(input),
            getTensorInfo<scalar_t, int>(grid),
            index_batch.packed_accessor<int16_t, 1, torch::RestrictPtrTraits, size_t>(),
            input_requires_grad ? getTensorInfo<scalar_t, int>(grad_input) : TensorInfo<scalar_t, int>(),
            getTensorInfo<scalar_t, int>(grad_grid),
            static_cast<GridSamplerInterpolation>(interpolation_mode),
            static_cast<GridSamplerPadding>(padding_mode),
            align_corners,
            /*grad_input_memory_span =*/input_requires_grad ? static_cast<int>(grad_input.numel()) : 0,
            input_requires_grad);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      } else {
        sparsed_gs_3d_bw_kernel<scalar_t>
          <<<at::cuda::detail::GET_BLOCKS(count, NUM_THREADS), NUM_THREADS, 0, 
          at::cuda::getCurrentCUDAStream()>>>(
            count,
            getTensorInfo<scalar_t, int64_t>(grad_output),
            getTensorInfo<scalar_t, int64_t>(input),
            getTensorInfo<scalar_t, int64_t>(grid),
            index_batch.packed_accessor<int16_t, 1, torch::RestrictPtrTraits, size_t>(),
            input_requires_grad ? getTensorInfo<scalar_t, int64_t>(grad_input) : TensorInfo<scalar_t, int64_t>(),
            getTensorInfo<scalar_t, int64_t>(grad_grid),
            static_cast<GridSamplerInterpolation>(interpolation_mode),
            static_cast<GridSamplerPadding>(padding_mode),
            align_corners,
            /*grad_input_memory_span =*/input_requires_grad ? grad_input.numel() : 0,
            input_requires_grad);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      }
    });
  }
}

void launch_sparsed_gs_2d_bw_kernel(const at::TensorBase& grad_input,
                                    const at::TensorBase& grad_grid,
                                    const at::TensorBase& grad_output, const at::TensorBase& input,
                                    const at::TensorBase& grid, const torch::Tensor& index_batch,
                                    int64_t interpolation_mode, int64_t padding_mode,
                                    bool align_corners, std::array<bool, 2> output_mask)
{
  at::globalContext().alertNotDeterministic("sparsed_gs_2d_bw_kernel");

  // grid: Npts, Coord
  int64_t count = grid.size(0);
  auto input_requires_grad = output_mask[0];
  int16_t NUM_THREADS = 256;

  // clang-format off
    if (count > 0) {
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "sparsed_gs_2d_bw_kernel", [&] {
      if (
          at::native::canUse32BitIndexMath(input) && 
          at::native::canUse32BitIndexMath(grid) &&
          at::native::canUse32BitIndexMath(grad_output)) 
          {
        sparsed_gs_2d_bw_kernel<scalar_t>
          <<<at::cuda::detail::GET_BLOCKS(count, NUM_THREADS), NUM_THREADS, 0, 
          at::cuda::getCurrentCUDAStream()>>>(
            static_cast<int>(count),
            getTensorInfo<scalar_t, int>(grad_output),
            getTensorInfo<scalar_t, int>(input),
            getTensorInfo<scalar_t, int>(grid),
            index_batch.packed_accessor<int16_t, 1, torch::RestrictPtrTraits, size_t>(),
            input_requires_grad ? getTensorInfo<scalar_t, int>(grad_input) : TensorInfo<scalar_t, int>(),
            getTensorInfo<scalar_t, int>(grad_grid),
            static_cast<GridSamplerInterpolation>(interpolation_mode),
            static_cast<GridSamplerPadding>(padding_mode),
            align_corners,
            /*grad_input_memory_span =*/input_requires_grad ? static_cast<int>(grad_input.numel()) : 0,
            input_requires_grad);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      } else {
        sparsed_gs_2d_bw_kernel<scalar_t>
          <<<at::cuda::detail::GET_BLOCKS(count, NUM_THREADS), NUM_THREADS, 0, 
          at::cuda::getCurrentCUDAStream()>>>(
            count,
            getTensorInfo<scalar_t, int64_t>(grad_output),
            getTensorInfo<scalar_t, int64_t>(input),
            getTensorInfo<scalar_t, int64_t>(grid),
            index_batch.packed_accessor<int16_t, 1, torch::RestrictPtrTraits, size_t>(),
            input_requires_grad ? getTensorInfo<scalar_t, int64_t>(grad_input) : TensorInfo<scalar_t, int64_t>(),
            getTensorInfo<scalar_t, int64_t>(grad_grid),
            static_cast<GridSamplerInterpolation>(interpolation_mode),
            static_cast<GridSamplerPadding>(padding_mode),
            align_corners,
            /*grad_input_memory_span =*/input_requires_grad ? grad_input.numel() : 0,
            input_requires_grad);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      }
    });
  }
}
