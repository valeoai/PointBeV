#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/detail/KernelUtils.h>
#include <c10/macros/Macros.h>
#include <torch/extension.h>

template <typename scalar_t>
C10_LAUNCH_BOUNDS_1(512)
__global__ void find_indices_kernel(
    int64_t* __restrict__ index_q, int64_t* __restrict__ index_k,
    const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> index_activ,
    const torch::PackedTensorAccessor64<scalar_t, 4, torch::RestrictPtrTraits> img_mask,
    std::tuple<int, int, int> ws, bool only_last_z)
{
  // index_q: [Nout]
  int ws_z = std::get<0>(ws);
  int ws_x = std::get<1>(ws);
  int ws_y = std::get<2>(ws);
  int64_t ws_prod = ws_z * ws_x * ws_y;
  int64_t nactiv = index_activ.size(0);
  scalar_t Z = img_mask.size(1);
  scalar_t X = img_mask.size(2);
  scalar_t Y = img_mask.size(3);

  // Threads
  int64_t elem = threadIdx.x + blockIdx.x * blockDim.x;

  if (elem >= nactiv)
  {
    return;
  }

  scalar_t range_z = static_cast<scalar_t>((ws_z - 1) / 2);
  scalar_t range_x = static_cast<scalar_t>((ws_x - 1) / 2);
  scalar_t range_y = static_cast<scalar_t>((ws_y - 1) / 2);

  scalar_t q_id_b = index_activ[elem][0];
  scalar_t q_id_z = index_activ[elem][1];
  scalar_t q_id_x = index_activ[elem][2];
  scalar_t q_id_y = index_activ[elem][3];
  if (only_last_z && (q_id_z != (Z - 1)))
  {
    return;
  }

  int64_t cnt = 0;
  for (int64_t iz = -range_z; iz <= range_z; iz++)
  {
    for (int64_t ix = -range_x; ix <= range_x; ix++)
    {
      for (int64_t iy = -range_y; iy <= range_y; iy++)
      {
        int32_t k_id_z = q_id_z + iz;
        int32_t k_id_x = q_id_x + ix;
        int32_t k_id_y = q_id_y + iy;

        if ((k_id_z < 0) || (k_id_x < 0) || (k_id_y < 0) || (k_id_z >= Z) || (k_id_x >= X) ||
            (k_id_y >= Y))
        {
          cnt++;
          continue;
        }

        if (img_mask[q_id_b][k_id_z][k_id_x][k_id_y] != 0)
        {
          // Outputs as [Nout,3]
          // index_q[elem * ws_2 + cnt][0] = q_id_b;
          // index_q[elem * ws_2 + cnt][1] = q_id_x;
          // index_q[elem * ws_2 + cnt][2] = q_id_y;

          // index_k[elem * ws_2 + cnt][0] = q_id_b;
          // index_k[elem * ws_2 + cnt][1] = k_id_x;
          // index_k[elem * ws_2 + cnt][2] = k_id_y;

          index_q[elem * ws_prod + cnt] = elem;
          index_k[elem * ws_prod + cnt] = img_mask[q_id_b][k_id_z][k_id_x][k_id_y] - 1;
        }

        cnt++;
      }
    }
  }
  return;
};

void launch_find_indices_kernel(torch::Tensor index_q, torch::Tensor index_k,
                                const torch::Tensor& index_activ, const torch::Tensor& img_mask,
                                std::tuple<int, int, int> ws, bool only_last_z)
{
  const int64_t threads = 512;

  // index_activ: [Nactiv, 4]
  const int64_t nactiv = index_activ.size(0);
  const int64_t count = nactiv;

  if (count > 0)
  {
    AT_DISPATCH_INTEGRAL_TYPES(
        index_activ.scalar_type(), "find_indices_kernel",
        (
            [&]
            {
              find_indices_kernel<scalar_t><<<at::cuda::detail::GET_BLOCKS(count, threads), threads,
                                              0, at::cuda::getCurrentCUDAStream()>>>(
                  index_q.data_ptr<int64_t>(), index_k.data_ptr<int64_t>(),
                  index_activ.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                  img_mask.packed_accessor64<scalar_t, 4, torch::RestrictPtrTraits>(), ws,
                  only_last_z);
            }));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
};