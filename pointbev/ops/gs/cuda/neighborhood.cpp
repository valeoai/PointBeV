#include <cmath>
#include <iostream>
#include <neighbourhood.h>
#include <torch/extension.h>

using namespace torch::indexing;

std::tuple<torch::Tensor, torch::Tensor> find_indices_cuda(const torch::Tensor& index_activ,
                                                           const torch::Tensor& img_mask,
                                                           std::tuple<int, int, int> ws,
                                                           bool only_last_z)
{
  TORCH_CHECK(index_activ.is_cuda(), " must be a CUDA tensor");
  TORCH_CHECK(index_activ.is_contiguous(), " must be contiguous");
  AT_ASSERTM(index_activ.dim() == 2, "index_activ must be a 2D tensor: (Nactiv,4)");
  AT_ASSERTM(index_activ.size(1) == 4, "index_activ must be a 2D tensor: (Nactiv,4)");

  TORCH_CHECK(img_mask.is_cuda(), " must be a CUDA tensor");
  TORCH_CHECK(img_mask.is_contiguous(), " must be contiguous");
  AT_ASSERTM(img_mask.dim() == 4, "index_activ must be a 4D tensor: (b,Z,X,Y)");

  // index_activ: (Nactiv,4)
  const int nactiv = index_activ.size(0);

  // Maximum comparisons:
  int ws_z = std::get<0>(ws);
  int ws_x = std::get<1>(ws);
  int ws_y = std::get<2>(ws);
  const int ws_prod = ws_z * ws_x * ws_y;

  // Outputs
  torch::Tensor index_q =
      torch::full({nactiv * ws_prod}, -1, index_activ.options().dtype(torch::kInt64));
  torch::Tensor index_k =
      torch::full({nactiv * ws_prod}, -1, index_activ.options().dtype(torch::kInt64));

  // Change mask to a matrix containing order at the location of the activated points.
  const int Z = img_mask.size(1);
  const int X = img_mask.size(2);
  const int Y = img_mask.size(3);
  torch::Tensor nonZeroIndices =
      index_activ.index({Slice(), 0}) * Z * X * Y + index_activ.index({Slice(), 1}) * X * Y +
      index_activ.index({Slice(), 2}) * Y + index_activ.index({Slice(), 3});

  // Activated indices now contained their order in the list of activated points.
  torch::Tensor arangeTensor = torch::arange(1, nactiv + 1, img_mask.options());
  img_mask.flatten().index_put_({nonZeroIndices}, arangeTensor);

  // Launch kernel
  launch_find_indices_kernel(index_q, index_k, index_activ, img_mask, ws, only_last_z);

  // Reinitialize mask to a matrix containing 1 at the location of the activated points.
  torch::Tensor oneTensor = torch::ones(nactiv, img_mask.options());
  img_mask.flatten().index_put_({nonZeroIndices}, oneTensor);

  auto idx_keep = index_q != -1;
  return std::make_tuple(index_q.masked_select(idx_keep), index_k.masked_select(idx_keep));
};