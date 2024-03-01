#include <torch/extension.h>

std::tuple<torch::Tensor, torch::Tensor> find_indices_cuda(const torch::Tensor& index_activ,
                                                           const torch::Tensor& img_mask,
                                                           std::tuple<int, int, int> ws,
                                                           bool only_last_z);

void launch_find_indices_kernel(torch::Tensor index_q, torch::Tensor index_k,
                                const torch::Tensor& index_activ, const torch::Tensor& img_mask,
                                std::tuple<int, int, int> ws, bool only_last_z);