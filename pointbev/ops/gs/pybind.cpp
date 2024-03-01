#include <pybind11/pybind11.h>

#include "gs.h"
#include "neighbourhood.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("forward_torch", &torch_gs_3d_fw_cuda);
  m.def("backward_torch", &torch_gs_3d_bw_cuda);
  m.def("forward_sparse", &sparsed_gs_3d_fw_cuda);
  m.def("backward_sparse", &sparsed_gs_3d_bw_cuda);
  m.def("forward_2d_sparse", &sparsed_gs_2d_fw_cuda);
  m.def("backward_2d_sparse", &sparsed_gs_2d_bw_cuda);
  m.def("find_indices", &find_indices_cuda);
}