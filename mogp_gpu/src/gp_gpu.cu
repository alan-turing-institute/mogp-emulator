#include "gp_gpu.hpp"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

namespace py = pybind11;

PYBIND11_MODULE(gpgpu_example, m) {
    py::class_<DenseGP_GPU>(m, "DenseGP_GPU")
      .def(py::init<unsigned int, unsigned int, const std::vector<double>&, const std::vector<double>&, const std::vector<double>&>())
      .def("data_length", &DenseGP_GPU::data_length);
}
