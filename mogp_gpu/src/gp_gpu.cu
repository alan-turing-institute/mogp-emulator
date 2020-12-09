#include "gp_gpu.hpp"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "pybind11/eigen.h"

namespace py = pybind11;

PYBIND11_MODULE(gpgpu_example, m) {
    py::class_<DenseGP_GPU>(m, "DenseGP_GPU")
        .def(py::init<unsigned int, unsigned int, mat_ref, vec_ref>())
        .def("data_length", &DenseGP_GPU::data_length)
        .def("predict", &DenseGP_GPU::predict_batch)
        .def("update_theta", &DenseGP_GPU::update_theta);
}
