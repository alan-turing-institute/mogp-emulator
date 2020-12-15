#include "gp_gpu.hpp"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "pybind11/eigen.h"

namespace py = pybind11;

PYBIND11_MODULE(libgpgpu, m) {
    py::class_<DenseGP_GPU>(m, "DenseGP_GPU")
        .def(py::init< mat_ref, vec_ref>())
        .def("data_length", &DenseGP_GPU::data_length)
        .def("predict", &DenseGP_GPU::predict)
        .def("predict_variance", &DenseGP_GPU::predict_variance)
        .def("predict_batch", &DenseGP_GPU::predict_batch)
        .def("predict_variance_batch", &DenseGP_GPU::predict_variance_batch)
        .def("update_theta", &DenseGP_GPU::update_theta)
        .def("get_invQ", &DenseGP_GPU::get_invQ)
        .def("get_invQt", &DenseGP_GPU::get_invQt)
        .def("get_logdetQ", &DenseGP_GPU::get_logdetQ)
        .def("dloglik_dtheta", &DenseGP_GPU::dloglik_dtheta);
}