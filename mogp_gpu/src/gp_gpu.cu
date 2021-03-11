#include "gp_gpu.hpp"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "pybind11/eigen.h"

namespace py = pybind11;

bool have_compatible_device(void)
{
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);

    return (err == cudaSuccess && device_count > 0);

    // TODO
    // Check whether device_prop.major, device_prop.minor meet our requirements

    // cudaDeviceProp device_prop;

    // Use the default device (device 0) only, for now
    // cudaGetDeviceProp(&device_prop, 0);
}


PYBIND11_MODULE(libgpgpu, m) {
    py::class_<DenseGP_GPU>(m, "DenseGP_GPU")
        .def(py::init<mat_ref, vec_ref, unsigned int>())
        .def("data_length", &DenseGP_GPU::data_length)
        .def("D", &DenseGP_GPU::D)
        .def("inputs", &DenseGP_GPU::inputs)
        .def("targets", &DenseGP_GPU::targets)
        .def("n_params", &DenseGP_GPU::n_params)
        .def("theta_fit_status", &DenseGP_GPU::theta_fit_status)
        .def("theta_reset_fit_status", &DenseGP_GPU::theta_reset_fit_status)
        .def("get_theta", &DenseGP_GPU::get_theta)
        .def("predict", &DenseGP_GPU::predict)
        .def("predict_variance", &DenseGP_GPU::predict_variance)
        .def("predict_batch", &DenseGP_GPU::predict_batch)
        .def("predict_variance_batch", &DenseGP_GPU::predict_variance_batch)
        .def("predict_deriv", &DenseGP_GPU::predict_deriv)
        .def("update_theta", &DenseGP_GPU::update_theta)
        .def("get_invQ", &DenseGP_GPU::get_invQ)
        .def("get_invQt", &DenseGP_GPU::get_invQt)
        .def("get_logpost", &DenseGP_GPU::get_logpost)
        .def("get_jitter", &DenseGP_GPU::get_jitter)
        .def("get_Cholesky_lower", &DenseGP_GPU::get_Cholesky_lower)
        .def("dloglik_dtheta", &DenseGP_GPU::dloglik_dtheta);

    py::enum_<nugget_type>(m, "nugget_type")
        .value("adaptive", NUG_ADAPTIVE)
        .value("fixed", NUG_FIXED)
        .value("fit", NUG_FIT);

    m.def("have_compatible_device", &have_compatible_device);
}
