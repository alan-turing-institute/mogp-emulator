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
        .def("n", &DenseGP_GPU::get_n,
             "Returns the number of training examples used for the emulator")

        .def("D", &DenseGP_GPU::get_D,
             "Returns the dimension of the emulator inputs")

        .def("inputs", &DenseGP_GPU::get_inputs,
             "Returns the array of inputs points used by the emulator, with shape ``(n, D)``")

        .def("targets", &DenseGP_GPU::get_targets,
             "Returns the array of targets used by the emulator, with shape ``(n,)``")

        .def("n_params", &DenseGP_GPU::get_n_params,
             "Returns the number of hyperparameters required for the emulator.")

        .def("theta_fit_status", &DenseGP_GPU::get_theta_fit_status,
             "Has the emulator been fit?  If `false`, :func:``GaussianProcessGPU.theta`` will be ``None``")

        .def("reset_theta_fit_status", &DenseGP_GPU::reset_theta_fit_status,
             "Marks this emulator as having not been fit: :func:`theta_fit_status` will be `false`")

        .def("get_theta", &DenseGP_GPU::get_theta,
             "Returns the emulator hyperparameters")

        .def("predict", &DenseGP_GPU::predict,
             R"(Single point predictive mean.

:param arg0: The input point

:returns: The predicted mean

Currently unused by :class`GaussianProcessGPU`, but useful for testing)")

        .def("predict_variance", &DenseGP_GPU::predict_variance,
             R"(Single point combined predictive mean and variance.

:param arg0: The input point, to predict
:param arg1: The variance prediction is written to `arg1[0]`

:returns: The predicted mean

Currently unused by :class`GaussianProcessGPU`, but useful for testing.
)")

        .def("predict_batch", &DenseGP_GPU::predict_batch,
             "Batched predictive means")

        .def("predict_variance_batch", &DenseGP_GPU::predict_variance_batch,
             "Batched predictive variances")

        .def("predict_deriv_batch", &DenseGP_GPU::predict_deriv_batch,
	   "Batched predictive means and derivatives")

        .def("fit", &DenseGP_GPU::fit)

        .def("get_invQ", &DenseGP_GPU::get_invQ)

        .def("get_invQt", &DenseGP_GPU::get_invQt)

        .def("get_logpost", &DenseGP_GPU::get_logpost)

        .def("get_jitter", &DenseGP_GPU::get_jitter)

        .def("get_cholesky_lower", &DenseGP_GPU::get_cholesky_lower)

        .def("logpost_deriv", &DenseGP_GPU::logpost_deriv);

    py::enum_<nugget_type>(m, "nugget_type")
        .value("adaptive", NUG_ADAPTIVE)
        .value("fixed", NUG_FIXED)
        .value("fit", NUG_FIT);

    m.def("have_compatible_device", &have_compatible_device);

    m.doc() = R"(
The libgpgpu library
--------------------

.. currentmodule:: libgpgpu

The members of this class are

.. autoclass:: libgpgpu.DenseGP_GPU
    :members:

)";

    // Include have_compatible_device in the docstring
}
