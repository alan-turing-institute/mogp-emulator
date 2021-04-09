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

}


PYBIND11_MODULE(libgpgpu, m) {
    py::class_<DenseGP_GPU>(m, "DenseGP_GPU")
        .def(py::init<mat_ref, vec_ref, unsigned int>())

////////////////////////////////////////
        .def("n", &DenseGP_GPU::get_n,
             "Returns the number of training examples used for the emulator")

////////////////////////////////////////
        .def("D", &DenseGP_GPU::get_D,
             "Returns the dimension of the emulator inputs")

////////////////////////////////////////
        .def("inputs", &DenseGP_GPU::get_inputs,
             "Returns the array of inputs points used by the emulator, with "
             "shape ``(n, D)``")

////////////////////////////////////////
        .def("targets", &DenseGP_GPU::get_targets,
             "Returns the array of targets used by the emulator, with shape "
             "``(n,)``")

////////////////////////////////////////
        .def("n_params", &DenseGP_GPU::get_n_params,
             "Returns the number of hyperparameters required for the emulator.")

////////////////////////////////////////
        .def("theta_fit_status", &DenseGP_GPU::get_theta_fit_status,
             "Has the emulator been fit?  If `false`, "
             ":func:``GaussianProcessGPU.theta`` will be ``None``")

////////////////////////////////////////
        .def("reset_theta_fit_status", &DenseGP_GPU::reset_theta_fit_status,
             "Marks this emulator as having not been fit: "
             ":func:`theta_fit_status` will be `false`")

////////////////////////////////////////
        .def("get_theta", &DenseGP_GPU::get_theta,
             "Returns the emulator hyperparameters")

////////////////////////////////////////
        .def("predict", &DenseGP_GPU::predict,
             R"(Single point predictive mean.

:param testing: The input point

:returns: The predicted mean

Currently unused by :class`GaussianProcessGPU`, but useful for testing)",
             py::arg("testing"))

////////////////////////////////////////
        .def("predict_variance", &DenseGP_GPU::predict_variance,
             R"(Single point combined predictive mean and variance.

:param testing: The input point to predict, with shape `(Nbatch, D)`
                (`Nbatch` is determined from the shape of the input).
                `Nbatch` must be less than `testing_size`.
:param var: (Output) The variance prediction is written to `var[0]`

:returns: The predicted mean

Currently unused by :class`GaussianProcessGPU`, but useful for testing.
)",
             py::arg("testing"),
             py::arg("var"))

////////////////////////////////////////
        .def("predict_batch", &DenseGP_GPU::predict_batch,
             R"(Batched predictive means.

:param testing: The input point to predict, with shape `(Nbatch, D)`
                (`Nbatch` is determined from the shape of the input).
                `Nbatch` must be less than `testing_size`.
:param result: (Output) The predicted mean at each `testing` point.
               Shape `(Nbatch, D)`.

:returns: None
)",
             py::arg("testing"),
             py::arg("result"))

////////////////////////////////////////
        .def("predict_variance_batch", &DenseGP_GPU::predict_variance_batch,
             R"(Batched predictive means and variances.

:param testing: The input point to predict, with shape `(Nbatch, D)`
                (`Nbatch` is determined from the shape of the input).
                `Nbatch` must be less than `testing_size`.
:param mean: (Output) The predicted mean at each `testing` point.
             Shape `(Nbatch, D)`.
:param var: (Output) The predicted variance at each `testing` point.
            Shape `(Nbatch, D)`.

:returns: None
)",
             py::arg("testing"),
             py::arg("mean"),
             py::arg("var"))

////////////////////////////////////////
        .def("predict_deriv", &DenseGP_GPU::predict_deriv,
             R"(Batched prediction of derivatives.

:param testing: The input point to predict, with shape `(Nbatch, D)`
                (`Nbatch` is determined from the shape of the input).
                `Nbatch` must be less than `testing_size`.
:param result: (Output) The predicted gradient of the mean with
               respect to the inputs.  Shape `(Nbatch, D)`.

:returns: None
)",
             py::arg("testing"),
             py::arg("result"))

////////////////////////////////////////

        .def("fit", &DenseGP_GPU::fit,
             R"(Sets theta and the nugget parameters, and updates the internal
state of the emulator accordingly.

:param theta: Values for the hyperparameters (length ``n_params``)
:param nugget: The interpretation of the nugget: adaptive, fixed or fit
:param nugget_size: The (initial) nugget value

:returns: None
)",
             py::arg("theta"),
             py::arg("nugget"),
             py::arg("nugget_size")
            )


////////////////////////////////////////
        .def("get_invQ", &DenseGP_GPU::get_invQ,
             R"(The inverse of the current covariance matrix

:param invQ_h: (Output) The matrix, of shape ``(n, n)``.

:returns: None
)",
             py::arg("invQ_h"))

////////////////////////////////////////
        .def("get_invQt", &DenseGP_GPU::get_invQt,
             R"(The inverse covariance matrix times the target vector.

:param invQt_h: (Output) The resulting vector, of shape ``(n,)``

:returns: None
)",
             py::arg("invQt_h"))

////////////////////////////////////////
        .def("get_logpost", &DenseGP_GPU::get_logpost,
	     "Return the log posterior"
            )

////////////////////////////////////////
        .def("get_jitter", &DenseGP_GPU::get_jitter,
             "The value of jitter used for an adaptive nugget.")

////////////////////////////////////////
        .def("get_cholesky_lower", &DenseGP_GPU::get_cholesky_lower,
             R"(Return the (lower) Cholesky factor of the covariance matrix.

This is the output of (cu)BLAS dpotrf.  The lower triangular entries correspond
to the factor; the upper triangular entries contain junk values.

:param result: (Output) the resulting Cholesky factor.  Shape ``(n,n)``.
               Lower triangular entries set.

:returns: None
 )",
             py::arg("result"))


////////////////////////////////////////
        .def("logpost_deriv", &DenseGP_GPU::logpost_deriv,
             R"(Calculate the partial derivatives of the negative log-posterior
likelihood of the hyperparameters, from the current state of the emulator.)

:param result: (Output) the resulting derivatives, shape ``(D+1,)``

:returns: `None`
)",
             py::arg("result"));



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
