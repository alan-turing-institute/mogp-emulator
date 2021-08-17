//#include "gp_gpu.hpp"
#include "fitting.hpp"
#include "multioutputgp_gpu.hpp"

#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "pybind11/eigen.h"

namespace py = pybind11;


PYBIND11_MODULE(libgpgpu, m) {
    py::class_<DenseGP_GPU>(m, "DenseGP_GPU")
      .def(py::init<mat_ref, vec_ref, unsigned int, BaseMeanFunc*, kernel_type>())

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
             R"(Sets theta, and updates the internal state of the emulator accordingly.

:param theta: Values for the hyperparameters (length ``n_params``)

:returns: None
)",
             py::arg("theta"))

////////////////////////////////////////
        .def("get_K", &DenseGP_GPU::get_K,
             R"(The current covariance matrix

:param K_h: (Output) The matrix, of shape ``(n, n)``.

:returns: None
)",
             py::arg("K_h"))

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
        .def("get_nugget_size", &DenseGP_GPU::get_nugget_size,
          "Get the value of the nugget.")

////////////////////////////////////////
        .def("set_nugget_size", &DenseGP_GPU::set_nugget_size,
          "Set the value of the nugget.")

////////////////////////////////////////
        .def("get_nugget_type", &DenseGP_GPU::get_nugget_type,
          "Get the type of the nugget.")

////////////////////////////////////////
        .def("set_nugget_type", &DenseGP_GPU::set_nugget_type,
          "Set the type of the nugget.")

////////////////////////////////////////
        .def("get_kernel_type", &DenseGP_GPU::get_kernel_type,
          "Get the type of the kernel.")

////////////////////////////////////////
          .def("get_kernel", &DenseGP_GPU::get_kernel,
          "Get the kernel object.")

////////////////////////////////////////
        .def("get_meanfunc", &DenseGP_GPU::get_meanfunc,
          "Get the mean function object.")

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

py::class_<DummyThing>(m, "DummyThing")
     .def(py::init<>()) ;            
////////////////////////////////////////
     py::class_<MultiOutputGP_GPU>(m, "MultiOutputGP_GPU")
     .def(py::init<mat_ref, std::vector<vec>&, unsigned int>())
       .def("predict_batch", &MultiOutputGP_GPU::predict_batch,
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
       .def("inputs", &MultiOutputGP_GPU::get_inputs,
                       "Return the inputs to the GP") 
       .def("emulator", &MultiOutputGP_GPU::get_emulator,
                       "Return the emulator at specified index",
                         py::arg("index"))  
       .def("targets", &MultiOutputGP_GPU::get_targets,
                         "Return the targets of the GP at specified index",
                         py::arg("index"))
       .def("fit_emulator", &MultiOutputGP_GPU::fit_emulator,
                         "Set hyperparameters of the GP at specified index",
                         py::arg("index"),
                         py::arg("theta"))  
       .def("fit", &MultiOutputGP_GPU::fit,
                         "Set hyperparameters of all emulators",
                         py::arg("theta"))  
       .def("predict", &MultiOutputGP_GPU::predict,
                         "Predict single value on all emulators",
                         py::arg("testing"))  
       .def("predict_batch", &MultiOutputGP_GPU::predict_batch,
                         "Predict multiple values on all emulators",
                         py::arg("testing"),
                         py::arg("results"))  
       .def("predict_variance_batch", &MultiOutputGP_GPU::predict_variance_batch,
                         "Predict multiple values on all emulators with variances",
                         py::arg("testing"),
                         py::arg("means"),
                         py::arg("vars"))  
       .def("predict_deriv", &MultiOutputGP_GPU::predict_deriv,
                         "Derivatives of predictions on all emulators",
                         py::arg("testing"),
                         py::arg("results"))  
       .def("n_emulators", &MultiOutputGP_GPU::n_emulators,
               "Return the number of emulators being used");                 

////////////////////////////////////////
    py::class_<SquaredExponentialKernel>(m, "SquaredExponentialKernel")
      .def(py::init<>())
        .def("kernel_f", &SquaredExponentialKernel::kernel_f,
             "Calculate the covariance matrix")
        .def("kernel_deriv", &SquaredExponentialKernel::kernel_deriv,
             "Calculate the derivative of the covariance matrix wrt hyperparameters")
        .def("kernel_inputderiv", &SquaredExponentialKernel::kernel_inputderiv,
	     "Derivative of covariance matrix wrt inputs");

////////////////////////////////////////
    py::class_<Matern52Kernel>(m, "Matern52Kernel")
      .def(py::init<>())
        .def("kernel_f", &Matern52Kernel::kernel_f,
             "Calculate the covariance matrix")
        .def("kernel_deriv", &Matern52Kernel::kernel_deriv,
             "Calculate the derivative of the covariance matrix wrt hyperparameters")
        .def("kernel_inputderiv", &Matern52Kernel::kernel_inputderiv,
	     "Derivative of covariance matrix wrt inputs");

////////////////////////////////////////
    py::class_<BaseMeanFunc>(m, "BaseMeanFunc");

////////////////////////////////////////
    py::class_<ConstMeanFunc, BaseMeanFunc>(m, "ConstMeanFunc")
      .def(py::init<>())
        .def("mean_f", &ConstMeanFunc::mean_f,
             "Evaluate the mean function at input points")
        .def("mean_deriv", &ConstMeanFunc::mean_deriv,
             "Calculate the derivative of the mean function wrt hyperparameters")
        .def("mean_inputderiv", &ConstMeanFunc::mean_inputderiv,
	     "Derivative of mean function wrt inputs")
        .def("get_n_params", &ConstMeanFunc::get_n_params,
	     "Number of parameters of mean function");

////////////////////////////////////////
    py::class_<PolyMeanFunc, BaseMeanFunc>(m, "PolyMeanFunc")
      .def(py::init<std::vector< std::pair<int, int> > >())
        .def("mean_f", &PolyMeanFunc::mean_f,
             "Evaluate the mean function at input points")
        .def("mean_deriv", &PolyMeanFunc::mean_deriv,
             "Calculate the derivative of the mean function wrt hyperparameters")
        .def("mean_inputderiv", &PolyMeanFunc::mean_inputderiv,
	     "Derivative of mean function wrt inputs")
        .def("get_n_params", &PolyMeanFunc::get_n_params,
	     "Number of parameters of mean function");

////////////////////////////////////////
    py::class_<FixedMeanFunc, BaseMeanFunc>(m, "FixedMeanFunc")
      .def(py::init<REAL>())
        .def("mean_f", &FixedMeanFunc::mean_f,
             "Evaluate the mean function at input points")
        .def("mean_deriv", &FixedMeanFunc::mean_deriv,
             "Calculate the derivative of the mean function wrt hyperparameters")
        .def("mean_inputderiv", &FixedMeanFunc::mean_inputderiv,
	     "Derivative of mean function wrt inputs")
        .def("get_n_params", &FixedMeanFunc::get_n_params,
	     "Number of parameters of mean function");

////////////////////////////////////////
    py::class_<ZeroMeanFunc, BaseMeanFunc>(m, "ZeroMeanFunc")
      .def(py::init<>())
        .def("mean_f", &ZeroMeanFunc::mean_f,
             "Evaluate the mean function at input points")
        .def("mean_deriv", &ZeroMeanFunc::mean_deriv,
             "Calculate the derivative of the mean function wrt hyperparameters")
        .def("mean_inputderiv", &ZeroMeanFunc::mean_inputderiv,
	     "Derivative of mean function wrt inputs")
        .def("get_n_params", &ZeroMeanFunc::get_n_params,
	     "Number of parameters of mean function");

    py::enum_<kernel_type>(m, "kernel_type")
        .value("SquaredExponential", SQUARED_EXPONENTIAL)
        .value("Matern52", MATERN52);

    py::enum_<nugget_type>(m, "nugget_type")
        .value("adaptive", NUG_ADAPTIVE)
        .value("fixed", NUG_FIXED)
        .value("fit", NUG_FIT);

    m.def("have_compatible_device", &have_compatible_device);

    m.def("fit_GP_MAP", &fit_GP_MAP, py::return_value_policy::reference);

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
