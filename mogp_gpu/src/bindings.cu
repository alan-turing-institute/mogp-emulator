#include "fitting.hpp"
#include "gpparams.hpp"
#include "gppriors.hpp"
#include "multioutputgp_gpu.hpp"

#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "pybind11/eigen.h"

namespace py = pybind11;


PYBIND11_MODULE(libgpgpu, m) {
    py::class_<DenseGP_GPU, std::unique_ptr<DenseGP_GPU, py::nodelete>>(m, "DenseGP_GPU")
      .def(py::init<mat_ref, vec_ref, unsigned int, mat_ref, kernel_type, nugget_type, double>())
//      py::return_value_policy::reference)

////////////////////////////////////////
        .def("n", &DenseGP_GPU::get_n,
             "Returns the number of training examples used for the emulator")

////////////////////////////////////////
        .def("D", &DenseGP_GPU::get_D,
             "Returns the dimension of the emulator inputs")

////////////////////////////////////////
        .def("n_corr", &DenseGP_GPU::get_n_corr,
             "Returns the number of correlation length parameters")

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
        .def("get_gppriors", &DenseGP_GPU::get_gppriors,
             py::return_value_policy::reference,
             "Returns the GPPriors object")
////////////////////////////////////////
        .def("set_gppriors", &DenseGP_GPU::set_gppriors,
             "Set the GPPriors object")

///////////////////////////////////////
        .def("create_gppriors", &DenseGP_GPU::create_gppriors,  
             "Instantiate and configure the GPPriors object")

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

        .def("fit", py::overload_cast<GPParams&>(&DenseGP_GPU::fit),
             R"(Sets theta, and updates the internal state of the emulator accordingly.

:param theta: Values for the hyperparameters (length ``n_data``)

:returns: None
)",
             py::arg("theta"))
////////////////////////////////////////

.def("fit", py::overload_cast<vec>(&DenseGP_GPU::fit),
             R"(Sets theta, and updates the internal state of the emulator accordingly.

:param theta: Values for the hyperparameters (length ``n_data``)

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
        .def("get_logpost", py::overload_cast<vec>(&DenseGP_GPU::get_logpost),
	     "Return the log posterior"
            )
////////////////////////////////////////
        .def("get_logpost", py::overload_cast<GPParams&>(&DenseGP_GPU::get_logpost),
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
        .def("get_design_matrix", &DenseGP_GPU::get_design_matrix,
          "Get the design_matrix.")

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
           
////////////////////////////////////////
     py::class_<MultiOutputGP_GPU>(m, "MultiOutputGP_GPU")
     .def(py::init<mat_ref, std::vector<vec>&, unsigned int, BaseMeanFunc*, kernel_type, nugget_type, double>())
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
       .def("n", &MultiOutputGP_GPU::get_n,
                     "Return the number of inputs to the GP")
       .def("D", &MultiOutputGP_GPU::get_D,
                     "Return the number of dimensions of the GP")
       .def("n_data_params", &MultiOutputGP_GPU::n_data_params,
                     "Return the number of data parameters of the GP")
       .def("n_corr_params", &MultiOutputGP_GPU::n_corr_params,
                     "Return the number of correlation length parameters of the GP")
       .def("get_nugget_type", &MultiOutputGP_GPU::get_nugget_type,
                     "Return the nugget type")
       .def("get_nugget_size", &MultiOutputGP_GPU::get_nugget_size,
                     "Return the nugget size")
       .def("get_fitted_indices", &MultiOutputGP_GPU::get_fitted_indices,
                     "Return a vector of indices of emulators where fitting succeeded")
       .def("get_unfitted_indices", &MultiOutputGP_GPU::get_unfitted_indices,
                     "Return a vector of indices of emulators where fitting failed or not done yet")
       .def("n_emulators", &MultiOutputGP_GPU::n_emulators,
                     "Return the number of GP emulators")
       .def("create_priors_for_emulator", &MultiOutputGP_GPU::create_priors_for_emulator,
                     "Instantiate GPPriors object for specified emulator index")
       .def("reset_fit_status", &MultiOutputGP_GPU::reset_fit_status,
                     "reset the fit status of all emulators")
       .def("targets", &MultiOutputGP_GPU::get_targets,
                         "Return the targets of the GP") 
       .def("targets_at_index", &MultiOutputGP_GPU::get_targets_at_index,
                         "Return the targets of the GP at specified index",
                         py::arg("index"))
       .def("fit_emulator", py::overload_cast<unsigned int, GPParams&>(&MultiOutputGP_GPU::fit_emulator),
                         "Set hyperparameters of the GP at specified index",
                         py::arg("index"),
                         py::arg("theta"))  
       .def("fit_emulator", py::overload_cast<unsigned int, vec&>(&MultiOutputGP_GPU::fit_emulator),
                         "Set hyperparameters of the GP at specified index",
                         py::arg("index"),
                         py::arg("theta"))  
       .def("fit", py::overload_cast<mat_ref>(&MultiOutputGP_GPU::fit),
                         "Set hyperparameters of all emulators",
                         py::arg("thetas"))  
       .def("fit", py::overload_cast<std::vector<GPParams>>(&MultiOutputGP_GPU::fit),
                         "Set hyperparameters of all emulators",
                         py::arg("thetas"))  
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
	     "Derivative of covariance matrix wrt inputs")
        .def("get_n_params", &SquaredExponentialKernel::get_n_params,
	     "Number of correlation length parameters");

////////////////////////////////////////
    py::class_<Matern52Kernel>(m, "Matern52Kernel")
      .def(py::init<>())
        .def("kernel_f", &Matern52Kernel::kernel_f,
             "Calculate the covariance matrix")
        .def("kernel_deriv", &Matern52Kernel::kernel_deriv,
             "Calculate the derivative of the covariance matrix wrt hyperparameters")
        .def("kernel_inputderiv", &Matern52Kernel::kernel_inputderiv,
	     "Derivative of covariance matrix wrt inputs")
        .def("get_n_params", &SquaredExponentialKernel::get_n_params,
	     "Number of correlation length parameters");
;

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

     py::class_<GPParams>(m, "GPParameters")
      .def(py::init<int, int, nugget_type>())
        .def("get_data", &GPParams::get_data,
             "get the raw data")
        .def("set_data", &GPParams::set_data,
             "set the raw data")
        .def("get_n_data", &GPParams::get_n_data,
             "get size of data array")
        .def("get_mean", &GPParams::get_mean,
	     "get the mean function parameters")
        .def("set_mean", &GPParams::set_mean,
	     "set the mean function parameters")
        .def("get_n_mean", &GPParams::get_n_mean,
	     "get the number of mean function parameters")
        .def("get_nugget_type", &GPParams::get_nugget_type,
	     "get the nugget_type")
        .def("set_nugget_type", &GPParams::set_nugget_type,
	     "set nugget_type")
        .def("get_nugget_size", &GPParams::get_nugget_size,
	     "get the nugget_size")
        .def("set_nugget_size", &GPParams::set_nugget_size,
	     "set nugget_size")
        .def("get_corr", &GPParams::get_corr,
	     "get the transformed correlation parameters")
        .def("set_corr", &GPParams::set_corr,
	     "set correlation parameters using transformed values")
        .def("get_n_corr", &GPParams::get_n_corr,
	     "get the number of correlation parameters")
        .def("get_cov", &GPParams::get_cov,
	     "get the transformed covariance parameter")
        .def("set_cov", &GPParams::set_cov,
	     "set covariance parameter using transformed values")
        .def("get_corr_raw", &GPParams::get_corr_raw,
	     "get the raw correlation parameters")
        .def("unset_data", &GPParams::unset_data,
	     "unset the flag to say params have been set")
        .def("data_has_been_set", &GPParams::data_has_been_set,
	     "get the flag to say whether or not params have been set")
        .def("test_same_shape", py::overload_cast<GPParams&>(&GPParams::test_same_shape, py::const_),
	     "test whether two GPParams objects are the same shape")
        .def("test_same_shape", py::overload_cast<vec&>(&GPParams::test_same_shape, py::const_),
	     "test whether two vectors of params are the same shape");
     ////////////////////////////////////////
     py::class_<WeakPrior>(m, "WeakPrior")
      .def(py::init<>())
          .def("sample", py::overload_cast<>(&WeakPrior::sample),
          "sample from the prior")
          .def("sample", py::overload_cast<const BaseTransform&>(&WeakPrior::sample),
          "sample from the prior");
     ////////////////////////////////////////
     py::class_<InvGammaPrior, WeakPrior>(m, "InvGammaPrior")
      .def(py::init<double, double>())
        .def("logp", &InvGammaPrior::logp,
             "Calculate the log probability")
        .def("dlogpdx", &InvGammaPrior::dlogpdx,
             "Calculate the derivative of the log probability")
        .def("d2logpdx2", &InvGammaPrior::d2logpdx2,
	     "Second Derivative of log prob wrt input")
        .def("sample_x", &InvGammaPrior::sample_x,
	     "Sample from the distribution");
     //////////////////////////////////////
     py::class_<GammaPrior, WeakPrior>(m, "GammaPrior")
          .def(py::init<double, double>())
            .def("logp", &GammaPrior::logp,
                 "Calculate the log probability")
            .def("dlogpdx", &GammaPrior::dlogpdx,
                 "Calculate the derivative of the log probability")
            .def("d2logpdx2", &GammaPrior::d2logpdx2,
              "Second Derivative of log prob wrt input")
            .def("sample_x", &GammaPrior::sample_x,
              "Sample from the distribution");
     //////////////////////////////////////
     py::class_<BaseTransform>(m, "BaseTransform");
     ////////////////////////////////////////
     py::class_<CovTransform, BaseTransform>(m, "CovTransform")
          .def(py::init<>()) 
          .def("raw_to_scaled", py::overload_cast<REAL>(&CovTransform::raw_to_scaled, py::const_),
          "transform from raw values to scaled ones")
          .def("raw_to_scaled", py::overload_cast<vec>(&CovTransform::raw_to_scaled, py::const_),
          "transform from raw values to scaled ones")
          .def("scaled_to_raw", py::overload_cast<REAL>(&CovTransform::scaled_to_raw, py::const_),
          "transform from scaled values to raw ones")
          .def("scaled_to_raw", py::overload_cast<vec>(&CovTransform::scaled_to_raw, py::const_),
          "transform from scaled values to raw ones")
          .def("dscaled_draw", py::overload_cast<REAL>(&CovTransform::dscaled_draw, py::const_),
          "derivative of scaled values wrt raw ones")
          .def("dscaled_draw", py::overload_cast<vec>(&CovTransform::dscaled_draw, py::const_),
          "derivative of scaled values wrt raw ones")
          .def("2scaled_draw2", py::overload_cast<REAL>(&CovTransform::d2scaled_draw2, py::const_),
          "second derivative of scaled values wrt raw ones")
          .def("d2scaled_draw2", py::overload_cast<vec>(&CovTransform::d2scaled_draw2, py::const_),
          "second derivative of scaled values wrt raw ones");
        
     ////////////////////////////////////////
     py::class_<CorrTransform, BaseTransform>(m, "CorrTransform")
          .def(py::init<>()) 
          .def("raw_to_scaled", py::overload_cast<REAL>(&CorrTransform::raw_to_scaled, py::const_),
          "transform from raw values to scaled ones")
          .def("raw_to_scaled", py::overload_cast<vec>(&CorrTransform::raw_to_scaled, py::const_),
          "transform from raw values to scaled ones")
          .def("scaled_to_raw", py::overload_cast<REAL>(&CorrTransform::scaled_to_raw, py::const_),
          "transform from scaled values to raw ones")
          .def("scaled_to_raw", py::overload_cast<vec>(&CorrTransform::scaled_to_raw, py::const_),
          "transform from scaled values to raw ones")
          .def("dscaled_draw", py::overload_cast<REAL>(&CorrTransform::dscaled_draw, py::const_),
          "derivative of scaled values wrt raw ones")
          .def("dscaled_draw", py::overload_cast<vec>(&CorrTransform::dscaled_draw, py::const_),
          "derivative of scaled values wrt raw ones")
          .def("2scaled_draw2", py::overload_cast<REAL>(&CorrTransform::d2scaled_draw2, py::const_),
          "second derivative of scaled values wrt raw ones")
          .def("d2scaled_draw2", py::overload_cast<vec>(&CorrTransform::d2scaled_draw2, py::const_),
          "second derivative of scaled values wrt raw ones");
     ////////////////////////////////////////
     py::class_<GPPriors>(m, "GPPriors")
          .def(py::init<int, nugget_type>())
          .def("set_corr", py::overload_cast<std::vector<WeakPrior*>>(&GPPriors::set_corr),
               "set correlation length priors")
          .def("set_corr", py::overload_cast<>(&GPPriors::set_corr),
               "set correlation length priors")
          .def("get_corr", &GPPriors::get_corr,
               "get correlation length priors")
          .def("set_cov", py::overload_cast<>(&GPPriors::set_cov),
               "set covariance priors")
          .def("set_cov", py::overload_cast<WeakPrior*>(&GPPriors::set_cov),
               "set covariance priors")
          .def("get_cov", &GPPriors::get_cov,
               "get covariance priors")
          .def("get_logp", &GPPriors::logp,
               "get log posterior ")
          .def("get_dlogpdtheta", &GPPriors::dlogpdtheta,
               "get derivative of log posterior wrt theta")
          .def("get_d2logpdtheta2", &GPPriors::d2logpdtheta2,
               "get 2nd derivative of log posterior wrt theta")
          .def("create_corr_priors", &GPPriors::create_corr_priors,
               "create default correlation length priors")
          .def("create_cov_prior", &GPPriors::create_cov_prior,
               "create default covariance parameter prior")
          .def("make_prior", &GPPriors::make_prior,
               "instantiate prior of chosen type")
          .def("sample", &GPPriors::sample,
               "sample from the priors");

     ////////////////////////////////////////
     py::class_<MeanPriors>(m, "MeanPriors")
          .def(py::init<vec, mat>())
          .def("get_mean", &MeanPriors::get_mean,
               "return the meanfunction vector")
          .def("get_cov", &MeanPriors::get_cov,
               "return the covariance matrix")
          .def("get_n_params", &MeanPriors::get_n_params,
               "how many parameters?")
          .def("has_weak_priors", &MeanPriors::has_weak_priors,
               "do we only have non-informative priors?")
          .def("dm_dot_b", &MeanPriors::dm_dot_b,
               "dot product of design matrix with mean")
          .def("inv_cov", &MeanPriors::get_inv_cov,
               "inverse covariance matrix")
          .def("inv_cov_b", &MeanPriors::get_inv_cov_b,
               "dot product of inverse covariance matrix with mean")
          .def("logdet_cov", &MeanPriors::logdet_cov,
               "log of the determinant of the covariance");
               
     ////////////////  enums   /////////////////////
     py::enum_<kernel_type>(m, "kernel_type")
        .value("SquaredExponential", SQUARED_EXPONENTIAL)
        .value("Matern52", MATERN52);

     py::enum_<nugget_type>(m, "nugget_type")
        .value("adaptive", NUG_ADAPTIVE)
        .value("fixed", NUG_FIXED)
        .value("fit", NUG_FIT);

     py::enum_<prior_type>(m, "prior_type")
        .value("LogNormal", LOGNORMAL)
        .value("Gamma", GAMMA)
        .value("InvGamma", INVGAMMA)
        .value("Weak", WEAK);

     m.def("have_compatible_device", &have_compatible_device);

     m.def("fit_GP_MAP", py::overload_cast<DenseGP_GPU&, const int, const std::vector<double>>(&fit_single_GP_MAP), 
          py::return_value_policy::reference);
     m.def("fit_GP_MAP", py::overload_cast<MultiOutputGP_GPU&, const int, const std::vector<double>>(&fit_GP_MAP), 
          py::return_value_policy::reference);

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
