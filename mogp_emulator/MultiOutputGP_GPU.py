import platform
import numpy as np
from mogp_emulator.GaussianProcess import (
    GaussianProcessBase,
    GaussianProcess,
    PredictResult
)
from mogp_emulator.GaussianProcessGPU import GaussianProcessGPU, parse_meanfunc_formula
from mogp_emulator.MeanFunction import MeanBase
from mogp_emulator.Kernel import Kernel, SquaredExponential, Matern52
from mogp_emulator.Priors import Prior
from mogp_emulator import LibGPGPU


class MultiOutputGP_GPU(object):
    """Implementation of a multiple-output Gaussian Process Emulator, using the GPU.

    Essentially a parallelized wrapper for the predict method. To fit
    in parallel, use the ``fit_GP_MAP`` routine

    Required arguments are ``inputs`` and ``targets``, both of which
    must be numpy arrays. ``inputs`` can be 1D or 2D (if 1D, assumes
    second axis has length 1). ``targets`` can be 1D or 2D (if 2D,
    assumes a single emulator and the first axis has length 1).

    Optional arguments specify how each individual emulator is
    constructed, including the mean function, kernel, priors, and how
    to handle the nugget. Each argument can take values allowed by the
    base ``GaussianProcess`` class, in which case all emulators are
    assumed to use the same value. Any of these arguments can
    alternatively be a list of values with length matching the number
    of emulators to set those values individually.

    Additional keyword arguments include ``inputdict``, and
    ``use_patsy``, which control how strings are parsed to mean
    functions, if using.

    """

    def __init__(self, inputs, targets, mean=None, kernel=SquaredExponential(), priors=None,
                 nugget="adaptive", inputdict={}, use_patsy=True, batch_size=8000):
        """
        Create a new multi-output GP Emulator
        """

        # check input types and shapes, reshape as appropriate for the case of a single emulator
        inputs = np.array(inputs)
        targets = np.array(targets)
        if len(inputs.shape) == 1:
            inputs = np.reshape(inputs, (-1, 1))
        if len(targets.shape) == 1:
            targets = np.reshape(targets, (1, -1))
        elif not (len(targets.shape) == 2):
            raise ValueError("targets must be either a 1D or 2D array")
        if not (len(inputs.shape) == 2):
            raise ValueError("inputs must be either a 1D or 2D array")
        if not (inputs.shape[0] == targets.shape[1]):
            raise ValueError("the first dimension of inputs must be the same length as the second dimension of targets (or first if targets is 1D))")

        # nugget type
        nugsize = 0.
        if nugget == "adaptive": 
            nugtype = LibGPGPU.nugget_type(0)
        elif nugget == "fit":
            nugtype = LibGPGPU.nugget_type(1)
        elif isinstance(nugget, float):
            nugtype = LibGPGPU.nugget_type(2)
            nugsize = nugget
        else:
            raise TypeError("nugget parameter must be a string or a non-negative float")

        # set the Mean Function
        if mean is None:
            meanfunc = LibGPGPU.ZeroMeanFunc()
        else:
            if not issubclass(type(mean), MeanBase):
                if isinstance(mean, str):
                    mean = MeanFunction(mean, inputdict, use_patsy)
                else:
                    raise ValueError("provided mean function must be a subclass of MeanBase,"+
                                     " a string formula, or None")
            # at this point, mean will definitely be a MeanBase.  We can call its __str__ and
            # parse this to create an instance of a C++ MeanFunction
            meanfunc = parse_meanfunc_formula(mean.__str__())
            # if we got None back from that function, something went wrong
            if not meanfunc:
                raise ValueError("""
                GPU implementation was unable to parse mean function formula {}.
                """.format(mean.__str__())
                )

        # set the kernel type
        if (isinstance(kernel, str) and kernel == "SquaredExponential") \
           or isinstance(kernel, SquaredExponential):
            kernel_type = LibGPGPU.kernel_type.SquaredExponential
        elif (isinstance(kernel, str) and kernel == "Matern52") \
           or isinstance(kernel, Matern52):
            kernel_type = LibGPGPU.kernel_type.Matern52
        else:
            raise ValueError("GPU implementation requires kernel to be SquaredExponential or Matern52")

        self._mogp_gpu = LibGPGPU.MultiOutputGP_GPU(inputs, targets, batch_size, meanfunc, kernel_type, nugtype, nugsize)

    @property
    def inputs(self):
        return self._mogp_gpu.inputs()

    @property
    def targets(self):
        return np.array(self._mogp_gpu.targets())

    @property
    def D(self):
        return self._mogp_gpu.D()

    @property
    def n(self):
        return self._mogp_gpu.n()

    @property
    def n_emulators(self):
        return self._mogp_gpu.n_emulators()

    @property
    def emulators(self):
        emuls = []
        for i in range(self.n_emulators):
            emuls.append(GaussianProcessGPU.from_cpp(self._mogp_gpu.emulator(i)))
        return emuls

    def get_emulator(self, index):
        emulator = self._mogp_gpu.emulator(index)
        return GaussianProcessGPU.from_cpp(emulator)

    def predict(self, testing, unc=True, deriv=True, include_nugget=True,
                allow_not_fit=False, processes=None):
        """Make a prediction for a set of input vectors

        Makes predictions for each of the emulators on a given set of
        input vectors. The input vectors must be passed as a
        ``(n_predict, D)`` or ``(D,)`` shaped array-like object, where
        ``n_predict`` is the number of different prediction points
        under consideration and ``D`` is the number of inputs to the
        emulator. If the prediction inputs array has shape ``(D,)``,
        then the method assumes ``n_predict == 1``.  The prediction
        points are passed to each emulator and the predictions are
        collected into an ``(n_emulators, n_predict)`` shaped numpy
        array as the first return value from the method.

        Optionally, the emulator can also calculate the uncertainties
        in the predictions (as a variance) and the derivatives with
        respect to each input parameter. If the uncertainties are
        computed, they are returned as the second output from the
        method as an ``(n_emulators, n_predict)`` shaped numpy
        array. If the derivatives are computed, they are returned as
        the third output from the method as an ``(n_emulators,
        n_predict, D)`` shaped numpy array. Finally, if uncertainties
        are computed, the ``include_nugget`` flag determines if the
        uncertainties should include the nugget. By default, this is
        set to ``True``.

        The ``allow_not_fit`` flag determines how the object handles
        any emulators that do not have fit hyperparameter values
        (because fitting presumably failed). By default,
        ``allow_not_fit=False`` and the method will raise an error
        if any emulators are not fit. Passing ``allow_not_fit=True``
        will override this and ``NaN`` will be returned from any
        emulators that have not been fit.

        As with the fitting, this computation can be done
        independently for each emulator and thus can be done in
        parallel.

        :param testing: Array-like object holding the points where
                        predictions will be made.  Must have shape
                        ``(n_predict, D)`` or ``(D,)`` (for a single
                        prediction)
        :type testing: ndarray
        :param unc: (optional) Flag indicating if the uncertainties
                    are to be computed.  If ``False`` the method
                    returns ``None`` in place of the uncertainty
                    array. Default value is ``True``.
        :type unc: bool
        :param deriv: (optional) Flag indicating if the derivatives
                      are to be computed.  If ``False`` the method
                      returns ``None`` in place of the derivative
                      array. Default value is ``True``.
        :type deriv: bool
        :param include_nugget: (optional) Flag indicating if the
                                nugget should be included in the
                                predictive variance. Only relevant if
                                ``unc = True``.  Default is ``True``.
        :type include_nugget: bool
        :param allow_not_fit: (optional) Flag that allows predictions
                              to be made even if not all emulators have
                              been fit. Default is ``False`` which
                              will raise an error if any unfitted
                              emulators are present.
        :type allow_not_fit: bool
        :param processes: (optional) Number of processes to use when
                          making the predictions.  Must be a positive
                          integer or ``None`` to use the number of
                          processors on the computer (default is
                          ``None``)
        :type processes: int or None
        :returns: ``PredictResult`` object holding numpy arrays
                  containing the predictions, uncertainties, and
                  derivatives, respectively. Predictions and
                  uncertainties have shape ``(n_emulators,
                  n_predict)`` while the derivatives have shape
                  ``(n_emulators, n_predict, D)``. If the ``do_unc``
                  or ``do_deriv`` flags are set to ``False``, then
                  those arrays are replaced by ``None``.
        :rtype: PredictResult

        """

        testing = np.array(testing)
        if self.D == 1 and testing.ndim == 1:
            testing = np.reshape(testing, (-1, 1))
        elif testing.ndim == 1:
            testing = np.reshape(testing, (1, len(testing)))
        assert testing.ndim == 2, "testing must be a 2D array"

        n_testing, D = np.shape(testing)
        assert D == self.D, "second dimension of testing must be the same as the number of input parameters"

        means = np.zeros([self.n_emulators, n_testing])
        uncs = np.zeros([self.n_emulators, n_testing])
        derivs = np.zeros([self.n_emulators, n_testing, self.D])
        if unc:
            self._mogp_gpu.predict_variance_batch(testing, means, uncs)
        else:
            self._mogp_gpu.predict_variance_batch(testing, means)
        if deriv:
            self._mogp_gpu.predict_deriv(testing, derivs)
        return PredictResult(mean=means, unc=uncs, deriv=derivs)

    def __call__(self, testing, process=None):
        """Interface to predict means by calling the object

        A MultiOutputGP object is callable, which makes predictions of
        the mean only for a given set of inputs. Works similarly to
        the same method of the base GP class.
        """

        return self.predict(testing, unc=False, deriv=False, processes=processes)[0]

    def fit(self, thetas):
        """
        Fit all emulators
        :param thetas: hyperparameters for all emulators. 2D array, 
                    must have first dimension of size n_emulators
        :type thetas: np.array
        """
        self._mogp_gpu.fit(thetas)

    def fit_emulator(self, index, theta):
        """
        Fit a specific emulator
        :param index: index of emulator whose hyperparameters we will set 
        :type index: int
        :param theta: hyperparameters for all emulators. 1D array of length n_param 
        :type theta: np.array
        """
        self._mogp_gpu.fit_emulator(index, theta)

    def get_indices_fit(self):
        """Returns the indices of the emulators that have been fit

        When a ``MultiOutputGP`` class is initialized, none of the
        emulators are fit. Fitting is done by passing the object to an
        external fitting routine, which modifies the object to fit the
        hyperparameters. Any emulators where the fitting fails are
        returned to the initialized state, and this method is used to
        determine the indices of the emulators that have succesfully
        been fit.

        Returns a list of non-negative integers indicating the
        indices of the emulators that have been fit.

        :returns: List of integer indicies indicating the emulators
                  that have been fit. If no emulators have been
                  fit, returns an empty list.
        :rtype: list of int

        """
 #       fitted_indices = []
  #      for idx, em in enumerate(self.emulators):
       #         fitted_indices.append(idx)
   ##         if em.theta is not None:
        return  self._mogp_gpu.get_fitted_indices() 
      #  return fitted_indices
 

    def get_indices_not_fit(self):
        """Returns the indices of the emulators that have not been fit

        When a ``MultiOutputGP`` class is initialized, none of the
        emulators are fit. Fitting is done by passing the object to an
        external fitting routine, which modifies the object to fit the
        hyperparameters. Any emulators where the fitting fails are
        returned to the initialized state, and this method is used to
        determine the indices of the emulators that have not been fit.

        Returns a list of non-negative integers indicating the
        indices of the emulators that have not been fit.

        :returns: List of integer indicies indicating the emulators
                  that have not been fit. If all emulators have been
                  fit, returns an empty list.
        :rtype: list of int

        """
     #   failed_indices = []
      #  for idx, em in enumerate(self.emulators):
       #     if em.theta is None:
       #         failed_indices.append(idx)
       # return failed_indices
        return  self._mogp_gpu.get_unfitted_indices() 

    def get_emulators_fit(self):
        """Returns the emulators that have been fit

        When a ``MultiOutputGP`` class is initialized, none of the
        emulators are fit. Fitting is done by passing the object to an
        external fitting routine, which modifies the object to fit the
        hyperparameters. Any emulators where the fitting fails are
        returned to the initialized state, and this method is used to
        obtain a list of emulators that have successfully been fit.

        Returns a list of ``GaussianProcess`` objects which have been
        fit (i.e. those which have a current valid set of
        hyperparameters).

        :returns: List of ``GaussianProcessGPU`` objects indicating the
                  emulators that have been fit. If no emulators
                  have been fit, returns an empty list.
        :rtype: list of ``GaussianProcessGPU`` objects

        """
        fitted_emulators = []
        for em in self.emulators:
            if em.theta is not None:
                fitted_emulators.append(em)
        return fitted_emulators

    def get_emulators_not_fit(self):
        """Returns the indices of the emulators that have not been fit

        When a ``MultiOutputGP`` class is initialized, none of the
        emulators are fit. Fitting is done by passing the object to an
        external fitting routine, which modifies the object to fit the
        hyperparameters. Any emulators where the fitting fails are
        returned to the initialized state, and this method is used to
        obtain a list of emulators that have not been fit.

        Returns a list of ``GaussianProcess`` objects which have
        not been fit (i.e. those which do not have a current set of
        hyperparameters).

        :returns: List of ``GaussianProcessGPU`` objects indicating the
                  emulators that have not been fit. If all emulators
                  have been fit, returns an empty list.
        :rtype: list of ``GaussianProcessGPU`` objects

        """
        failed_emulators = []
        for em in self.emulators:
            if em.theta is None:
                failed_emulators.append(em)
        return failed_emulators


    def __str__(self):
        """Returns a string representation of the model

        :returns: A string representation of the model (indicates
                  number of sub-components and array shapes) :rtype:
                  str
        """

        return ("Multi-Output Gaussian Process with:\n"+
                 str(self.n_emulators)+" emulators\n"+
                 str(self.n)+" training examples\n"+
                 str(self.D)+" input variables")


def _gp_predict_default_NaN(gp, testing, unc, deriv, include_nugget):
    """Prediction method for GPs that defaults to NaN for unfit GPs

    Wrapper function for the ``GaussianProcess`` predict method that
    returns NaN if the GP is not fit. Allows ``MultiOutputGP`` objects
    that do not have all emulators fit to still return predictions
    with unfit emulator predictions replaced with NaN.

    The first argument to this function is the GP that will be used
    for prediction. All other arguments are the same as the
    arguments for the ``predict`` method of ``GaussianProcess``.

    :param gp: The ``GaussianProcess`` object (or related class) for
               which predictions will be made.
    :type gp: GaussianProcess
    :param testing: Array-like object holding the points where
                    predictions will be made.  Must have shape
                    ``(n_predict, D)`` or ``(D,)`` (for a single
                    prediction)
    :type testing: ndarray
    :param unc: (optional) Flag indicating if the uncertainties
                are to be computed.  If ``False`` the method
                returns ``None`` in place of the uncertainty
                array. Default value is ``True``.
    :type unc: bool
    :param deriv: (optional) Flag indicating if the derivatives
                  are to be computed.  If ``False`` the method
                  returns ``None`` in place of the derivative
                  array. Default value is ``True``.
    :type deriv: bool
    :param include_nugget: (optional) Flag indicating if the
                           nugget should be included in the
                           predictive variance. Only relevant if
                           ``unc = True``.  Default is ``True``.
    :type include_nugget: bool
    :returns: Tuple of numpy arrays holding the predictions,
              uncertainties, and derivatives,
              respectively. Predictions and uncertainties have
              shape ``(n_predict,)`` while the derivatives have
              shape ``(n_predict, D)``. If the ``unc`` or
              ``deriv`` flags are set to ``False``, then those
              arrays are replaced by ``None``.
    :rtype: tuple
    """

    assert isinstance(gp, GaussianProcessBase)

    try:
        return gp.predict(testing, unc, deriv, include_nugget)
    except ValueError:

        n_predict = testing.shape[0]

        if unc:
            unc_array = np.array([np.nan]*n_predict)
        else:
            unc_array = None

        if deriv:
            deriv_array = np.reshape(np.array([np.nan]*n_predict*gp.D),
                                     (n_predict, gp.D))
        else:
            deriv_array = None

        return PredictResult(mean =np.array([np.nan]*n_predict),
                             unc=unc_array, deriv=deriv_array)
