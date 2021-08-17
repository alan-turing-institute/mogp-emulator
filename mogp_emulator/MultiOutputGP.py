from multiprocessing import Pool
import platform
import numpy as np
from mogp_emulator.GaussianProcess import (
    GaussianProcessBase,
    GaussianProcess,
    PredictResult
)
from mogp_emulator.GaussianProcessGPU import GaussianProcessGPU
from mogp_emulator.MeanFunction import MeanBase
from mogp_emulator.Kernel import Kernel, SquaredExponential, Matern52
from mogp_emulator.Priors import Prior


class MultiOutputGP(object):
    """Implementation of a multiple-output Gaussian Process Emulator.

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
                 nugget="adaptive", inputdict={}, use_patsy=True, use_gpu=False):
        """
        Create a new multi-output GP Emulator
        """

        # if use_gpu is selected, check whether we found the GPU .so file, and raise error if not
        self.use_gpu = use_gpu
        if self.use_gpu:
            self.GPClass = GaussianProcessGPU
        else:
            self.GPClass = GaussianProcess

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

        self.n_emulators = targets.shape[0]
        self.n = inputs.shape[0]
        self.D = inputs.shape[1]

        if mean is None or isinstance(mean, str) or issubclass(type(mean), MeanBase):
            mean = self.n_emulators*[mean]

        assert isinstance(mean, list), "mean must be None, a string, a mean function, or a list of None/string/mean functions"
        assert len(mean) == self.n_emulators

        if isinstance(kernel, str):
            if kernel == "SquaredExponential":
                kernel = SquaredExponential()
            elif kernel == "Matern52":
                kernel = Matern52()
            else:
                raise ValueError("provided kernel '{}' not a supported kernel type".format(kernel))
        if issubclass(type(kernel), Kernel):
            kernel = self.n_emulators*[kernel]

        assert isinstance(kernel, list), "kernel must be a Kernel subclass or a list of Kernel subclasses"
        assert len(kernel) == self.n_emulators

        if priors is None:
            priors = []
        assert isinstance(priors, list), "priors must be a list of lists of Priors/None"

        if len(priors) == 0:
            priors = self.n_emulators*[[]]

        if not isinstance(priors[0], list):
            priors = self.n_emulators*[priors]

        assert len(priors) == self.n_emulators

        if isinstance(nugget, (str, float)):
            nugget = self.n_emulators*[nugget]

        assert isinstance(nugget, list), "nugget must be a string, float, or a list of strings and floats"
        assert len(nugget) == self.n_emulators

        self.emulators = [ self.GPClass(inputs, single_target, m, k, p, n, inputdict, use_patsy)
                           for (single_target, m, k, p, n) in zip(targets, mean, kernel, priors, nugget)]


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

        if not processes is None:
            processes = int(processes)
            assert processes > 0, "number of processes must be a positive integer"

        if allow_not_fit:
            predict_method = _gp_predict_default_NaN
        else:
            predict_method = self.GPClass.predict

        if platform.system() == "Windows" or self.use_gpu:
            predict_vals = [predict_method(gp, testing, unc, deriv, include_nugget)
                            for gp in self.emulators]
        else:
            with Pool(processes) as p:
                predict_vals = p.starmap(predict_method,
                                         [(gp, testing, unc, deriv, include_nugget)
                                          for gp in self.emulators])

        # repackage predictions into numpy arrays

        predict_unpacked, unc_unpacked, deriv_unpacked = [np.array(t) for t in zip(*predict_vals)]

        if not unc:
            unc_unpacked = None
        if not deriv:
            deriv_unpacked = None

        return PredictResult(mean=predict_unpacked, unc=unc_unpacked, deriv=deriv_unpacked)

    def __call__(self, testing, process=None):
        """Interface to predict means by calling the object

        A MultiOutputGP object is callable, which makes predictions of
        the mean only for a given set of inputs. Works similarly to
        the same method of the base GP class.
        """

        return self.predict(testing, unc=False, deriv=False, processes=processes)[0]


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

        return [idx for (failed_fit, idx)
                in zip([em.theta is None for em in self.emulators],
                       list(range(len(self.emulators)))) if not failed_fit]

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

        return [idx for (failed_fit, idx)
                in zip([em.theta is None for em in self.emulators],
                       list(range(len(self.emulators)))) if failed_fit]

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

        :returns: List of ``GaussianProcess`` objects indicating the
                  emulators that have been fit. If no emulators
                  have been fit, returns an empty list.
        :rtype: list of ``GaussianProcess`` objects

        """

        return [gpem for (failed_fit, gpem)
                in zip([em.theta is None for em in self.emulators],
                       self.emulators) if not failed_fit]

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

        :returns: List of ``GaussianProcess`` objects indicating the
                  emulators that have not been fit. If all emulators
                  have been fit, returns an empty list.
        :rtype: list of ``GaussianProcess`` objects

        """

        return [gpem for (failed_fit, gpem)
                in zip([em.theta is None for em in self.emulators],
                       self.emulators) if failed_fit]


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
