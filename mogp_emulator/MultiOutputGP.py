from multiprocessing import Pool
import platform
import numpy as np
from mogp_emulator.GaussianProcess import GaussianProcess, PredictResult
from mogp_emulator.MeanFunction import MeanBase
from mogp_emulator.Kernel import Kernel, SquaredExponential, Matern52
from mogp_emulator.Priors import Prior

class MultiOutputGP(object):
    """
    Implementation of a multiple-output Gaussian Process Emulator.

    Essentially a parallelized wrapper for the predict method. To fit in parallel, use the fit_GP_MAP
    routine

    Required arguments are ``inputs`` and ``targets``, both of which must be numpy arrays. ``inputs``
    can be 1D or 2D (if 1D, assumes second axis has length 1). ``targets`` can be 1D or 2D (if 2D,
    assumes a single emulator and the first axis has length 1).

    Optional arguments specify how each individual emulator is constructed, including the mean
    function, kernel, priors, and how to handle the nugget. Each argument can take values allowed
    by the base ``GaussianProcess`` class, in which case all emulators are assumed to use the
    same value. Any of these arguments can alternatively be a list of values with length matching
    the number of emulators to set those values individually.

    Additional keyword arguments include ``inputdict``, and ``use_patsy``, which control how strings
    are parsed to mean functions, if using.
    """

    def __init__(self, inputs, targets, mean=None, kernel=SquaredExponential(), priors=None,
                 nugget="adaptive", inputdict={}, use_patsy=True):
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

        assert isinstance(kernel, list), "kernel must be a Kernal subclass or a list of Kernel subclasses"
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

        self.emulators = [ GaussianProcess(inputs, single_target, m, k, p, n, inputdict, use_patsy)
                           for (single_target, m, k, p, n) in zip(targets, mean, kernel, priors, nugget)]


    def predict(self, testing, unc=True, deriv=True, include_nugget=True, processes=None):
        """
        Make a prediction for a set of input vectors

        Makes predictions for each of the emulators on a given set of input vectors. The
        input vectors must be passed as a ``(n_predict, D)`` or ``(D,)`` shaped array-like
        object, where ``n_predict`` is the number of different prediction points under
        consideration and ``D`` is the number of inputs to the emulator. If the prediction
        inputs array has shape ``(D,)``, then the method assumes ``n_predict == 1``.
        The prediction points are passed to each emulator and the predictions are collected
        into an ``(n_emulators, n_predict)`` shaped numpy array as the first return value
        from the method.

        Optionally, the emulator can also calculate the uncertainties in the predictions
        (as a variance) and the derivatives with respect to each input parameter. If the
        uncertainties are computed, they are returned as the second output from the method
        as an ``(n_emulators, n_predict)`` shaped numpy array. If the derivatives are
        computed, they are returned as the third output from the method as an
        ``(n_emulators, n_predict, D)`` shaped numpy array. Finally, if uncertainties
        are computed, the ``include_nugget`` flag determines if the uncertainties should
        include the nugget. By default, this is set to ``True``.

        As with the fitting, this computation can be done independently for each emulator
        and thus can be done in parallel.

        :param testing: Array-like object holding the points where predictions will be made.
                        Must have shape ``(n_predict, D)`` or ``(D,)`` (for a single prediction)
        :type testing: ndarray
        :param unc: (optional) Flag indicating if the uncertainties are to be computed.
                    If ``False`` the method returns ``None`` in place of the uncertainty
                    array. Default value is ``True``.
        :type unc: bool
        :param deriv: (optional) Flag indicating if the derivatives are to be computed.
                      If ``False`` the method returns ``None`` in place of the derivative
                      array. Default value is ``True``.
        :type deriv: bool
        :param include_nugget: (optional) Flag indicating if the nugget should be included
                               in the predictive variance. Only relevant if ``unc = True``.
                               Default is ``True``.
        :type include_nugget: bool
        :param processes: (optional) Number of processes to use when making the predictions.
                          Must be a positive integer or ``None`` to use the number of
                          processors on the computer (default is ``None``)
        :type processes: int or None
        :returns: Tuple of numpy arrays holding the predictions, uncertainties, and derivatives,
                  respectively. Predictions and uncertainties have shape ``(n_emulators, n_predict)``
                  while the derivatives have shape ``(n_emulators, n_predict, D)``. If
                  the ``do_unc`` or ``do_deriv`` flags are set to ``False``, then those arrays
                  are replaced by ``None``.
        :rtype: tuple
        """

        testing = np.array(testing)
        if testing.shape == (self.D,):
            testing = np.reshape(testing, (1, self.D))
        assert len(testing.shape) == 2, "testing must be a 2D array"
        assert testing.shape[1] == self.D, "second dimension of testing must be the same as the number of input parameters"
        if not processes is None:
            processes = int(processes)
            assert processes > 0, "number of processes must be a positive integer"

        if platform.system() == "Windows":
            predict_vals = [GaussianProcess.predict(gp, testing, unc, deriv, include_nugget) for gp in self.emulators]
        else:
            with Pool(processes) as p:
                predict_vals = p.starmap(GaussianProcess.predict, [(gp, testing, unc, deriv, include_nugget) for gp in self.emulators])

        # repackage predictions into numpy arrays

        predict_unpacked, unc_unpacked, deriv_unpacked = [np.array(t) for t in zip(*predict_vals)]

        if not unc:
            unc_unpacked = None
        if not deriv:
            deriv_unpacked = None

        return PredictResult(mean=predict_unpacked, unc=unc_unpacked, deriv=deriv_unpacked)

    def __call__(self, testing):
        """
        Interface to predict means by calling the object

        A MultiOutputGP object is callable, which makes predictions of the mean only
        for a given set of inputs. Works similarly to the same method of the base GP
        class. Predictions are made in parallel using the number of available processors.
        """

        return self.predict(testing, unc=False, deriv=False, processes=None)[0]

    def __str__(self):
        """
        Returns a string representation of the model

        :returns: A string representation of the model (indicates number of sub-components
                  and array shapes)
        :rtype: str
        """

        return ("Multi-Output Gaussian Process with:\n"+
                 str(self.n_emulators)+" emulators\n"+
                 str(self.n)+" training examples\n"+
                 str(self.D)+" input variables")

