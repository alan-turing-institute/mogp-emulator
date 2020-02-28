from multiprocessing import Pool
import numpy as np
from .GaussianProcess import GaussianProcess, PredictResult
from .MeanFunction import MeanBase
from .Kernel import Kernel, SquaredExponential
from .Priors import Prior
from functools import partial

class MultiOutputGP(object):
    """
    Implementation of a multiple-output Gaussian Process Emulator.

    Essentially a parallelized wrapper for the predict method. To fit in parallel, use the fit_GP_MAP
    routine
    """

    def __init__(self, inputs, targets, mean=None, kernel=SquaredExponential(), priors=[],
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
            raise ValueError("inputs must be 2D array")
        if not (inputs.shape[0] == targets.shape[1]):
            raise ValueError("the first dimension of inputs must be the same length as the second dimension of targets (or first if targets is 1D))")

        self.n_emulators = targets.shape[0]
        self.n = inputs.shape[0]
        self.D = inputs.shape[1]

        if mean is None or issubclass(type(mean), MeanBase):
            mean = self.n_emulators*[mean]

        assert isinstance(mean, list), "mean must be None, a mean function, or a list of None/mean functions"
        assert len(mean) == self.n_emulators

        if issubclass(type(kernel), Kernel):
            kernel = self.n_emulators*[kernel]

        assert isinstance(kernel, list), "kernel must be a Kernal subclass or a list of Kernel subclasses"
        assert len(kernel) == self.n_emulators

        if len(priors) == 0:
            priors = self.n_emulators*[[]]

        assert isinstance(priors, list), "priors must be a list of lists of Priors/None"
        assert len(priors) == self.n_emulators

        self.emulators = [ GaussianProcess(inputs, single_target, m, k, p, nugget, inputdict, use_patsy)
                           for (single_target, m, k, p) in zip(targets, mean, kernel, priors)]


    def predict(self, testing, unc = True, deriv = True, processes = None):
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
        and the derivatives with respect to each input parameter. If the uncertainties are
        computed, they are returned as the second output from the method as an
        ``(n_emulators, n_predict)`` shaped numpy array. If the derivatives are computed,
        they are returned as the third output from the method as an
        ``(n_emulators, n_predict, D)`` shaped numpy array.

        As with the fitting, this computation can be done independently for each emulator
        and thus can be done in parallel.

        :param testing: Array-like object holding the points where predictions will be made.
                        Must have shape ``(n_predict, D)`` or ``(D,)`` (for a single prediction)
        :type testing: ndarray
        :param do_deriv: (optional) Flag indicating if the derivatives are to be computed.
                         If ``False`` the method returns ``None`` in place of the derivative
                         array. Default value is ``True``.
        :type do_deriv: bool
        :param do_unc: (optional) Flag indicating if the uncertainties are to be computed.
                         If ``False`` the method returns ``None`` in place of the uncertainty
                         array. Default value is ``True``.
        :type do_unc: bool
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

        with Pool(processes) as p:
            predict_vals = p.starmap(GaussianProcess.predict, [(gp, testing) for gp in self.emulators])

        # repackage predictions into numpy arrays

        predict_unpacked, unc_unpacked, deriv_unpacked = [np.array(t) for t in zip(*predict_vals)]

        if not unc:
            unc_unpacked = None
        if not deriv:
            deriv_unpacked = None

        return PredictResult(mean=predict_unpacked, unc=unc_unpacked, deriv=deriv_unpacked)

    def __call__(self, testing):
        "Interface to predict means like the base GP class"

        return self.predict(testing, unc=False, deriv=False, processes=None)[0]

    def __str__(self):
        """
        Returns a string representation of the model

        :returns: A string representation of the model (indicates number of sub-components
                  and array shapes)
        :rtype: str
        """

        return ("Multi-Output Gaussian Process with:\n"+
                 str(self.get_n_emulators())+" emulators\n"+
                 str(self.get_n())+" training examples\n"+
                 str(self.get_D())+" input variables")

