import numpy as np
from .MeanFunction import MeanFunction, MeanBase
from .Kernel import Kernel, SquaredExponential
from scipy import linalg
from scipy.optimize import OptimizeResult
from .linalg.cholesky import jit_cholesky

class GaussianProcess(object):
    """
    Implementation of a Gaussian Process Emulator.

    This class provides a representation of a Gaussian Process Emulator. It contains
    methods for fitting the GP to a given set of hyperparameters, computing the
    negative log marginal likelihood and its derivatives, and making predictions on
    unseen data. Note that routines to estimate hyperparameters are not included in the
    class definition, and are instead provided externally to facilitate implementation
    of high performance versions.

    The required arguments to initialize a GP is a set of training data consisting
    of the inputs and targets for each of those inputs. These must be numpy arrays
    whose first axis is of the same length. Targets must be a 1D array, while inputs
    can be 2D or 1D. If inputs is 1D, then it is assumed that the length of the second
    axis is unity.

    Optional arguments are the particular mean function to use (default is zero mean),
    the covariance kernel to use (default is the squared exponential covariance),
    and the method for handling the nugget parameter. The nugget is additional
    "noise" that is added to the diagonal of the covariance kernel as a variance.
    This nugget can represent uncertainty in the target values themselves, or simply be
    used to stabilize the numerical inversion of the covariance matrix. The nugget
    can be fixed (a non-negative float), can be found adaptively (make the noise only
    as large as necessary to successfully invert the covariance matrix), or can be
    fit as a hyperparameter (negative float).

    The internal emulator structure involves arrays for the inputs, targets, and hyperparameters.
    Other useful information are the number of training examples ``n``, the number of input
    parameters ``D``, and the number of hyperparameters ``n_params``. These parameters can
    be obtained externally through provided methods

    Example: ::

        >>> import numpy as np
        >>> from mogp_emulator import GaussianProcess
        >>> x = np.array([[1., 2., 3.], [4., 5., 6.]])
        >>> y = np.array([4., 6.])
        >>> gp = GaussianProcess(x, y)
        >>> print(gp)
        Gaussian Process with 2 training examples and 3 input variables
        >>> gp.get_n()
        2
        >>> gp.get_D()
        3
        >>> gp.get_n_params()
        5
        >>> gp.fit(np.zeros(gp.get_n_params()))
        >>> x_predict = np.array([[2., 3., 4.], [7., 8., 9.]])
        >>> gp.predict(x_predict)
        (array([4.74687618, 6.84934016]), array([0.01639298, 1.05374973]),
        array([[8.91363045e-05, 7.18827798e-01, 3.74439445e-16],
               [4.64005897e-06, 3.74191346e-02, 1.94917337e-17]]))

    """
    def __init__(self, inputs, targets, mean=None, kernel=SquaredExponential(), nugget=None,
                 inputdict = {}, use_patsy=True):
        """
        Create a new GaussianProcess Emulator

        Creates a new GaussianProcess Emulator from either the input data and targets to be fit and
        optionally a mean function, covariance kernel, and nugget parameter/method.

        Required arguments are numpy arrays ``inputs`` and ``targets``, described below.
        Additional arguments are ``mean`` to specify the mean function (default is
        ``None`` for zero mean), ``kernel`` to specify the covariance kernel (default
        is the squared exponential kernel), and ``nugget`` to specify how the nugget
        parameter is handled (see below; default is to fit the nugget adaptively).

        ``inputs`` is a 2D array-like object holding the input data, whose shape is
        ``n`` by ``D``, where ``n`` is the number of training examples to be fit and ``D``
        is the number of input variables to each simulation. If ``inputs`` is a 1D array,
        it is assumed that ``D = 1``.

        ``targets`` is the target data to be fit by the emulator, also held in an array-like
        object. This must be a 1D array of length ``n``.

        ``nugget`` is the additional noise added to the emulator targets when fitting. This
        can take on values ``None`` (meaning noise will be added adaptively to
        stabilize fitting), a non-negative float (meaning a fixed noise level
        will be used), or a negative float (meaning that the nugget is considered to be
        a hyperparameter). If no value is specified for the ``nugget`` parameter, ``None``
        is the default.


        :param inputs: Numpy array holding emulator input parameters. Must be 1D with length
                       ``n`` or 2D with shape ``n`` by ``D``, where ``n`` is the number of
                       training examples and ``D`` is the number of input parameters for
                       each output.
        :type inputs: ndarray
        :param targets: Numpy array holding emulator targets. Must be 1D with length ``n``
        :type targets: ndarray
        :param mean: Mean function to be used (optional, default is ``None`` for a zero mean)
        :type mean: None or MeanFunction
        :param kernel: Covariance kernel to be used (optional, default is Squared Exponential)
        :type kernel: Kernel
        :param nugget: Noise to be added to the diagonal or ``None``. A non-negative float
                       specifies the noise level explicitly, a negative float indicates that
                       the nugget is considered a hyperparameter, and ``None`` indicates
                       that the noise will set to be as small as possible to ensure stable
                       inversion of the covariance matrix. Optional, default is ``None``.
        :type nugget: float or None
        :returns: New ``GaussianProcess`` instance
        :rtype: GaussianProcess

        """
        inputs = np.array(inputs)
        if inputs.ndim == 1:
            inputs = np.reshape(inputs, (-1, 1))
        assert inputs.ndim == 2

        targets = np.array(targets)
        assert targets.ndim == 1
        assert targets.shape[0] == inputs.shape[0]

        if not issubclass(type(mean), MeanBase):
            if not (mean is None or isinstance(mean, str)):
                raise ValueError("provided mean function must be a subclass of MeanFunction,"+
                                 " a string formula, or None")

        if not issubclass(type(kernel), Kernel):
            raise ValueError("provided kernel is not a subclass of Kernel")

        if not nugget is None:
            try:
                nugget = float(nugget)
            except ValueError:
                raise ValueError("nugget must be None or a float")

        self.inputs = inputs
        self.targets = targets

        self.n, self.D = self.inputs.shape

        if not issubclass(type(mean), MeanBase):
            self.mean = MeanFunction(mean, inputdict, use_patsy)
        else:
            self.mean = mean

        self.kernel = kernel

        self.nugget = nugget

        self.theta = None

    def get_n(self):
        """
        Returns number of training examples for the emulator

        :returns: Number of training examples for the emulator object
        :rtype: int
        """

        return self.n

    def get_D(self):
        """
        Returns number of inputs for the emulator

        :returns: Number of inputs for the emulator object
        :rtype: int
        """

        return self.D

    def get_n_params(self):
        """
        Returns number of hyperparameters

        Returns the number of hyperparameters for the emulator. The number depends on the
        choice of mean function, covariance function, and nugget strategy, and possibly the
        number of inputs for certain choices of the mean function.

        :returns: Number of hyperparameters
        :rtype: int
        """

        return self.mean.get_n_params(self.inputs) + self.D + 2

    def get_inputs(self):
        """
        Returns inputs for the emulator as a numpy array

        :returns: Emulator inputs, 2D array with shape ``(n, D)``
        :rtype: ndarray
        """
        return self.inputs

    def get_targets(self):
        """
        Returns targets for the emulator as a numpy array

        :returns: Emulator targets, 1D array with shape ``(n,)``
        :rtype: ndarray
        """
        return self.targets

    def get_params(self):
        """
        Returns emulator hyperparameters

        Returns current hyperparameters for the emulator as a numpy array if they have been fit.
        If no parameters have been fit, returns ``None``. Note that the number of parameters
        depends on the mean function, so the length of this array will vary across instances.

        :returns: Current parameter values (numpy array of length ``n_params``), or ``None`` if the
                  parameters have not been fit.
        :rtype: ndarray or None
        """

        return self.theta

    def get_nugget(self):
        """
        Returns emulator nugget parameter

        Returns current value of the nugget parameter. If the nugget is selected adaptively,
        returns ``None``. If the nugget is treated as a variable hyperparameter nugget
        will be negative.

        :returns: Current nugget value, either a float or ``None``
        :rtype: float or None
        """

        return self.nugget

    def set_nugget(self, nugget):
        """
        Set the nugget parameter for the emulator

        Method for changing the ``nugget`` parameter for the emulator. When a new emulator is
        initilized, this is set to None.

        The ``nugget`` parameter controls how noise is added to the covariance matrix in order to
        stabilize the inversion or smooth the emulator predictions. If ``nugget`` is a non-negative
        float, then that particular value is used for the nugget. Note that setting this parameter
        to be zero enforces that the emulator strictly interpolates between points. A negative
        float means that the nugget is treated as a hyperparameter, and is the last entry
        in the ``params`` array. Alternatively, if ``nugget`` is set to be ``None``, the fitting
        routine will adaptively make the noise parameter as large as is needed to ensure that the
        emulator can be fit.

        :param nugget: Controls how noise is added to the emulator. If ``nugget`` is a nonnegative
                       float, then this manually sets the noise parameter (if negative, this will
                       lead to an error), with ``nugget = 0`` resulting in interpolation with no
                       smoothing noise added. A negative float means that the nugget is treated
                       as a hyperparameter. ``nugget = None`` will adaptively select the
                       smallest value of the noise term that still leads to a stable inversion of
                       the matrix. Default behavior is ``nugget = None``.
        :type nugget: None or float
        :returns: None
        :rtype: None
        """

        if not nugget is None:
            try:
                nugget = float(nugget)
            except ValueError:
                raise ValueError("nugget must be None or a float")

        self.nugget = nugget

    def fit(self, theta):
        """
        Fits the emulator and sets the parameters

        Pre-calculates the matrices needed to compute the log-likelihood and its derivatives
        and make subsequent predictions. This is called any time the hyperparameter values are
        changed in order to ensure that all the information is needed to evaluate the
        log-likelihood and its derivatives, which are needed when fitting the optimal
        hyperparameters.

        The method computes the mean function and covariance matrix and inverts the covariance
        matrix using the method specified by the value of ``nugget``. The factorized matrix
        and the product of the inverse with the difference between the targets and the mean
        are cached for later use, and the negative marginal log-likelihood is also cached.
        This method has no inputs and no return value, but it does modify the state of the object.

        :returns: None
        """
        theta = np.array(theta)

        assert theta.shape == (self.get_n_params(),)

        self.theta = theta

        switch = self.mean.get_n_params(self.inputs)

        m = self.mean.mean_f(self.inputs, self.theta[:switch])
        Q = self.kernel.kernel_f(self.inputs, self.inputs, self.theta[switch:-1])

        if self.nugget == None:
            self.L, nugget = jit_cholesky(Q)
            Z = Q + nugget*np.eye(self.n)
        else:
            if self.nugget < 0.:
                nugget = np.exp(self.theta[-1])
            else:
                nugget = self.nugget
            Z = Q + nugget*np.eye(self.n)
            self.L = linalg.cholesky(Z, lower=True)

        self.invQt = linalg.cho_solve((self.L, True), self.targets - m)

        self.current_loglike = 0.5*(2.0*np.sum(np.log(np.diag(self.L))) +
                                    np.dot(self.targets - m, self.invQt) +
                                    self.n*np.log(2. * np.pi))


    def loglikelihood(self, theta):
        """
        Calculate the negative log-likelihood at a particular value of the hyperparameters

        Calculate the negative log-likelihood for the given set of parameters. Calling this
        method sets the parameter values and computes the needed inverse matrices in order
        to evaluate the log-likelihood and its derivatives. In addition to returning the
        log-likelihood value, it stores the current value of the hyperparameters and
        log-likelihood in attributes of the object.

        :param theta: Value of the hyperparameters. Must be array-like with shape ``(n_params,)``
        :type theta: ndarray
        :returns: negative log-likelihood
        :rtype: float
        """

        if self.theta is None or not np.allclose(theta, self.theta, rtol=1.e-10, atol=1.e-15):
            self.fit(theta)

        return self.current_loglike

    def loglike_deriv(self, theta):
        """
        Calculate the partial derivatives of the negative log-likelihood

        Calculate the partial derivatives of the negative log-likelihood with respect to
        the hyperparameters. Note that this function is normally used only when fitting
        the hyperparameters, and it is not needed to make predictions.

        During normal use, the ``loglike_deriv`` method is called after evaluating the
        ``loglikelihood`` method. The implementation takes advantage of this by reusing
        cached results, as the factorized covariance matrix is expensive to compute and is
        used by the ``loglikelihood``, ``loglike_deriv``, and ``loglike_hessian`` methods.
        If the function is evaluated with a different set of parameters than was previously
        used to set the log-likelihood, the method calls ``fit`` (and subsequently resets
        the cached information).

        :param theta: Value of the hyperparameters. Must be array-like with shape
                      ``(n_params,)``
        :type theta: ndarray
        :returns: partial derivatives of the negative log-likelihood with respect to the
                  hyperparameters (array with shape ``(n_params,)``)
        :rtype: ndarray
        """

        theta = np.array(theta)

        assert theta.shape == (self.get_n_params(),), "Parameter vector must have length number of inputs + 1"

        if self.theta is None or not np.allclose(theta, self.theta, rtol=1.e-10, atol=1.e-15):
            self.fit(theta)

        partials = np.zeros(self.get_n_params())

        switch = self.mean.get_n_params(self.inputs)

        dmdtheta = self.mean.mean_deriv(self.inputs, self.theta[:switch])
        dKdtheta = self.kernel.kernel_deriv(self.inputs, self.inputs, self.theta[switch:-1])

        partials[:switch] = -np.dot(dmdtheta, self.invQt)

        for d in range(self.D + 1):
            invQ_dot_dKdtheta_trace = np.trace(linalg.cho_solve((self.L, True), dKdtheta[d]))
            partials[switch + d] = -0.5*(np.dot(self.invQt, np.dot(dKdtheta[d], self.invQt)) -
                                         invQ_dot_dKdtheta_trace)

        if not self.nugget is None and self.nugget < 0.:
            nugget = np.exp(self.theta[-1])
            partials[-1] = 0.5*nugget*(np.trace(linalg.cho_solve((self.L, True), np.eye(self.n))) -
                                       np.dot(self.invQt, self.invQt))

        return partials

    def loglike_hessian(self, theta):
        """
        Calculate the Hessian of the negative log-likelihood

        Calculate the Hessian of the negative log-likelihood with respect to
        the hyperparameters. Note that this function is normally used only when fitting
        the hyperparameters, and it is not needed to make predictions. It is also used
        to estimate an appropriate step size when fitting hyperparameters using
        the lognormal approximation or MCMC sampling.

        When used in an optimization routine, the ``loglike_hessian`` method is called after
        evaluating the ``loglikelihood`` method. The implementation takes advantage of
        this by storing the inverse of the covariance matrix, which is expensive to
        compute and is used by the ``loglikelihood`` and ``loglike_deriv`` methods as well.
        If the function is evaluated with a different set of parameters than was previously
        used to set the log-likelihood, the method calls ``fit`` to compute the needed
        information and changes the cached values.

        :param theta: Value of the hyperparameters. Must be array-like with shape
                      ``(n_params,)``
        :type theta: ndarray
        :returns: Hessian of the negative log-likelihood (array with shape
                  ``(n_params, n_params)``)
        :rtype: ndarray
        """

        assert theta.shape == (self.get_n_params(),), "Parameter vector must have length number of inputs + 1"

        if self.theta is None or not np.allclose(theta, self.theta, rtol=1.e-10, atol=1.e-15):
            self.fit(theta)

        hessian = np.zeros((self.get_n_params(), self.get_n_params()))

        switch = self.mean.get_n_params(self.inputs)

        dmdtheta = self.mean.mean_deriv(self.inputs, self.theta[:switch])
        d2mdtheta2 = self.mean.mean_hessian(self.inputs, self.theta[:switch])
        dKdtheta = self.kernel.kernel_deriv(self.inputs, self.inputs, self.theta[switch:-1])
        d2Kdtheta2 = self.kernel.kernel_hessian(self.inputs, self.inputs, self.theta[switch:-1])

        hessian[:switch, :switch] = -(np.dot(d2mdtheta2, self.invQt) -
                                      np.dot(dmdtheta, linalg.cho_solve((self.L, True),
                                                                        np.transpose(dmdtheta))))

        hessian[:switch, switch:-1] = np.dot(dmdtheta,
                                             linalg.cho_solve((self.L, True),
                                                              np.transpose(np.dot(dKdtheta, self.invQt))))

        hessian[switch:-1, :switch] = np.transpose(hessian[:switch, switch:-1])

        for d1 in range(self.D + 1):
            invQ_dot_d1 = linalg.cho_solve((self.L, True), dKdtheta[d1])
            for d2 in range(self.D + 1):
                invQ_dot_d2 = linalg.cho_solve((self.L, True), dKdtheta[d2])
                invQ_dot_d1d2 = linalg.cho_solve((self.L, True), d2Kdtheta2[d1, d2])
                term_1 = np.linalg.multi_dot([self.invQt,
                                              2.*np.dot(dKdtheta[d1], invQ_dot_d2) - d2Kdtheta2[d1, d2],
                                              self.invQt])
                term_2 = np.trace(np.dot(invQ_dot_d1, invQ_dot_d2) - invQ_dot_d1d2)
                hessian[switch + d1, switch + d2] = 0.5*(term_1 - term_2)

        if not self.nugget is None and self.nugget < 0.:
            nugget = np.exp(self.theta[-1])
            invQinvQt = linalg.cho_solve((self.L, True), self.invQt)
            hessian[:switch, -1] = nugget*np.dot(dmdtheta, invQinvQt)
            for d in range(self.D + 1):
                hessian[switch + d, -1] = nugget*(np.linalg.multi_dot([self.invQt, dKdtheta[d], invQinvQt]) -
                                                  0.5*np.trace(linalg.cho_solve((self.L, True),
                                                                                np.dot(dKdtheta[d],
                                                                                       linalg.cho_solve((self.L, True),
                                                                                                        np.eye(self.n))))))

            hessian[-1, -1] = 0.5*nugget*(np.trace(linalg.cho_solve((self.L, True), np.eye(self.n))) -
                                                   np.dot(self.invQt, self.invQt))
            hessian[-1, -1] += nugget**2*(np.dot(self.invQt, invQinvQt) -
                                          0.5*np.trace(linalg.cho_solve((self.L, True),
                                                                        linalg.cho_solve((self.L, True),
                                                                                         np.eye(self.n)))))

            hessian[-1, :-1] = np.transpose(hessian[:-1, -1])

        return hessian

    def predict(self, testing, unc=True, deriv=True):
        """
        Make a prediction for a set of input vectors for a single set of hyperparameters

        Makes predictions for the emulator on a given set of input vectors. The input vectors
        must be passed as a ``(n_predict, D)``, ``(n_predict,)`` or ``(D,)`` shaped array-like
        object, where ``n_predict`` is the number of different prediction points under
        consideration and ``D`` is the number of inputs to the emulator. If the prediction
        inputs array is 1D and ``D == 1`` for the GP instance, then the 1D array must have
        shape ``(n_predict,)``. Otherwise, if the array is 1D it must have shape
        ``(D,)``, and the method assumes ``n_predict == 1``. The prediction is returned as an
        ``(n_predict, )`` shaped numpy array as the first return value from the method.

        Optionally, the emulator can also calculate the variances in the predictions
        and the derivatives with respect to each input parameter. If the uncertainties are
        computed, they are returned as the second output from the method as an ``(n_predict,)``
        shaped numpy array. If the derivatives are computed, they are returned as the third
        output from the method as an ``(n_predict, D)`` shaped numpy array.

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
        :returns: Tuple of numpy arrays holding the predictions, uncertainties, and derivatives,
                  respectively. Predictions and uncertainties have shape ``(n_predict,)``
                  while the derivatives have shape ``(n_predict, D)``. If the ``unc`` or
                  ``deriv`` flags are set to ``False``, then those arrays are replaced by
                  ``None``.
        :rtype: tuple
        """

        if self.theta is None:
            raise ValueError("hyperparameters have not been fit for this Gaussian Process")

        testing = np.array(testing)
        if self.D == 1 and testing.ndim == 1:
            testing = np.reshape(testing, (-1, 1))
        elif testing.ndim == 1:
            testing = np.reshape(testing, (1, len(testing)))
        assert testing.ndim == 2

        n_testing, D = np.shape(testing)
        assert D == self.D

        sigma_2 = np.exp(self.theta[-2])

        switch = self.mean.get_n_params(testing)
        mtest = self.mean.mean_f(testing, self.theta[:switch])
        Ktest = self.kernel.kernel_f(self.inputs, testing, self.theta[switch:-1])

        mu = mtest + np.dot(Ktest.T, self.invQt)

        var = None
        if unc:
            var = np.maximum(sigma_2 - np.sum(Ktest*linalg.cho_solve((self.L, True), Ktest), axis=0),
                             0.)

        inputderiv = None
        if deriv:
            inputderiv = np.zeros((n_testing, self.D))
            mean_deriv = self.mean.mean_inputderiv(testing, self.theta[:switch])
            kern_deriv = self.kernel.kernel_inputderiv(testing, self.inputs, self.theta[switch:-1])
            inputderiv = np.transpose(mean_deriv + np.dot(kern_deriv, self.invQt))

        return PredictResult(mean=mu, unc=var, deriv=inputderiv)

    def __str__(self):
        """
        Returns a string representation of the model

        :returns: A string representation of the model (indicates number of training examples
                  and inputs)
        :rtype: str
        """

        return ("Gaussian Process with " + str(self.n) + " training examples and " +
                str(self.D) + " input variables")

class PredictResult(dict):
    """
    Prediction results object

    Dictionary-like object containing mean, uncertainty (variance), and derivatives with
    respect to the inputs of an emulator prediction. Values can be accessed like a dictionary
    with keys ``'mean'``, ``'unc'``, and ``'deriv'`` (or indices 0, 1, and 2 for the mean,
    uncertainty, and derivative for backwards compatability), or using attributes
    (``p.mean`` if ``p`` is an instance of ``PredictResult``). Also supports iteration and
    unpacking with the ordering ``(mean, unc, deriv)`` to be consistent with indexing behavior.

    Code is mostly based on scipy's ``OptimizeResult`` class, with some additional code to
    support iteration and integer indexing.

    :ivar mean: Predicted mean for each input point. Numpy array with shape ``(n_predict,)``
    :type mean: ndarray
    :ivar unc: Predicted variance for each input point. Numpy array with shape ``(n_predict,)``
    :type mean: ndarray
    :ivar deriv: Predicted derivative with respect to the inputs for each input point.
                 Numpy array with shape ``(n_predict, D)``
    :type deriv: ndarray
    """

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if self.index == 0:
            self.index += 1
            return self['mean']
        elif self.index == 1:
            self.index += 1
            return self['unc']
        elif self.index == 2:
            self.index += 1
            return self['deriv']
        else:
            raise StopIteration


    def __getitem__(self, key):
        if not isinstance(key, (int, str)):
            raise KeyError(key)
        if key == 0:
            newkey = "mean"
        elif key == 1:
            newkey = "unc"
        elif key == 2:
            newkey = "deriv"
        else:
            newkey = key

        return dict.__getitem__(self, newkey)

    def __repr__(self):
        if self.keys():
            m = max(map(len, list(self.keys()))) + 1
            return '\n'.join([k.rjust(m) + ': ' + repr(v)
                              for k, v in zip(["mean", "unc", "deriv"], self)])
        else:
            return self.__class__.__name__ + "()"