import numpy as np
<<<<<<< .merge_file_F65fzN
from .Kernel import SquaredExponential
from .MCMC import sample_MCMC
from scipy.optimize import minimize
from scipy import linalg
from scipy.linalg import lapack
import logging
import warnings
=======
from mogp_emulator.MeanFunction import MeanFunction, MeanBase
from mogp_emulator.Kernel import Kernel, SquaredExponential, Matern52
from mogp_emulator.Priors import Prior
from scipy import linalg
from scipy.optimize import OptimizeResult
from mogp_emulator.linalg.cholesky import jit_cholesky
>>>>>>> .merge_file_wR9ZhR

class GaussianProcess(object):
    """
    Implementation of a Gaussian Process Emulator.

    This class provides a representation of a Gaussian Process Emulator. It contains
    methods for fitting the GP to a given set of hyperparameters, computing the
    negative log marginal likelihood plus prior (so negative log posterior) and its
    derivatives, and making predictions on unseen data. Note that routines to
    estimate hyperparameters are not included in the class definition, and are instead
    provided externally to facilitate implementation of high performance versions.

    The required arguments to initialize a GP is a set of training data consisting
    of the inputs and targets for each of those inputs. These must be numpy arrays
    whose first axis is of the same length. Targets must be a 1D array, while inputs
    can be 2D or 1D. If inputs is 1D, then it is assumed that the length of the second
    axis is unity.

    Optional arguments are the particular mean function to use (default is zero mean),
    the covariance kernel to use (default is the squared exponential covariance),
    a list of prior distributions for each hyperparameter (default is no prior information
    on any hyperparameters) and the method for handling the nugget parameter.
    The nugget is additional "noise" that is added to the diagonal of the covariance
    kernel as a variance. This nugget can represent uncertainty in the target values
    themselves, or simply be used to stabilize the numerical inversion of the covariance
    matrix. The nugget can be fixed (a non-negative float), can be found adaptively
    (by passing the string ``"adaptive"`` to make the noise only as large as necessary
    to successfully invert the covariance matrix), or can be fit as a hyperparameter
    (by passing the string ``"fit"``).

    The internal emulator structure involves arrays for the inputs, targets, and hyperparameters.
    Other useful information are the number of training examples ``n``, the number of input
    parameters ``D``, and the number of hyperparameters ``n_params``. These parameters can
    be obtained externally by accessing these attributes

    Example: ::

        >>> import numpy as np
        >>> from mogp_emulator import GaussianProcess
        >>> x = np.array([[1., 2., 3.], [4., 5., 6.]])
        >>> y = np.array([4., 6.])
        >>> gp = GaussianProcess(x, y)
        >>> print(gp)
        Gaussian Process with 2 training examples and 3 input variables
        >>> gp.n
        2
        >>> gp.D
        3
        >>> gp.n_params
        5
        >>> gp.fit(np.zeros(gp.n_params))
        >>> x_predict = np.array([[2., 3., 4.], [7., 8., 9.]])
        >>> gp.predict(x_predict)
        (array([4.74687618, 6.84934016]), array([0.01639298, 1.05374973]),
        array([[8.91363045e-05, 7.18827798e-01, 3.74439445e-16],
               [4.64005897e-06, 3.74191346e-02, 1.94917337e-17]]))

    """
    def __init__(self, inputs, targets, mean=None, kernel=SquaredExponential(), priors=None,
                 nugget="adaptive", inputdict = {}, use_patsy=True):
        """
        Create a new GaussianProcess Emulator

        Creates a new GaussianProcess Emulator from either the input data and targets to be fit and
        optionally a mean function, covariance kernel, and nugget parameter/method.

        Required arguments are numpy arrays ``inputs`` and ``targets``, described below.
        Additional arguments are ``mean`` to specify the mean function (default is
        ``None`` for zero mean), ``kernel`` to specify the covariance kernel (default
        is the squared exponential kernel), ``priors`` to indicate prior distributions
        on the hyperparameters, and ``nugget`` to specify how the nugget
        parameter is handled (see below; default is to fit the nugget adaptively).

        ``inputs`` is a 2D array-like object holding the input data, whose shape is
        ``n`` by ``D``, where ``n`` is the number of training examples to be fit and ``D``
        is the number of input variables to each simulation. If ``inputs`` is a 1D array,
        it is assumed that ``D = 1``.

        ``targets`` is the target data to be fit by the emulator, also held in an array-like
        object. This must be a 1D array of length ``n``.

        ``prior`` must be a list of length ``n_params`` whose elements are either ``Prior``-derived
        objects or ``None``.  Each element is used as the prior for the corresponding parameter (with
        ``None`` indicating an uninformative prior).  Passing the empty list or ``None`` as this
        argument (in its entirety) may be used as an abbreviation for a list of ``n_params`` where
        all list elements are ``None``.

        ``nugget`` controls how additional noise is added to the emulator targets when fitting.
        This can be specified in several ways. If a string is provided, it can take the
        values of ``"adaptive"`` or ``"fit"``, which indicate that the nugget will be
        chosen in the fitting process. ``"adaptive"`` means that the nugget will be made only
        as large as necessary to invert the covariance matrix, while ``"fit"`` means that
        the nugget will be treated as a hyperparameter to be optimized. Alternatively,
        a non-negative float can be used to specify a fixed noise level. If no value is
        specified for the ``nugget`` parameter, ``"adaptive"`` is the default.


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
                       Can provide either a ``Kernel`` object or a string matching the
                       kernel type to be used.
        :type kernel: Kernel or str
        :param priors: List of priors to be used. Must be None (default) or an empty list
                       (indicates uninformative priors) or list of length ``n_params``.
                       Any parameter for which you wish to specify an uninformative prior,
                       pass ``None``. Number of parameters is the number of parameters in
                       the mean function plus ``D + 2`` (one correlation length per input
                       plus a covariance scale and a nugget). If the nugget will not be fit,
                       the list can have length ``n_params - 1``.
        :type priors: list or None
        :param nugget: Noise to be added to the diagonal, specified as a string or a float.
                       A non-negative float specifies the noise level explicitly, while a string
                       indicates that the nugget will be found via fitting, either as ``"adaptive"``
                       or ``"fit"`` (see above for a description). Default is ``"adaptive"``.
        :type nugget: float or str
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

        if isinstance(kernel, str):
            if kernel == "SquaredExponential":
                kernel = SquaredExponential()
            elif kernel == "Matern52":
                kernel = Matern52()
            else:
                raise ValueError("provided kernel '{}' not a supported kernel type".format(kernel))
        if not issubclass(type(kernel), Kernel):
            raise ValueError("provided kernel is not a subclass of Kernel")

        self._inputs = inputs
        self._targets = targets

        if not issubclass(type(mean), MeanBase):
            self.mean = MeanFunction(mean, inputdict, use_patsy)
        else:
            self.mean = mean

        self.kernel = kernel

        self.nugget = nugget
<<<<<<< .merge_file_F65fzN
        
        self.kernel =  SquaredExponential()
        
        if not (emulator_file is None or theta is None):
            self._set_params(theta)
        else:
            self.theta = None
            
        self.mle_theta = None
        self.samples = None

    def _load_emulator(self, filename):
        """
        Load saved emulator and parameter values from file
        
        Method takes the filename of a saved emulator (using the ``save_emulator`` method).
        The saved emulator may or may not contain the fitted parameters. If there are no
        parameters found in the emulator file, the method returns ``None`` for the
        parameters.
        
        :param filename: File where the emulator parameters are saved. Can be a string
                         filename or a file object.
        :type filename: str or file
        :returns: inputs, targets, and (optionally) fitted parameter values from the
                  saved emulator file
        :rtype: tuple containing 3 ndarrays and a float or 2 ndarrays, a None type, and
                a float (if no theta values are found in the emulator file)
        """
        
        emulator_file = np.load(filename, allow_pickle=True)
        
        try:
            inputs = np.array(emulator_file['inputs'])
            targets = np.array(emulator_file['targets'])
        except KeyError:
            raise KeyError("Emulator file does not contain emulator inputs and targets")
            
        try:
            if emulator_file['theta'] is None:
                theta = None
            else:
                theta = np.array(emulator_file['theta'])
        except KeyError:
            theta = None
            
        try:
            if emulator_file['nugget'] == None:
                nugget = None
            else:
                nugget = float(emulator_file['nugget'])
        except KeyError:
            nugget = None
            
        return inputs, targets, theta, nugget

    def save_emulator(self, filename):
        """
        Write emulators to disk
        
        Method saves the emulator to disk using the given filename or file handle. The method
        writes the inputs and targets arrays to file. If the model has been assigned parameters,
        either manually or by fitting, those parameters are saved as well. Once saved, the
        emulator can be read by passing the file name or handle to the one-argument ``__init__``
        method.
        
        :param filename: Name of file (or file handle) to which the emulator will be saved.
        :type filename: str or file
        :returns: None
        """
        
        emulator_dict = {}
        emulator_dict['targets'] = self.targets
        emulator_dict['inputs'] = self.inputs
        emulator_dict['nugget'] = self.nugget
        emulator_dict['theta'] = self.theta
                    
        np.savez(filename, **emulator_dict)
=======

        self.priors = priors

        self._theta = None

    @property
    def inputs(self):
        """
        Returns inputs for the emulator as a numpy array

        :returns: Emulator inputs, 2D array with shape ``(n, D)``
        :rtype: ndarray
        """
        return self._inputs

    @property
    def targets(self):
        """
        Returns targets for the emulator as a numpy array
>>>>>>> .merge_file_wR9ZhR

        :returns: Emulator targets, 1D array with shape ``(n,)``
        :rtype: ndarray
        """
        return self._targets

    @property
    def n(self):
        """
        Returns number of training examples for the emulator

        :returns: Number of training examples for the emulator object
        :rtype: int
        """

        return self.inputs.shape[0]

    @property
    def D(self):
        """
        Returns number of inputs for the emulator

        :returns: Number of inputs for the emulator object
        :rtype: int
        """

        return self.inputs.shape[1]

    @property
    def n_params(self):
        """
<<<<<<< .merge_file_F65fzN
        
        return self.theta
            
    def get_nugget(self):
=======
        Returns number of hyperparameters

        Returns the number of hyperparameters for the emulator. The number depends on the
        choice of mean function, covariance function, and nugget strategy, and possibly the
        number of inputs for certain choices of the mean function.

        :returns: Number of hyperparameters
        :rtype: int
        """

        return self.mean.get_n_params(self.inputs) + self.D + 2

    @property
    def nugget_type(self):
        """
        Returns method used to select nugget parameter

        Returns a string indicating how the nugget parameter is treated, either ``"adaptive"``,
        ``"fit"``, or ``"fixed"``. This is automatically set when changing the ``nugget``
        property.

        :returns: Current nugget fitting method
        :rtype: str
        """
        return self._nugget_type

    @property
    def nugget(self):
>>>>>>> .merge_file_wR9ZhR
        """
        Returns emulator nugget parameter

        Returns current value of the nugget parameter. If the nugget is to be selected
        adaptively or by fitting the emulator and the nugget has not been fit,
        returns ``None``.

        :returns: Current nugget value, either a float or ``None``
        :rtype: float or None

        The ``nugget`` parameter controls how noise is added to the covariance matrix in order to
        stabilize the inversion or smooth the emulator predictions. If ``nugget`` is a non-negative
        float, then that particular value is used for the nugget. Note that setting this parameter
        to be zero enforces that the emulator strictly interpolates between points. Alternatively,
        a string can be provided. A value of ``"fit"`` means that the nugget is treated as a
        hyperparameter, and is the last entry in the ``theta`` array. Alternatively, if ``nugget``
        is set to be ``"adaptive"``, the fitting routine will adaptively make the noise parameter
        as large as is needed to ensure that the emulator can be fit.

        Internally, this modifies both the way the nugget is chosen (which can be determined
        via the ``nugget_type`` property) and the value itself (the ``nugget`` property)

        :param nugget: Noise to be added to the diagonal, specified as a string or a float.
                       A non-negative float specifies the noise level explicitly, while a string
                       indicates that the nugget will be found via fitting, either as ``"adaptive"``
                       or ``"fit"`` (see above for a description).
        :type nugget: float or str
        :returns: None
        :rtype: None
        """

        return self._nugget

    @nugget.setter
    def nugget(self, nugget):
        """
        Set the nugget parameter for the emulator

        Method for changing the ``nugget`` parameter for the emulator. When a new emulator is
        initilized, this is set to None.

        The ``nugget`` parameter controls how noise is added to the covariance matrix in order to
        stabilize the inversion or smooth the emulator predictions. If ``nugget`` is a non-negative
        float, then that particular value is used for the nugget. Note that setting this parameter
        to be zero enforces that the emulator strictly interpolates between points. Alternatively,
        a string can be provided. A value of ``"fit"`` means that the nugget is treated as a
        hyperparameter, and is the last entry in the ``theta`` array. Alternatively, if ``nugget``
        is set to be ``"adaptive"``, the fitting routine will adaptively make the noise parameter
        as large as is needed to ensure that the emulator can be fit.

        Internally, this modifies both the way the nugget is chosen (which can be determined
        via the ``nugget_type`` property) and the value itself (the ``nugget`` property)

        :param nugget: Noise to be added to the diagonal, specified as a string or a float.
                       A non-negative float specifies the noise level explicitly, while a string
                       indicates that the nugget will be found via fitting, either as ``"adaptive"``
                       or ``"fit"`` (see above for a description).
        :type nugget: float or str
        :returns: None
        :rtype: None
        """

        if not isinstance(nugget, (str, float)):
            try:
                nugget = float(nugget)
            except TypeError:
                raise TypeError("nugget parameter must be a string or a non-negative float")

        if isinstance(nugget, str):
            if nugget == "adaptive":
                self._nugget_type = "adaptive"
            elif nugget == "fit":
                self._nugget_type = "fit"
            else:
                raise ValueError("bad value of nugget, must be a float or 'adaptive' or 'fit'")
            self._nugget = None
        else:
            if nugget < 0.:
                raise ValueError("nugget parameter must be non-negative")
            self._nugget_type = "fixed"
            self._nugget = float(nugget)

    @property
    def theta(self):
        """
        Returns emulator hyperparameters

        Returns current hyperparameters for the emulator as a numpy array if they have been fit.
        If no parameters have been fit, returns ``None``. Note that the number of parameters
        depends on the mean function, so the length of this array will vary across instances.

        :returns: Current parameter values (numpy array of length ``n_params``), or ``None`` if the
                  parameters have not been fit.
        :rtype: ndarray or None

        When set, pre-calculates the matrices needed to compute the log-likelihood and its derivatives
        and make subsequent predictions. This is called any time the hyperparameter values are
        changed in order to ensure that all the information is needed to evaluate the
        log-likelihood and its derivatives, which are needed when fitting the optimal
        hyperparameters.

        The method computes the mean function and covariance matrix and inverts the covariance
        matrix using the method specified by the value of ``nugget``. The factorized matrix
        and the product of the inverse with the difference between the targets and the mean
        are cached for later use, and the negative marginal log-likelihood is also cached.
        This method has no return value, but it does modify the state of the object.

        :param theta: Values of the hyperparameters to use in fitting. Must be a numpy
                      array with length ``n_params``
        :type theta: ndarray
        """

        return self._theta

    @theta.setter
    def theta(self, theta):
        """
        Fits the emulator and sets the parameters (property-based setter alias for ``fit``)

        Pre-calculates the matrices needed to compute the log-likelihood and its derivatives
        and make subsequent predictions. This is called any time the hyperparameter values are
        changed in order to ensure that all the information is needed to evaluate the
        log-likelihood and its derivatives, which are needed when fitting the optimal
        hyperparameters.

        The method computes the mean function and covariance matrix and inverts the covariance
        matrix using the method specified by the value of ``nugget``. The factorized matrix
        and the product of the inverse with the difference between the targets and the mean
        are cached for later use, and the negative marginal log-likelihood is also cached.
        This method has no return value, but it does modify the state of the object.

        :param theta: Values of the hyperparameters to use in fitting. Must be a numpy
                      array with length ``n_params``
        :type theta: ndarray
        :returns: None
        """
<<<<<<< .merge_file_F65fzN
        
        assert not self.theta is None, "Must set a parameter value to fit a GP"

        self.Q = self.kernel.kernel_f(self.inputs, self.inputs, self.theta)
        
        if self.nugget == None:
            L, nugget = self._jit_cholesky(self.Q)
            self.Z = self.Q + nugget*np.eye(self.n)
        else:
            self.Z = self.Q + self.nugget*np.eye(self.n)
            L = linalg.cholesky(self.Z, lower=True)
        
        self.invQ = np.linalg.inv(L.T).dot(np.linalg.inv(L))
        self.invQt = np.dot(self.invQ, self.targets)
        self.logdetQ = 2.0 * np.sum(np.log(np.diag(L)))
        
    def _set_params(self, theta):
        """
        Method for setting the hyperparameters for the emulator
        
        This method is used to reset the value of the hyperparameters for the emulator and
        update the log-likelihood. It is used after fitting the hyperparameters or when loading
        an emulator from file. Input ``theta`` must be array-like with shape
        ``(D + 1,)``, where ``D`` is the number of input parameters.
        
        :param theta: Parameter values to be used for the emulator. Must be array-like and
                      have shape ``(D + 1,)``
=======
        self.fit(theta)

    @property
    def priors(self):
        """
        The current list priors used in computing the log posterior

        To set the priors, must be a list or ``None``. Entries can be ``None`` or a subclass of ``Prior``.
        ``None`` indicates weak prior information. An empty list or ``None`` means all uninformative
        priors. Otherwise list should have the same length as the number of hyperparameters,
        or alternatively can be one shorter than the number of hyperparameters
        if ``nugget_type`` is ``"adaptive"`` or ``"fixed"`` meaning that the nugget hyperparameter
        is not fit but is instead fixed or found adaptively. If the nugget hyperparameter is not fit,
        the prior for the nugget will automatically be set to ``None`` even if a distribution is
        provided.
        """
        return self._priors

    @priors.setter
    def priors(self, priors):
        """
        Sets the priors to a list of prior objects/None

        Sets the priors, must be a list or ``None``. Entries can be ``None`` or a subclass of ``Prior``.
        ``None`` indicates weak prior information. An empty list or ``None`` means all uninformative
        priors. Otherwise list should have the same length as the number of hyperparameters,
        or alternatively can be one shorter than the number of hyperparameters
        if ``nugget_type`` is ``"adaptive"`` or ``"fixed"`` meaning that the nugget hyperparameter
        is not fit but is instead fixed or found adaptively. If the nugget hyperparameter is not fit,
        the prior for the nugget will automatically be set to ``None`` even if a distribution is
        provided.
        """

        if priors is None:
            priors = []

        if not isinstance(priors, list):
            raise TypeError("priors must be a list of Prior-derived objects")

        if len(priors) == 0:
            priors = self.n_params*[None]

        if self.nugget_type in ["adaptive", "fixed"]:
            if len(priors) == self.n_params - 1:
                priors.append(None)

        if not len(priors) == self.n_params:
            raise ValueError("bad length for priors; must have length n_params")

        if self.nugget_type in ["adaptive", "fixed"]:
            if not priors[-1] is None:
                priors[-1] = None

        for p in priors:
            if not p is None and not issubclass(type(p), Prior):
                raise TypeError("priors must be a list of Prior-derived objects")

        self._priors = list(priors)


    def get_K_matrix(self):
        """
        Returns current value of the covariance matrix as a numpy array. Does not include the nugget
        parameter, as this is dependent on how the nugget is fit.
        """
        switch = self.mean.get_n_params(self.inputs)

        return self.kernel.kernel_f(self.inputs, self.inputs, self.theta[switch:-1])

    def fit(self, theta):
        """
        Fits the emulator and sets the parameters

        Pre-calculates the matrices needed to compute the log-likelihood and its derivatives
        and make subsequent predictions. This is called any time the hyperparameter values are
        changed in order to ensure that all the information is needed to evaluate the
        log-likelihood and its derivatives, which are needed when fitting the optimal
        hyperparameters.

        The method computes the mean function and covariance matrix and inverts the covariance
        matrix using the method specified by the value of ``nugget_type``. The factorized matrix
        and the product of the inverse with the difference between the targets and the mean
        are cached for later use, and the negative marginal log-likelihood is also cached.
        This method has no return value, but it does modify the state of the object.

        :param theta: Values of the hyperparameters to use in fitting. Must be a numpy
                      array with length ``n_params``
>>>>>>> .merge_file_wR9ZhR
        :type theta: ndarray
        :returns: None
        """

        theta = np.array(theta)

        assert theta.shape == (self.n_params,), "bad shape for hyperparameters"

        self._theta = theta

        switch = self.mean.get_n_params(self.inputs)

        m = self.mean.mean_f(self.inputs, self.theta[:switch])
        Q = self.kernel.kernel_f(self.inputs, self.inputs, self.theta[switch:-1])

        if self.nugget_type == "adaptive":
            self.L, self._nugget = jit_cholesky(Q)
        else:
            if self.nugget_type == "fit":
                self._nugget = np.exp(self.theta[-1])
            Q += self._nugget*np.eye(self.n)
            self.L = linalg.cholesky(Q, lower=True)

        self.invQt = linalg.cho_solve((self.L, True), self.targets - m)

        self.current_logpost = 0.5*(2.0*np.sum(np.log(np.diag(self.L))) +
                                    np.dot(self.targets - m, self.invQt) +
                                    self.n*np.log(2. * np.pi))

        for i in range(self.n_params):
            if not self._priors[i] is None:
                self.current_logpost -= self._priors[i].logp(self.theta[i])


    def logposterior(self, theta):
        """
        Calculate the negative log-posterior at a particular value of the hyperparameters

        Calculate the negative log-posterior for the given set of parameters. Calling this
        method sets the parameter values and computes the needed inverse matrices in order
        to evaluate the log-posterior and its derivatives. In addition to returning the
        log-posterior value, it stores the current value of the hyperparameters and
        log-posterior in attributes of the object.

        :param theta: Value of the hyperparameters. Must be array-like with shape ``(n_params,)``
        :type theta: ndarray
        :returns: negative log-posterior
        :rtype: float
        """
<<<<<<< .merge_file_F65fzN
        
        self._set_params(theta)

        loglikelihood = (0.5 * self.logdetQ +
                         0.5 * np.dot(self.targets, self.invQt) +
                         0.5 * self.n * np.log(2. * np.pi))

        return loglikelihood
    
    def partial_devs(self, theta):
        """
        Calculate the partial derivatives of the negative log-likelihood
        
        Calculate the partial derivatives of the negative log-likelihood with respect to
=======

        if self.theta is None or not np.allclose(theta, self.theta, rtol=1.e-10, atol=1.e-15):
            self.fit(theta)

        return self.current_logpost

    def logpost_deriv(self, theta):
        """
        Calculate the partial derivatives of the negative log-posterior

        Calculate the partial derivatives of the negative log-posterior with respect to
>>>>>>> .merge_file_wR9ZhR
        the hyperparameters. Note that this function is normally used only when fitting
        the hyperparameters, and it is not needed to make predictions.

        During normal use, the ``loglike_deriv`` method is called after evaluating the
        ``logposterior`` method. The implementation takes advantage of this by reusing
        cached results, as the factorized covariance matrix is expensive to compute and is
        used by the ``logposterior``, ``logpost_deriv``, and ``logpost_hessian`` methods.
        If the function is evaluated with a different set of parameters than was previously
        used to set the log-posterior, the method calls ``fit`` (and subsequently resets
        the cached information).

        :param theta: Value of the hyperparameters. Must be array-like with shape
                      ``(n_params,)``
        :type theta: ndarray
        :returns: partial derivatives of the negative log-posterior with respect to the
                  hyperparameters (array with shape ``(n_params,)``)
        :rtype: ndarray
        """

<<<<<<< .merge_file_F65fzN
        assert theta.shape == (self.D + 1,), "Parameter vector must have length number of inputs + 1"
        
        if not np.allclose(np.array(theta), self.theta):
            self._set_params(theta)
            
        partials = np.zeros(self.D + 1)
        
        dKdtheta = self.kernel.kernel_deriv(self.inputs, self.inputs, self.theta)
        
=======
        theta = np.array(theta)

        assert theta.shape == (self.n_params,), "bad shape for new parameters"

        if self.theta is None or not np.allclose(theta, self.theta, rtol=1.e-10, atol=1.e-15):
            self.fit(theta)

        partials = np.zeros(self.n_params)

        switch = self.mean.get_n_params(self.inputs)

        dmdtheta = self.mean.mean_deriv(self.inputs, self.theta[:switch])
        dKdtheta = self.kernel.kernel_deriv(self.inputs, self.inputs, self.theta[switch:-1])

        partials[:switch] = -np.dot(dmdtheta, self.invQt)

>>>>>>> .merge_file_wR9ZhR
        for d in range(self.D + 1):
            invQ_dot_dKdtheta_trace = np.trace(linalg.cho_solve((self.L, True), dKdtheta[d]))
            partials[switch + d] = -0.5*(np.dot(self.invQt, np.dot(dKdtheta[d], self.invQt)) -
                                         invQ_dot_dKdtheta_trace)

        if self.nugget_type == "fit":
            nugget = np.exp(self.theta[-1])
            partials[-1] = 0.5*nugget*(np.trace(linalg.cho_solve((self.L, True), np.eye(self.n))) -
                                       np.dot(self.invQt, self.invQt))

        for i in range(self.n_params):
            if not self._priors[i] is None:
                partials[i] -= self._priors[i].dlogpdtheta(self.theta[i])

        return partials

    def logpost_hessian(self, theta):
        """
        Calculate the Hessian of the negative log-posterior

        Calculate the Hessian of the negative log-posterior with respect to
        the hyperparameters. Note that this function is normally used only when fitting
        the hyperparameters, and it is not needed to make predictions. It is also used
        to estimate an appropriate step size when fitting hyperparameters using
        the lognormal approximation or MCMC sampling.

        When used in an optimization routine, the ``logpost_hessian`` method is called after
        evaluating the ``logposterior`` method. The implementation takes advantage of
        this by storing the inverse of the covariance matrix, which is expensive to
        compute and is used by the ``logposterior`` and ``logpost_deriv`` methods as well.
        If the function is evaluated with a different set of parameters than was previously
        used to set the log-posterior, the method calls ``fit`` to compute the needed
        information and changes the cached values.

        :param theta: Value of the hyperparameters. Must be array-like with shape
                      ``(n_params,)``
        :type theta: ndarray
        :returns: Hessian of the negative log-posterior (array with shape
                  ``(n_params, n_params)``)
        :rtype: ndarray
        """

<<<<<<< .merge_file_F65fzN
        assert theta.shape == (self.D + 1,), "Parameter vector must have length number of inputs + 1"
        
        if not np.allclose(np.array(theta), self.theta):
            self._set_params(theta)
            
        hessian = np.zeros((self.D + 1, self.D + 1))
        
        dKdtheta = self.kernel.kernel_deriv(self.inputs, self.inputs, self.theta)
        d2Kdtheta2 = self.kernel.kernel_hessian(self.inputs, self.inputs, self.theta)
        
=======
        assert theta.shape == (self.n_params,), "Parameter vector must have length number of inputs + 1"

        if self.theta is None or not np.allclose(theta, self.theta, rtol=1.e-10, atol=1.e-15):
            self.fit(theta)

        hessian = np.zeros((self.n_params, self.n_params))

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

>>>>>>> .merge_file_wR9ZhR
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

        if self.nugget_type == "fit":
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

        for i in range(self.n_params):
            if not self._priors[i] is None:
                hessian[i, i] -= self._priors[i].d2logpdtheta2(self.theta[i])

        return hessian
<<<<<<< .merge_file_F65fzN
    
    def _learn(self, theta0, method = 'L-BFGS-B', **kwargs):
        """
        Minimize log-likelihood function wrt the hyperparameters
        
        Minimize the negative log-likelihood function, with a starting value given by ``theta0``.
        This is done via any gradient-based method available through scipy (see the scipy
        documentation for details), and options can be passed to the minimization routine.
        The default minimization routine is ``L-BFGS-B'``, but this can be specified.
        
        The method is dumb and returns the last value returned from the minimization routine
        irrespective of any error flags returned by the minimization function. This is not
        necessarily a cause for concern, as (1) the parent routine to this function is
        configured to do a certain number of attempts, taking the best result and (2) application
        of the GP does not require that the hyperparameters are at a true minimum of the
        log-likelihood function, just at a value that leads to predictions that are good enough.
        
        The method returns the hyperparameter values as an array with shape ``(D + 1,)`` 
        and the minimimum negative log-likelihood value found.
        
        :param theta0: Starting value for the minimization routine. Must be an array with shape
                       ``(D + 1,)``
        :type theta0: ndarray
        :param method: Minimization method. Must be a gradient-based method available in the
                       ``scipy.optimize.minimize`` function (optional, default is ``'L-BFGS-B'``
                       with no bounds given)
        :type method: str
        :param **kwargs: Additional keyword arguments to be passed to ``scipy.optimize.minimize``
        :returns: minimum hyperparameter values in an array of shape ``(D + 1,)`` and the
                  minimum negative log-likelihood value
        :rtype: tuple containing a ndarray and a float
        """
        
        self._set_params(theta0)
        
        fmin_dict = minimize(self.loglikelihood, theta0, method = method, jac = self.partial_devs, 
                             options = kwargs)
        
        return fmin_dict['x'], fmin_dict['fun']
    
    def learn_hyperparameters(self, n_tries = 15, theta0 = None, method = 'L-BFGS-B', **kwargs):
        """
        Fit hyperparameters by attempting to minimize the negative log-likelihood
        
        Fits the hyperparameters by attempting to minimize the negative log-likelihood multiple times
        from a given starting location and using a particular minimization method. The best result
        found among all of the attempts is returned, unless all attempts to fit the parameters result
        in an error (see below).
        
        If the method encounters an overflow (this can result because the parameter values stored are
        the logarithm of the actual hyperparameters to enforce positivity) or a linear algebra error
        (occurs when the covariance matrix cannot be inverted, even with the addition of additional
        noise added along the diagonal if adaptive noise was selected by setting the nugget parameter
        to be None), the iteration is skipped. If all attempts to find optimal hyperparameters result
        in an error, then the method raises an exception.
        
        The ``theta0`` parameter is the point at which the first iteration will start. If more than
        one attempt is made, subsequent attempts will use random starting points.
        
        The user can specify the details of the minimization method, using any of the gradient-based
        optimizers available in ``scipy.optimize.minimize``. Any additional parameters beyond the method
        specification can be passed as keyword arguments.
        
        The method returns the minimum negative log-likelihood found and the parameter values at
        which that minimum was obtained. The method also sets the current values of the hyperparameters
        to these optimal values and pre-computes the matrices needed to make predictions.
        
        :param n_tries: Number of attempts to minimize the negative log-likelihood function.
                        Must be a positive integer (optional, default is 15)
        :type n_tries: int
        :param theta0: Initial starting point for the first iteration. If present, must be
                       array-like with shape ``(D + 1,)``. If ``None`` is given, then
                       a random value is chosen. (Default is ``None``)
        :type theta0: None or ndarray
        :param method: Minimization method to be used. Can be any gradient-based optimization
                       method available in ``scipy.optimize.minimize``. (Default is ``'L-BFGS-B'``)
        :type method: str
        :param ``**kwargs``: Additional keyword arguments to be passed to the minimization routine.
                         see available parameters in ``scipy.optimize.minimize`` for details.
        :returns: Minimum negative log-likelihood values and hyperparameters (numpy array with shape
                  ``(D + 1,)``) used to obtain those values. The method also sets the current values
                  of the hyperparameters to these optimal values and pre-computes the matrices needed
                  to make predictions.
        :rtype: tuple containing a float and an ndarray
        """
    
        n_tries = int(n_tries)
        assert n_tries > 0, "number of attempts must be positive"
        
        np.seterr(divide = 'raise', over = 'raise', invalid = 'raise')
        
        loglikelihood_values = []
        theta_values = []
        
        theta_startvals = 5.*(np.random.rand(n_tries, self.D + 1) - 0.5)
        if not theta0 is None:
            theta0 = np.array(theta0)
            assert theta0.shape == (self.D + 1,), "theta0 must be a 1D array with length D + 1"
            theta_startvals[0,:] = theta0

        for theta in theta_startvals:
            try:
                min_theta, min_loglikelihood = self._learn(theta, method, **kwargs)
                loglikelihood_values.append(min_loglikelihood)
                theta_values.append(min_theta)
            except linalg.LinAlgError:
                print("Matrix not positive definite, skipping this iteration")
            except FloatingPointError:
                print("Floating point error in optimization routine, skipping this iteration")
                
        if len(loglikelihood_values) == 0:
            raise RuntimeError("Minimization routine failed to return a value")
            
        loglikelihood_values = np.array(loglikelihood_values)
        idx = np.argmin(loglikelihood_values)
        
        self._set_params(theta_values[idx])
        self.mle_theta = theta_values[idx]
        
        return loglikelihood_values[idx], theta_values[idx]
    
    def compute_local_covariance(self):
        """
        Estimate local covariance matrix around the MLE parameters
        
        This method inverts the hessian matrix to get the local covariance matrix around the
        MLE parameters. Note that if the MLE parameters have not been estimated, they will be
        found first prior to inverting the Hessian. The local Hessian should be positive definite
        if the MLE parameters are at a local minimum of the negative log-likelihood, so if the
        routine encounters a non-positive definite matrix it will raise an error. Returns the
        inverse of the Hessian matrix evaluated at the MLE parameter values.
        
        :returns: Inverse of the Hessian matrix evaluated at the MLE parameter values. This is
                  a 2D array with shape ``(D + 1, D + 1)``.
        :rtype: ndarray
        """

        if self.mle_theta is None:
            self.learn_hyperparameters()

        hess = self.hessian(self.mle_theta)
    
        assert hess.ndim == 2
        assert hess.shape[0] == hess.shape[1]
    
        try:
            L = np.linalg.cholesky(hess)
            cov = np.linalg.inv(L.T).dot(np.linalg.inv(L))
        except linalg.LinAlgError:
            raise linalg.LinAlgError("Hessian matrix is not symmetric positive definite, optimization may not have converged")
        
        return cov
    
    def learn_hyperparameters_MLE(self, n_tries = 15, theta0 = None, method = 'L-BFGS-B', **kwargs):
        """
        Fit hyperparameters by attempting to minimize the negative log-likelihood
        
        This method an alias for ``learn_hyperparameters`` to distinguish it from other methods
        for estimating hyperparameters.
        
        Fits the hyperparameters by attempting to minimize the negative log-likelihood multiple times
        from a given starting location and using a particular minimization method. The best result
        found among all of the attempts is returned, unless all attempts to fit the parameters result
        in an error (see below).
        
        If the method encounters an overflow (this can result because the parameter values stored are
        the logarithm of the actual hyperparameters to enforce positivity) or a linear algebra error
        (occurs when the covariance matrix cannot be inverted, even with the addition of additional
        noise added along the diagonal if adaptive noise was selected by setting the nugget parameter
        to be None), the iteration is skipped. If all attempts to find optimal hyperparameters result
        in an error, then the method raises an exception.
        
        The ``theta0`` parameter is the point at which the first iteration will start. If more than
        one attempt is made, subsequent attempts will use random starting points.
        
        The user can specify the details of the minimization method, using any of the gradient-based
        optimizers available in ``scipy.optimize.minimize``. Any additional parameters beyond the method
        specification can be passed as keyword arguments.
        
        The method returns the minimum negative log-likelihood found and the parameter values at
        which that minimum was obtained. The method also sets the current values of the hyperparameters
        to these optimal values and pre-computes the matrices needed to make predictions.
        
        :param n_tries: Number of attempts to minimize the negative log-likelihood function.
                        Must be a positive integer (optional, default is 15)
        :type n_tries: int
        :param theta0: Initial starting point for the first iteration. If present, must be
                       array-like with shape ``(D + 1,)``. If ``None`` is given, then
                       a random value is chosen. (Default is ``None``)
        :type theta0: None or ndarray
        :param method: Minimization method to be used. Can be any gradient-based optimization
                       method available in ``scipy.optimize.minimize``. (Default is ``'L-BFGS-B'``)
        :type method: str
        :param ``**kwargs``: Additional keyword arguments to be passed to the minimization routine.
                         see available parameters in ``scipy.optimize.minimize`` for details.
        :returns: Minimum negative log-likelihood values and hyperparameters (numpy array with shape
                  ``(D + 1,)``) used to obtain those values. The method also sets the current values
                  of the hyperparameters to these optimal values and pre-computes the matrices needed
                  to make predictions.
        :rtype: tuple containing a float and an ndarray
        """
        
        return self.learn_hyperparameters(n_tries, theta0, method, **kwargs)
    
    def learn_hyperparameters_normalapprox(self, n_samples = 1000):
        """
        Sample hyperparameters via a normal approximation around MLE solution
        
        Sample hyperparameters via a multivariate normal approximation around the MLE parameters.
        This method first obtains an MLE estimate of the hyperparameters, and then draws
        samples assuming the posterior follows an approximate normal distribution around the MLE
        value. This is a reasonable approximation for most unimodal posterior distributions,
        and it is computationally much cheaper than using full MCMC estimation. The local
        covariance matrix is found by inverting the Hessian around the MLE parameters, and then
        samples are generated using a multivariate normal distribution. Does not return a value,
        but sets the ``samples`` class attribute to a 2D array with shape ``(n_samples, D + 1)``,
        where the first dimension indicates the different samples and the second dimension specifies
        the different hyperparameters.
        
        :param n_samples: Number of samples to be drawn. Must be a positive integer.
        :type n_samples: int
        :returns: None
        """
        
        n_samples = int(n_samples)
        assert n_samples > 0
    
        n_params = self.D + 1
        
        if self.mle_theta is None:
            self.learn_hyperparameters()
        
        cov = self.compute_local_covariance()
    
        self.samples = np.random.multivariate_normal(self.mle_theta, cov, size=n_samples)
        
    def learn_hyperparameters_MCMC(self, n_samples = 1000, thin = 0):
        """
        Sample hyperparameters via MCMC estimation
        
        Sample hyperparameters via MCMC estimation. Parameters are found by doing a random
        walk in parameter space, choosing new points via the Metropolis-Hastings algorithm.
        Steps are drawn from a multivariate normal distribution around the current parameters,
        and the steps are accepted and rejected based on the marginal log-likelihood function.
        The chain is started from the MLE parameter values, and the step sizes are estimated
        by inverting the local Hessian at the MLE solution. Because of this, the MCMC chain
        does not require a "burn-in" phase. Optional parameters specify the number of MCMC
        steps to take (must be a positive integer, default is 1000) and information about
        how to thin the MCMC chain to obtain uncorrelated samples.
        
        Thinning may be specified with a non-negative integer. If a positive integer is
        given, the chain will be thinned by only keeping every ``thin`` steps. Note that
        ``thin = 1`` means that the chain will not be thinned. If ``thin = 0`` is given
        (the default value), the chain will automatically be thinned by computing the
        autocorrelation of the chain for each parameter separately and estimating the value
        needed to eliminate correlations in the chain. If the autothinning method fails
        (usually occurrs if the posterior is multimodal), the chain will not be thinned
        and a warning will be given. More details on the autothinning procedure are
        described in the corresponding function.
        
        Does not return a value, but sets the ``samples`` class attribute to a 2D array with shape
        ``(n_chain, D + 1)``, where the first dimension indicates the different samples and the
        second dimension specifies the different hyperparameters. Note that ``n_chain``
        will only be the same as ``n_samples`` if ``thin = 1`` is specified or if
        autothinning fails. If you wish to obtain a specific number of samples in the thinned
        chain, you will need to modify ``n_samples`` and ``thin`` appropriately.
        
        Note that at present, the return information from the MCMC sampler is not returned
        or cached. The code does give a warning if a problem arises, in particular if the
        acceptance rate is not within the target range of 20% to 60% or if the final MCMC
        chain has a first lag autocorrelation that indicates samples may not be independent.
        If either of these warnings occur, the MCMC chain may require further inspection.
        At the moment, this can only be done by re-running the MCMC samples using the function
        ``sample_MCMC`` in the ``MCMC`` submodule manually.
        
        :param n_samples: Number of MCMC steps to be taken. Must be a positive integer.
        :type n_samples: int
        :param thin: Specifies how the chain is thinned to remove correlations. Must be
                     a non-negative integer. If a positive integer ``k`` is used, it will 
                     keep every ``k`` samples. Note that ``thin = 1`` indicates that the
                     chain will not be thinned. ``thin = 0`` will attempt to autothin
                     the chain using the autocorrelation of the MCMC chain. Default is 0.
        :type thin: int
        :returns: None
        """

        n_samples = int(n_samples)
        thin = int(thin)

        assert n_samples > 0
        assert thin >= 0

        n_params = self.D + 1
        
        if self.mle_theta is None:
            self.learn_hyperparameters()

        step_size = 2.4/np.sqrt(n_params)*self.compute_local_covariance()

        self.samples, rejected, acceptance, first_lag = sample_MCMC(self.loglikelihood,
                                                                    self.mle_theta, step_size,
                                                                    n_samples, thin, loglike_sign = -1.)

        if acceptance < 0.2 or acceptance > 0.6:
            warnings.warn("acceptance rate of "+str(100.*acceptance)+"% not within bounds")
            
        if np.max(first_lag) > 3./np.sqrt(len(self.samples)):
            warnings.warn("autocorrelation of "+str(np.max(first_lag))+
                          " not within bounds. posterior may be multimodal or require thinning.")

    def _predict_single(self, testing, do_deriv = True, do_unc = True):
        """
        Make a prediction for a set of input vectors for a single set of hyperparameters
        
        Note that the class provides a public ``predict`` method which calls this method for the
        appropriate case, so this should not need to be used in ordinary circumstances.
        
=======

    def predict(self, testing, unc=True, deriv=True, include_nugget=True):
        """
        Make a prediction for a set of input vectors for a single set of hyperparameters

>>>>>>> .merge_file_wR9ZhR
        Makes predictions for the emulator on a given set of input vectors. The input vectors
        must be passed as a ``(n_predict, D)``, ``(n_predict,)`` or ``(D,)`` shaped array-like
        object, where ``n_predict`` is the number of different prediction points under
        consideration and ``D`` is the number of inputs to the emulator. If the prediction
        inputs array is 1D and ``D == 1`` for the GP instance, then the 1D array must have
        shape ``(n_predict,)``. Otherwise, if the array is 1D it must have shape
        ``(D,)``, and the method assumes ``n_predict == 1``. The prediction is returned as an
        ``(n_predict, )`` shaped numpy array as the first return value from the method.
<<<<<<< .merge_file_F65fzN
        
        Optionally, the emulator can also calculate the variances in the predictions 
=======

        Optionally, the emulator can also calculate the variances in the predictions
>>>>>>> .merge_file_wR9ZhR
        and the derivatives with respect to each input parameter. If the uncertainties are
        computed, they are returned as the second output from the method as an ``(n_predict,)``
        shaped numpy array. If the derivatives are computed, they are returned as the third
        output from the method as an ``(n_predict, D)`` shaped numpy array.
<<<<<<< .merge_file_F65fzN
        
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
=======

        The final input to the method determines if the predictive variance should include
        the nugget or not. For situations where the nugget represents observational error
        and predictions are estimating the true underlying function, this should be set to
        ``False``. However, most other cases should probably include the nugget, as the
        emulator is using it to represent some of the uncertainty in the underlying simulator,
        so the default value is ``True``.

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
>>>>>>> .merge_file_wR9ZhR
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

        switch = self.mean.get_n_params(testing)
        mtest = self.mean.mean_f(testing, self.theta[:switch])
        Ktest = self.kernel.kernel_f(self.inputs, testing, self.theta[switch:-1])

        mu = mtest + np.dot(Ktest.T, self.invQt)

        var = None
<<<<<<< .merge_file_F65fzN
        if do_unc:
            var = np.maximum(exp_theta[self.D] - np.sum(Ktest * np.dot(self.invQ, Ktest), axis=0), 0.)
        
        deriv = None
        if do_deriv:
            deriv = np.zeros((n_testing, self.D))
            for d in range(self.D):
                aa = (self.inputs[:, d].flatten()[None, :] - testing[:, d].flatten()[:, None])
                c = Ktest * aa.T
                deriv[:, d] = exp_theta[d] * np.dot(c.T, self.invQt)
                
        return mu, var, deriv
    
    def _predict_samples(self, testing, do_deriv = True, do_unc = True):
        """
        Make a prediction for a set of input vectors for a set of hyperparameter posterior samples
        
        Note that the class provides a public ``predict`` method which calls this method for the
        appropriate case, so this should not need to be used in ordinary circumstances.
        
        Makes predictions for the emulator on a given set of input vectors. The input vectors
        must be passed as a ``(n_predict, D)`` or ``(D,)`` shaped array-like object, where
        ``n_predict`` is the number of different prediction points under consideration and
        ``D`` is the number of inputs to the emulator. If the prediction inputs array has shape
        ``(D,)``, then the method assumes ``n_predict == 1``. The prediction is returned as an
        ``(n_predict, )`` shaped numpy array as the first return value from the method.
        
        Optionally, the emulator can also calculate the variances in the predictions 
        and the derivatives with respect to each input parameter. If the uncertainties are
        computed, they are returned as the second output from the method as an ``(n_predict,)``
        shaped numpy array. If the derivatives are computed, they are returned as the third
        output from the method as an ``(n_predict, D)`` shaped numpy array.
        
        For this method to work, hyperparameter samples must have been drawn via the
        ``learn_hyperparameters_normalapprox`` or ``learn_hyperparameters_MCMC``
        methods. If samples have not been drawn, predictions fall back onto using the MLE
        parameters as a single set of parameters. Note that the code does not save the inverted
        covariance matrix for each set of hyperparameter samples, so these predictions can be
        expensive for large numbers of samples or large numbers of inputs as the matrix inverse
        must be computed for each hyperparameter samples. Predictions from a single set of
        parameters used the cached matrix inverse, so these predictions are much more efficient.
        
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
        :returns: Tuple of numpy arrays holding the predictions, uncertainties, and derivatives,
                  respectively. Predictions and uncertainties have shape ``(n_predict,)``
                  while the derivatives have shape ``(n_predict, D)``. If the ``do_unc`` or
                  ``do_deriv`` flags are set to ``False``, then those arrays are replaced by
                  ``None``.
        :rtype: tuple
        """
        
        testing = np.array(testing)
        if len(testing.shape) == 1:
            testing = np.reshape(testing, (1, len(testing)))
        assert len(testing.shape) == 2
                        
        n_testing, D = np.shape(testing)
        assert D == self.D
        
        if self.samples is None:
            warnings.warn("hyperparameter samples have not been drawn, trying single parameter predictions")
            return self.predict(testing, do_deriv, do_unc, predict_from_samples = False)
        
        n_samples = self.samples.shape[0]

        mu = np.zeros((n_samples, n_testing))
        var = np.zeros((n_samples, n_testing))
        deriv = np.zeros((n_samples, n_testing, self.D))

        for i in range(n_samples):
            self._set_params(self.samples[i])
            mu[i], var[i], deriv[i] = self._predict_single(testing, do_deriv, do_unc)
        
        mu_mean = np.mean(mu, axis = 0)
        if do_unc:
            var_mean = np.mean(var, axis = 0)+np.var(mu, axis = 0)
        else:
            var_mean = None
        if do_deriv:
            deriv_mean = np.mean(deriv, axis = 0)
        else:
            deriv_mean = None
        
        return mu_mean, var_mean, deriv_mean
    
    def predict(self, testing, do_deriv = True, do_unc = True, predict_from_samples = False):
        """
        Make a prediction for a set of input vectors
        
        Makes predictions for the emulator on a given set of input vectors. The input vectors
        must be passed as a ``(n_predict, D)`` or ``(D,)`` shaped array-like object, where
        ``n_predict`` is the number of different prediction points under consideration and
        ``D`` is the number of inputs to the emulator. If the prediction inputs array has shape
        ``(D,)``, then the method assumes ``n_predict == 1``. The prediction is returned as an
        ``(n_predict, )`` shaped numpy array as the first return value from the method.
        
        Optionally, the emulator can also calculate the variances in the predictions 
        and the derivatives with respect to each input parameter. If the uncertainties are
        computed, they are returned as the second output from the method as an ``(n_predict,)``
        shaped numpy array. If the derivatives are computed, they are returned as the third
        output from the method as an ``(n_predict, D)`` shaped numpy array.
        
        If predictions based on samples of the hyperparameters (drawn by either assuming a
        normal posterior or using MCMC sampling) are desired, hyperparameter samples must have
        been drawn via the ``learn_hyperparameters_normalapprox`` or ``learn_hyperparameters_MCMC``
        methods. This is controlled by setting ``predict_from_samples = True``. If samples
        have not been drawn, predictions fall back onto using the MLE parameters as a single set
        of parameters. Default behavior is to use a single set of parameters. Note that the
        code does not save the inverted covariance matrix for each set of hyperparameter
        samples, so these predictions can be expensive for large numbers of samples or
        large numbers of inputs as the matrix inverse must be computed for each hyperparameter
        samples. Predictions from a single set of parameters used the cached matrix inverse,
        so these predictions are much more efficient.
        
        If predictions from a single set of parameters are desired, and the GP does not have
        a current set of parameters, the code raises an error. If the code does have a current
        set of parameters but the MLE parameters have not been estimated, it gives a warning
        but continues with the predictions using the current parameters. If it has current
        parameters as well as MLE parameters but they differ, the code issues a warning but
        continues with the predictions using the current parameters.
        
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
        :param predict_from_samples: (optional) Flag indicating if predictions are to be made
                                     from samples. Default is ``False``
        :type predict_from_samples: bool
        :returns: Tuple of numpy arrays holding the predictions, uncertainties, and derivatives,
                  respectively. Predictions and uncertainties have shape ``(n_predict,)``
                  while the derivatives have shape ``(n_predict, D)``. If the ``do_unc`` or
                  ``do_deriv`` flags are set to ``False``, then those arrays are replaced by
                  ``None``.
        :rtype: tuple
        """
        
        assert not self.theta is None, "Must set a parameter value to make predictions"
        
        if predict_from_samples:
            return self._predict_samples(testing, do_deriv, do_unc)
        else:
            if self.mle_theta is None:
                warnings.warn("Warning: GP has not been fit")
            elif not np.allclose(self.mle_theta, self.theta):
                warnings.warn("Warning: Current parameters are not MLE values")
            return self._predict_single(testing, do_deriv, do_unc)
        
        
=======
        if unc:
            sigma_2 = np.exp(self.theta[-2])

            if include_nugget:
                sigma_2 += self._nugget

            var = np.maximum(sigma_2 - np.sum(Ktest*linalg.cho_solve((self.L, True), Ktest), axis=0),
                             0.)

        inputderiv = None
        if deriv:
            inputderiv = np.zeros((n_testing, self.D))
            mean_deriv = self.mean.mean_inputderiv(testing, self.theta[:switch])
            kern_deriv = self.kernel.kernel_inputderiv(testing, self.inputs, self.theta[switch:-1])
            inputderiv = np.transpose(mean_deriv + np.dot(kern_deriv, self.invQt))

        return PredictResult(mean=mu, unc=var, deriv=inputderiv)

    def __call__(self, testing):
        """A Gaussian process object is callable: calling it is the same as
        calling `predict` without uncertainty and derivative
        predictions, and extracting the zeroth component for the
        'mean' prediction.
        """
        return (self.predict(testing, unc=False, deriv=False)[0])

>>>>>>> .merge_file_wR9ZhR
    def __str__(self):
        """
        Returns a string representation of the model

        :returns: A string representation of the model (indicates number of training examples
                  and inputs)
        :rtype: str
        """
<<<<<<< .merge_file_F65fzN
        
        return "Gaussian Process with "+str(self.n)+" training examples and "+str(self.D)+" input variables"

=======

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
>>>>>>> .merge_file_wR9ZhR
