import numpy as np
from .Kernel import SquaredExponential
from .MCMC import sample_MCMC
from scipy.optimize import minimize
from scipy import linalg
from scipy.linalg import lapack
import logging
import warnings

class GaussianProcess(object):
    """
    Implementation of a Gaussian Process Emulator.

    This class provides an interface to fit a Gaussian Process Emulator to a set of training
    data. The class can be initialized from either a pair of inputs/targets arrays, or a file
    holding data saved from a previous emulator instance (saved via the ``save_emulator``
    method). Once the emulator has been created, the class provides methods for fitting
    optimal hyperparameters, changing hyperparameter values, making predictions, and other
    calculations associated with fitting and making predictions.

    The internal emulator structure involves arrays for the inputs, targets, and hyperparameters.
    Other useful information are the number of training examples ``n`` and the number of input
    parameters ``D``. These parameters are available externally through the ``get_n`` and
    ``get_D`` methods

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
        >>> np.random.seed(47)
        >>> gp.learn_hyperparameters()
        (5.140462159403397, array([-13.02460687,  -4.02939647, -39.2203646 ,   3.25809653]))
        >>> x_predict = np.array([[2., 3., 4.], [7., 8., 9.]])
        >>> gp.predict(x_predict)
        (array([4.74687618, 6.84934016]), array([0.01639298, 1.05374973]),
        array([[8.91363045e-05, 7.18827798e-01, 3.74439445e-16],
               [4.64005897e-06, 3.74191346e-02, 1.94917337e-17]]))

    """

    def __init__(self, *args):
        """
        Create a new GP Emulator

        Creates a new GP Emulator from either the input data and targets to be fit or a
        file holding the input/targets and (optionally) learned parameter values.

        Arguments passed to the ``__init__`` method must be either two arguments which
        are numpy arrays ``inputs`` and ``targets``, described below, three arguments
        which are the same ``inputs`` and ``targets`` arrays plus a float representing
        the ``nugget`` parameter, or a single argument which is the filename (string or file
        handle) of a previously saved emulator.

        ``inputs`` is a 2D array-like object holding the input data, whose shape is
        ``n`` by ``D``, where ``n`` is the number of training examples to be fit and ``D``
        is the number of input variables to each simulation.

        ``targets`` is the target data to be fit by the emulator, also held in an array-like
        object. This must be a 1D array of length ``n``.

        ``nugget`` is the additional noise added to the emulator targets when fitting. This
        can take on values ``None`` (in which case, noise will be added adaptively to
        stabilize fitting), or a non-negative float (in which case, a fixed noise level
        will be used). If no value is specified for the ``nugget`` parameter, ``None``
        is the default.

        If two or three input arguments ``inputs``, ``targets``, and optionally ``nugget`` are
        given:

        :param inputs: Numpy array holding emulator input parameters. Must be 2D with shape
                       ``n`` by ``D``, where ``n`` is the number of training examples and
                       ``D`` is the number of input parameters for each output.
        :type inputs: ndarray
        :param targets: Numpy array holding emulator targets. Must be 1D with length ``n``
        :type targets: ndarray
        :param nugget: Noise to be added to the diagonal or ``None``. A float specifies the
                       noise level explicitly, while if ``None`` is given, the noise will set
                       to be as small as possible to ensure stable inversion of the covariance
                       matrix. Optional, default is ``None``.

        If one input argument ``emulator_file`` is given:

        :param emulator_file: Filename or file object for saved emulator parameters (using
                              the ``save_emulator`` method)

        :type emulator_file: str or file
        :returns: New ``GaussianProcess`` instance
        :rtype: GaussianProcess

        """

        emulator_file = None
        theta = None
        nugget = None

        if len(args) == 1:
            emulator_file = args[0]
            inputs, targets, theta, nugget = self._load_emulator(emulator_file)
        elif len(args) == 2 or len(args) == 3:
            inputs = np.array(args[0])
            targets = np.array(args[1])
            if targets.shape == (1,) and len(inputs.shape) == 1:
                inputs = np.reshape(inputs, (1, len(inputs)))
            if not len(inputs.shape) == 2:
                raise ValueError("Inputs must be a 2D array")
            if not len(targets.shape) == 1:
                raise ValueError("Targets must be a 1D array")
            if not len(targets) == inputs.shape[0]:
                raise ValueError("First dimensions of inputs and targets must be the same length")
            if len(args) == 3:
                nugget = args[2]
                if not nugget == None:
                    nugget = float(nugget)
                    if nugget < 0.:
                        raise ValueError("nugget parameter must be nonnegative or None")
        else:
            raise ValueError("Init method of GaussianProcess requires 1 (file) or 2 (input array, target array) arguments")

        self.inputs = np.array(inputs)
        self.targets = np.array(targets)

        self.n = self.inputs.shape[0]
        self.D = self.inputs.shape[1]

        self.nugget = nugget

        self.kernel =  SquaredExponential()

        if not (emulator_file is None or theta is None):
            self._set_params(theta)
        else:
            self.theta = None

        self.mle_theta = None
        self.samples = None


    @classmethod
    def train_model(Cls, *init_args):
        gp = Cls(*init_args)
        gp.learn_hyperparameters()
        return gp


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

    def get_params(self):
        """
        Returns emulator parameters

        Returns current parameters for the emulator as a numpy array if they have been fit. If no
        parameters have been fit, returns None.

        :returns: Current parameter values (numpy array of length ``D + 1``), or ``None`` if the
                  parameters have not been fit.
        :rtype: ndarray or None
        """

        return self.theta

    def get_nugget(self):
        """
        Returns emulator nugget parameter

        Returns current value of the nugget parameter. If the nugget is selected adaptively, returns None.

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
        to be zero enforces that the emulator strictly interpolates between points. Alternatively,
        if ``nugget`` is set to be ``None``, the fitting routine will adaptively make the noise
        parameter as large as is needed to ensure that the emulator can be fit.

        :param nugget: Controls how noise is added to the emulator. If ``nugget`` is a nonnegative
                       float, then this manually sets the noise parameter (if negative, this will
                       lead to an error), with ``nugget = 0`` resulting in interpolation with no
                       smoothing noise added. ``nugget = None`` will adaptively select the
                       smallest value of the noise term that still leads to a stable inversion of
                       the matrix. Default behavior is ``nugget = None``.
        :type nugget: None or float
        :returns: None
        :rtype: None
        """

        if not nugget == None:
            nugget = float(nugget)
            assert nugget >= 0., "noise parameter must be nonnegative"
        self.nugget = nugget

    def _jit_cholesky(self, Q, maxtries = 5):
        """
        Performs Jittered Cholesky Decomposition

        Performs a Jittered Cholesky decomposition, adding noise to the diagonal of the matrix as needed
        in order to ensure that the matrix can be inverted. Adapted from code in GPy.

        On occasion, the matrix that needs to be inverted in fitting a GP is nearly singular. This arises
        when the training samples are very close to one another, and can be averted by adding a noise term
        to the diagonal of the matrix. This routine performs an exact Cholesky decomposition if it can
        be done, and if it cannot it successively adds noise to the diagonal (starting with 1.e-6 times
        the mean of the diagonal and incrementing by a factor of 10 each time) until the matrix can be
        decomposed or the algorithm reaches ``maxtries`` attempts. The routine returns the lower
        triangular matrix and the amount of noise necessary to stabilize the decomposition.

        :param Q: The matrix to be inverted as an array of shape ``(n,n)``. Must be a symmetric positive
                  definite matrix.
        :type Q: ndarray
        :param maxtries: (optional) Maximum allowable number of attempts to stabilize the Cholesky
                         Decomposition. Must be a positive integer (default = 5)
        :type maxtries: int
        :returns: Lower-triangular factored matrix (shape ``(n,n)`` and the noise that was added to
                  the diagonal to achieve that result.
        :rtype: tuple containing an ndarray and a float
        """

        assert int(maxtries) > 0, "maxtries must be a positive integer"

        Q = np.ascontiguousarray(Q)
        L, info = lapack.dpotrf(Q, lower = 1)
        if info == 0:
            return L, 0.
        else:
            diagQ = np.diag(Q)
            if np.any(diagQ <= 0.):
                raise linalg.LinAlgError("not pd: non-positive diagonal elements")
            jitter = diagQ.mean() * 1e-6
            num_tries = 1
            while num_tries <= maxtries and np.isfinite(jitter):
                try:
                    L = linalg.cholesky(Q + np.eye(Q.shape[0]) * jitter, lower=True)
                    return L, jitter
                except:
                    jitter *= 10
                finally:
                    num_tries += 1
            raise linalg.LinAlgError("not positive definite, even with jitter.")
        import traceback
        try: raise
        except:
            logging.warning('\n'.join(['Added jitter of {:.10e}'.format(jitter),
                '  in '+traceback.format_list(traceback.extract_stack(limit=3)[-2:-1])[0][2:]]))
        return L, jitter

    def _prepare_likelihood(self):
        """
        Pre-calculates matrices needed for fitting and making predictions

        Pre-calculates the matrices needed to compute the log-likelihood and make subsequent
        predictions. This is called any time the hyperparameter values are changed in order
        to ensure that all the information is needed to evaluate the log-likelihood and
        its derivatives, which are needed when fitting the optimal hyperparameters.

        The method computes the covariance matrix (assuming a squared exponential kernel)
        and inverts it using the jittered cholesky decomposition. Some additional information
        is also pre-computed and stored. This method has no inputs and no return value,
        but it does modify the state of the object.

        :returns: None
        """

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
        :type theta: ndarray
        :returns: None
        """

        theta = np.array(theta)
        assert theta.shape == (self.D + 1,), "Parameter vector must have length number of inputs + 1"

        self.theta = theta
        self._prepare_likelihood()

    def loglikelihood(self, theta):
        """
        Calculate the negative log-likelihood at a particular value of the hyperparameters

        Calculate the negative log-likelihood for the given set of parameters. Calling this
        method sets the parameter values and computes the needed inverse matrices in order
        to evaluate the log-likelihood and its derivatives. In addition to returning the
        log-likelihood value, it stores the current value of the hyperparameters and
        log-likelihood in attributes of the object.

        :param theta: Value of the hyperparameters. Must be array-like with shape ``(D + 1,)``
        :type theta: ndarray
        :returns: negative log-likelihood
        :rtype: float
        """

        self._set_params(theta)

        loglikelihood = (0.5 * self.logdetQ +
                         0.5 * np.dot(self.targets, self.invQt) +
                         0.5 * self.n * np.log(2. * np.pi))

        return loglikelihood

    def partial_devs(self, theta):
        """
        Calculate the partial derivatives of the negative log-likelihood

        Calculate the partial derivatives of the negative log-likelihood with respect to
        the hyperparameters. Note that this function is normally used only when fitting
        the hyperparameters, and it is not needed to make predictions.

        During normal use, the ``partial_devs`` method is called after evaluating the
        ``loglikelihood`` method. The implementation takes advantage of this by storing
        the inverse of the covariance matrix, which is expensive to compute and is used
        by the ``loglikelihood``, ``partial_devs``, and ``hessian`` methods. If the function
        is evaluated with a different set of parameters than was previously used to set
        the log-likelihood, the method calls ``_set_params`` to compute the needed
        information. However, caling ``partial_devs`` does not evaluate the log-likelihood,
        so it does not change the cached values of the parameters or log-likelihood.

        :param theta: Value of the hyperparameters. Must be array-like with shape ``(D + 1,)``
        :type theta: ndarray
        :returns: partial derivatives of the negative log-likelihood (array with shape
                  ``(D + 1,)``)
        :rtype: ndarray
        """

        assert theta.shape == (self.D + 1,), "Parameter vector must have length number of inputs + 1"

        if not np.allclose(np.array(theta), self.theta):
            self._set_params(theta)

        partials = np.zeros(self.D + 1)

        dKdtheta = self.kernel.kernel_deriv(self.inputs, self.inputs, self.theta)

        for d in range(self.D + 1):
            partials[d] = -0.5 * (np.dot(self.invQt, np.dot(dKdtheta[d], self.invQt)) - np.sum(self.invQ * dKdtheta[d]))

        return partials

    def hessian(self, theta):
        """
        Calculate the Hessian of the negative log-likelihood

        Calculate the Hessian of the negative log-likelihood with respect to
        the hyperparameters. Note that this function is normally used only when fitting
        the hyperparameters, and it is not needed to make predictions. It is also used
        to estimate an appropriate step size when fitting hyperparameters using
        the lognormal approximation or MCMC sampling.

        When used in an optimization routine, the ``hessian`` method is called after
        evaluating the ``loglikelihood`` method. The implementation takes advantage of
        this by storing the inverse of the covariance matrix, which is expensive to
        compute and is used by the ``loglikelihood`` and ``partial_devs`` methods as well.
        If the function is evaluated with a different set of parameters than was previously
        used to set the log-likelihood, the method calls ``_set_params`` to compute the needed
        information. However, caling ``hessian`` does not evaluate the log-likelihood,
        so it does not change the cached values of the parameters or log-likelihood.

        :param theta: Value of the hyperparameters. Must be array-like with shape ``(D + 1,)``
        :type theta: ndarray
        :returns: Hessian of the negative log-likelihood (array with shape
                  ``(D + 1, D + 1)``)
        :rtype: ndarray
        """

        assert theta.shape == (self.D + 1,), "Parameter vector must have length number of inputs + 1"

        if not np.allclose(np.array(theta), self.theta):
            self._set_params(theta)

        hessian = np.zeros((self.D + 1, self.D + 1))

        dKdtheta = self.kernel.kernel_deriv(self.inputs, self.inputs, self.theta)
        d2Kdtheta2 = self.kernel.kernel_hessian(self.inputs, self.inputs, self.theta)

        for d1 in range(self.D + 1):
            for d2 in range(self.D + 1):
                hessian[d1, d2] = 0.5*(np.linalg.multi_dot([self.invQt,
                                        2.*np.linalg.multi_dot([dKdtheta[d1], self.invQ, dKdtheta[d2]])-d2Kdtheta2[d1, d2],
                                        self.invQt])-
                                        np.trace(np.linalg.multi_dot([self.invQ, dKdtheta[d1], self.invQ, dKdtheta[d2]])
                                                 -np.dot(self.invQ, d2Kdtheta2[d1, d2])))

        return hessian

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

        exp_theta = np.exp(self.theta)

        Ktest = self.kernel.kernel_f(self.inputs, testing, self.theta)

        mu = np.dot(Ktest.T, self.invQt)

        var = None
        if do_unc:
            var = np.maximum(exp_theta[self.D] - np.sum(Ktest * np.dot(self.invQ, Ktest), axis=0), 0.)

        deriv = None
        if do_deriv:
            deriv = np.zeros((n_testing, self.D))
            kern_deriv = self.kernel.kernel_inputderiv(testing, self.inputs, self.theta)
            for d in range(self.D):
                deriv[:, d] = np.dot(kern_deriv[d], self.invQt)

        return mu, var, deriv


    def __call__(self, testing):
        """A Gaussian process object is callable: calling it is the same as
        calling `predict` without uncertainty and derivative
        predictions, and extracting the zeroth component for the
        'value' prediction.
        """
        return (self.predict(testing, do_deriv=False, do_unc=False)[0])


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


    def __str__(self):
        """
        Returns a string representation of the model

        :returns: A string representation of the model (indicates number of training examples
                  and inputs)
        :rtype: str
        """

        return "Gaussian Process with "+str(self.n)+" training examples and "+str(self.D)+" input variables"
