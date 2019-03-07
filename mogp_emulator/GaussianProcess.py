import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
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
        >>> mogp.learn_hyperparameters()
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
        are numpy arrays ``inputs`` and ``targets``, described below, or a single argument
        which is the filename (string or file handle) of a previously saved emulator.
        
        ``inputs`` is a 2D array-like object holding the input data, whose shape is
        ``n`` by ``D``, where ``n`` is the number of training examples to be fit and ``D``
        is the number of input variables to each simulation.
        
        ``targets`` is the target data to be fit by the emulator, also held in an array-like
        object. This must be a 1D array of length ``n``.
        
        If two input arguments ``inputs`` and ``targets`` are given:
        :param inputs: Numpy array holding emulator input parameters. Must be 2D with shape
                       ``n`` by ``D``, where ``n`` is the number of training examples and
                       ``D`` is the number of input parameters for each output.
        :type inputs: ndarray
        :param targets: Numpy array holding emulator targets. Must be 1D with length ``n``
        :type targets: ndarray
        
        If one input argument ``emulator_file`` is given:
        :param emulator_file: Filename or file object for saved emulator parameters (using
                              the ``save_emulator`` method)
        :type emulator_file: str or file
        
        :returns: New ``GaussianProcess`` instance
        :rtype: GaussianProcess
        """
        
        emulator_file = None
        theta = None
        
        if len(args) == 1:
            emulator_file = args[0]
            inputs, targets, theta = self._load_emulator(emulator_file)
        elif len(args) == 2:
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
        else:
            raise ValueError("Init method of GaussianProcess requires 1 (file) or 2 (input array, target array) arguments")

        self.inputs = np.array(inputs)
        self.targets = np.array(targets)
        
        self.n = self.inputs.shape[0]
        self.D = self.inputs.shape[1]
        
        if not (emulator_file is None or theta is None):
            self._set_params(theta)

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
        :rtype: tuple containing 3 ndarrays or 2 ndarrays and None (if no theta values
                are found in the emulator file)
        """
        
        emulator_file = np.load(filename)
        
        try:
            inputs = np.array(emulator_file['inputs'])
            targets = np.array(emulator_file['targets'])
        except KeyError:
            raise KeyError("Emulator file does not contain emulator inputs and targets")
            
        try:
            theta = np.array(emulator_file['theta'])
        except KeyError:
            theta = None
            
        return inputs, targets, theta

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
        
        try:
            emulator_dict['theta'] = self.theta
        except AttributeError:
            pass
        
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
        triangular matrix and the amount of jitter necessary to stabilize the decomposition.
        
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
        
        exp_theta = np.exp(self.theta)
        
        self.Q = cdist(np.sqrt(exp_theta[: (self.D)])*self.inputs,
                       np.sqrt(exp_theta[: (self.D)])*self.inputs,
                       "sqeuclidean")
        self.Q = exp_theta[self.D] * np.exp(-0.5 * self.Q)
        
        L, jitter = self._jit_cholesky(self.Q)
        self.Z = self.Q + jitter*np.eye(self.n)
        
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
        self.current_theta = theta
        self.current_loglikelihood = loglikelihood
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
        by both the ``loglikelihood`` and ``partial_devs`` methods. If the function
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
        
        if not np.allclose(np.array(theta), self.theta):
            self._set_params(theta)
        
        partials = np.zeros(self.D + 1)
        
        for d in range(self.D):
            dKdtheta = 0.5 * cdist(np.reshape(self.inputs[:,d], (self.n, 1)),
                                   np.reshape(self.inputs[:,d], (self.n, 1)), "sqeuclidean") * self.Q
            partials[d] = (0.5 * np.exp(self.theta[d]) * (np.dot(self.invQt, np.dot(dKdtheta, self.invQt))
                           - np.sum(self.invQ * dKdtheta)))
        
        partials[self.D] = -0.5 * (np.dot(self.invQt, np.dot(self.Q, self.invQt)) - np.sum(self.invQ * self.Q))
        
        return partials
    
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
        "jitter" or noise added along the diagonal), the iteration is skipped. If all attempts to
        find optimal hyperparameters result in an error, then the method raises an exception.
        
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
        :param **kwargs: Additional keyword arguments to be passed to the minimization routine.
                         see available parameters in ``scipy.optimize.minimize`` for details.
        :returns: Minimum negative log-likelihood values and hyperparameters (numpy array with shape
                  ``(D + 1,)``) used to obtain those values. The method also sets the current values
                  of the hyperparameters to these optimal values and pre-computes the matrices needed
                  to make predictions.
        :rtype: tuple containing a float and an ndarray
        """
    
        n_tries = int(n_tries)
        assert n_tries > 0, "number of attempts must be positive"
        
        np.seterr(over = 'raise')
        
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
                print("Overflow in optimization routine, skipping this iteration")
                
        if len(loglikelihood_values) == 0:
            raise RuntimeError("Minimization routine failed to return a value")
            
        loglikelihood_values = np.array(loglikelihood_values)
        idx = np.argsort(loglikelihood_values)[0]
        
        self._set_params(theta_values[idx])
        
        return loglikelihood_values[idx], theta_values[idx]
    
    def predict(self, testing, do_deriv = True, do_unc = True):
        """
        Make a prediction for a set of input vectors
        
        Makes predictions for the emulator on a given set of input vectors. The input vectors
        must be passed as a ``(n_predict, D)`` or ``(D,)`` shaped array-like object, where
        ``n_predict`` is the number of different prediction points under consideration and
        ``D`` is the number of inputs to the emulator. If the prediction inputs array has shape
        ``(D,)``, then the method assumes ``n_predict == 1``. The prediction is returned as an
        ``(n_predict, )`` shaped numpy array as the first return value from the method.
        
        Optionally, the emulator can also calculate the uncertainties in the predictions 
        and the derivatives with respect to each input parameter. If the uncertainties are
        computed, they are returned as the second output from the method as an ``(n_predict,)``
        shaped numpy array. If the derivatives are computed, they are returned as the third
        output from the method as an ``(n_predict, D)`` shaped numpy array.
        
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

        Ktest = cdist(np.sqrt(exp_theta[: (self.D)]) * self.inputs,
                      np.sqrt(exp_theta[: (self.D)]) * testing, "sqeuclidean")

        Ktest = exp_theta[self.D] * np.exp(-0.5 * Ktest)

        mu = np.dot(Ktest.T, self.invQt)
        
        var = None
        if do_unc:
            var = exp_theta[self.D] - np.sum(Ktest * np.dot(self.invQ, Ktest), axis=0)
        
        deriv = None
        if do_deriv:
            deriv = np.zeros((n_testing, self.D))
            for d in range(self.D):
                aa = (self.inputs[:, d].flatten()[None, :] - testing[:, d].flatten()[:, None])
                c = Ktest * aa.T
                deriv[:, d] = exp_theta[d] * np.dot(c.T, self.invQt)
        return mu, var, deriv
        
    def __str__(self):
        """
        Returns a string representation of the model
        
        :returns: A string representation of the model (indicates number of training examples
                  and inputs)
        :rtype: str
        """
        return "Gaussian Process with "+str(self.n)+" training examples and "+str(self.D)+" input variables"