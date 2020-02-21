import numpy as np
import scipy.stats
from scipy.linalg import LinAlgError
from scipy.optimize import minimize
from .GaussianProcess import GaussianProcess

def fit_GP_MLE(*args, n_tries=15, theta0=None, method="L-BFGS-B", **kwargs):
    """
    Fit a Gaussian Process by attempting to minimize the negative log-likelihood

    Fits the hyperparameters of a Gaussian Process by attempting to minimize the negative
    log-likelihood multiple times from a given starting location and using a particular
    minimization method. The best result found among all of the attempts is returned,
    unless all attempts to fit the parameters result in an error (see below).

    The arguments to the method can either be an existing ``GaussianProcess`` instance,
    or a list of arguments to be passed to the ``__init__`` method of ``GaussianProcess``.
    Keyword arguments for creating a new ``GaussianProcess`` object can either be
    passed as part of the ``*args`` list or as keywords (if present in ``**kwargs``, they
    will be extracted and passed separately to the ``__init__`` method).

    If the method encounters an overflow (this can result because the parameter values stored are
    the logarithm of the actual hyperparameters to enforce positivity) or a linear algebra error
    (occurs when the covariance matrix cannot be inverted, even with additional noise added along
    the diagonal if adaptive noise was selected), the iteration is skipped. If all attempts to find
    optimal hyperparameters result in an error, then the method raises an exception.

    The ``theta0`` parameter is the point at which the first iteration will start. If more than
    one attempt is made, subsequent attempts will use random starting points.

    The user can specify the details of the minimization method, using any of the gradient-based
    optimizers available in ``scipy.optimize.minimize``. Any additional parameters beyond the method
    specification can be passed as keyword arguments.

    The function returns a fit ``GaussianProcess`` instance, either the original one passed
    to the function, or the new one created from the included arguments.

    :param ``*args``: Either a single ``GaussianProcess`` instance, or arguments to passed
                      to the ``__init__`` method when creating a new ``GaussianProcess``
    :param n_tries: Number of attempts to minimize the negative log-likelihood function.
                    Must be a positive integer (optional, default is 15)
    :type n_tries: int
    :param theta0: Initial starting point for the first iteration. If present, must be
                   array-like with shape ``(n_params,)`` based on the specific
                   ``GaussianProcess`` being fit. If ``None`` is given, then a random value
                   is chosen. (Default is ``None``)
    :type theta0: None or ndarray
    :param method: Minimization method to be used. Can be any gradient-based optimization
                   method available in ``scipy.optimize.minimize``. (Default is ``'L-BFGS-B'``)
    :type method: str
    :param ``**kwargs``: Additional keyword arguments to be passed to ``GaussianProcess.__init__``
                         or the minimization routine. Relevant parameters for the GP are
                         automatically split out from those used in the minimization function.
                         See available parameters in the corresponding functions for details.
    :returns: Fit GP instance
    :rtype: GaussianProcess
    """

    if len(args) == 1:
        gp = args[0]
        if not isinstance(gp, GaussianProcess):
            raise TypeError("single arg to fit_GP_MLE must be a GaussianProcess instance")
    elif len(args) < 2:
        raise TypeError("missing required inputs/targets arrays to GaussianProcess")
    else:
        gp_kwargs = {}
        for key in ["mean", "kernel", "nugget", "inputdict", "use_patsy"]:
            if key in kwargs:
                gp_kwargs[key] = kwargs[key]
                del kwargs[key]
        gp = GaussianProcess(*args, **gp_kwargs)

    n_tries = int(n_tries)
    assert n_tries > 0, "number of attempts must be positive"

    np.seterr(divide = 'raise', over = 'raise', invalid = 'raise')

    loglike_values = []
    theta_values = []

    theta_startvals = 5.*(np.random.rand(n_tries, gp.n_params) - 0.5)
    if not theta0 is None:
        theta0 = np.array(theta0)
        assert theta0.shape == (gp.n_params,), "theta0 must be a 1D array with length n_params"
        theta_startvals[0,:] = theta0

    for theta in theta_startvals:
        try:
            min_dict = minimize(gp.loglikelihood, theta, method = method,
                                jac = gp.loglike_deriv, options = kwargs)

            min_theta = min_dict['x']
            min_loglike = min_dict['fun']

            loglike_values.append(min_loglike)
            theta_values.append(min_theta)
        except LinAlgError:
            print("Matrix not positive definite, skipping this iteration")
        except FloatingPointError:
            print("Floating point error in optimization routine, skipping this iteration")

    if len(loglike_values) == 0:
        raise RuntimeError("Minimization routine failed to return a value")

    loglike_values = np.array(loglike_values)
    idx = np.argmin(loglike_values)

    gp.fit(theta_values[idx])

    return gp