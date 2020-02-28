import numpy as np
import scipy.stats
from scipy.linalg import LinAlgError
from scipy.optimize import minimize
from .GaussianProcess import GaussianProcess
from .MultiOutputGP import MultiOutputGP
from multiprocessing import Pool
from functools import partial

def fit_GP_MAP(*args, n_tries=15, theta0=None, method="L-BFGS-B", **kwargs):
    """
    Fit one or more Gaussian Processes by attempting to minimize the negative log-posterior

    Fits the hyperparameters of one or more Gaussian Processes by attempting to minimize
    the negative log-posterior multiple times from a given starting location and using
    a particular minimization method. The best result found among all of the attempts is
    returned, unless all attempts to fit the parameters result in an error (see below).

    The arguments to the method can either be an existing ``GaussianProcess`` or
    ``MultiOutputGP`` instance, or a list of arguments to be passed to the ``__init__``
    method of ``GaussianProcess`` or ``MultiOutputGP`` if more than one output is detected.
    Keyword arguments for creating a new ``GaussianProcess`` or ``MultiOutputGP`` object can
    either be passed as part of the ``*args`` list or as keywords (if present in ``**kwargs``, they
    will be extracted and passed separately to the ``__init__`` method).

    If the method encounters an overflow (this can result because the parameter values stored are
    the logarithm of the actual hyperparameters to enforce positivity) or a linear algebra error
    (occurs when the covariance matrix cannot be inverted, even with additional noise added along
    the diagonal if adaptive noise was selected), the iteration is skipped. If all attempts to find
    optimal hyperparameters result in an error, then the method raises an exception.

    The ``theta0`` parameter is the point at which the first iteration will start. If more than
    one attempt is made, subsequent attempts will use random starting points. Note that the
    same starting point is assumed for all emulators if using Multiple Outputs, so if the
    emulators require differing numbers of parameters a fixed start point cannot be used.

    The user can specify the details of the minimization method, using any of the gradient-based
    optimizers available in ``scipy.optimize.minimize``. Any additional parameters beyond the method
    specification can be passed as keyword arguments.

    The function returns a fit ``GaussianProcess`` or ``MultiOutputGP`` instance, either the original
    one passed to the function, or the new one created from the included arguments.

    :param ``*args``: Either a single ``GaussianProcess`` or ``MultiOutputGP`` instance,
                      or arguments to be passed to the ``__init__`` method when creating a new
                      ``GaussianProcess`` or ``MultiOutputGP`` instance.
    :param n_tries: Number of attempts to minimize the negative log-posterior function.
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
    :param ``**kwargs``: Additional keyword arguments to be passed to ``GaussianProcess.__init__``,
                         ``MultiOutputGP.__init__``, or the minimization routine. Relevant parameters
                         for the GP classes are automatically split out from those used in the
                         minimization function. See available parameters in the corresponding functions
                         for details.
    :returns: Fit GP or Multi-Output GP instance
    :rtype: GaussianProcess or MultiOutputGP
    """

    if len(args) == 1:
        gp = args[0]
        if isinstance(gp, MultiOutputGP):
            return _fit_MOGP_MAP(gp, n_tries, theta0, method, **kwargs)
        elif isinstance(gp, GaussianProcess):
            return _fit_single_GP_MAP(gp, n_tries, theta0, method, **kwargs)
        else:
            raise TypeError("single arg to fit_GP_MAP must be a GaussianProcess or MultiOutputGP instance")
    elif len(args) < 2:
        raise TypeError("missing required inputs/targets arrays to GaussianProcess")
    else:
        gp_kwargs = {}
        for key in ["mean", "kernel", "priors", "nugget", "inputdict", "use_patsy"]:
            if key in kwargs:
                gp_kwargs[key] = kwargs[key]
                del kwargs[key]
        try:
            gp = GaussianProcess(*args, **gp_kwargs)
            return _fit_single_GP_MAP(gp, n_tries, theta0, method, **kwargs)
        except AssertionError:
            gp = MultiOutputGP(*args, **gp_kwargs)
            return _fit_MOGP_MAP(gp, n_tries, theta0, method, **kwargs)

def _fit_single_GP_MAP(gp, n_tries=15, theta0=None, method='L-BFGS-B', **kwargs):
    """
    Fit hyperparameters using MAP for a single GP
    """

    assert isinstance(gp, GaussianProcess)

    n_tries = int(n_tries)
    assert n_tries > 0, "number of attempts must be positive"

    np.seterr(divide = 'raise', over = 'raise', invalid = 'raise')

    logpost_values = []
    theta_values = []

    theta_startvals = 5.*(np.random.rand(n_tries, gp.n_params) - 0.5)
    if not theta0 is None:
        theta0 = np.array(theta0)
        assert theta0.shape == (gp.n_params,), "theta0 must be a 1D array with length n_params"
        theta_startvals[0,:] = theta0

    for theta in theta_startvals:
        try:
            min_dict = minimize(gp.logposterior, theta, method = method,
                                jac = gp.logpost_deriv, options = kwargs)

            min_theta = min_dict['x']
            min_logpost = min_dict['fun']

            logpost_values.append(min_logpost)
            theta_values.append(min_theta)
        except LinAlgError:
            print("Matrix not positive definite, skipping this iteration")
        except FloatingPointError:
            print("Floating point error in optimization routine, skipping this iteration")

    if len(logpost_values) == 0:
        raise RuntimeError("Minimization routine failed to return a value")

    logpost_values = np.array(logpost_values)
    idx = np.argmin(logpost_values)

    gp.fit(theta_values[idx])

    return gp

def _fit_MOGP_MAP(gp, n_tries=15, theta0=None, method='L-BFGS-B', **kwargs):
    """
    Fit hyperparameters using MAP for multiple GPs in parallel
    """

    assert isinstance(gp, MultiOutputGP)

    try:
        processes = kwargs['processes']
        del kwargs['processes']
    except KeyError:
        processes = None

    assert int(n_tries) > 0, "n_tries must be a positive integer"
    if not theta0 is None:
        theta0 = np.array(theta0)
    if not processes is None:
        processes = int(processes)
        assert processes > 0, "number of processes must be positive"

    n_tries = int(n_tries)

    with Pool(processes) as p:
        fit_MOGP = p.starmap(partial(fit_GP_MAP, n_tries=n_tries, theta0=theta0, method=method, **kwargs),
                            [(emulator,) for emulator in gp.emulators])

    gp.emulators = fit_MOGP

    return gp