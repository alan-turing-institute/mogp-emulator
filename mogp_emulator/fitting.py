import numpy as np
import scipy.stats
from scipy.linalg import LinAlgError
from scipy.optimize import minimize
from mogp_emulator.GaussianProcess import GaussianProcess
from mogp_emulator.MultiOutputGP import MultiOutputGP
from multiprocessing import Pool
from functools import partial
import platform

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
    one attempt is made, subsequent attempts will use random starting points. If you are fitting
    Multiple Outputs, then this argument can take any of the following forms: (1) None (random
    start points for all emulators), (2) a list of numpy arrays or ``NoneTypes`` with length
    ``n_emulators``, (3) a numpy array of shape ``(n_params,)`` or ``(n_emulators, n_params)``
    which with either use the same start point for all emulators or the specified start
    point for all emulators. Note that if you us a numpy array, all emulators must have the
    same number of parameters, while using a list allows more flexibility.

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
                   ``GaussianProcess`` being fit. If a ``MultiOutputGP`` is being fit
                   it must be a list of length ``n_emulators`` with each entry as either
                   ``None`` or a numpy array of shape ``(n_params,)``, or a numpy array
                   with shape ``(n_emulators, n_params)`` (note that if the various emulators
                   have different numbers of parameters, the numpy array option will not work).
                   If ``None`` is given, then a random value is chosen. (Default is ``None``)
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

    Returns a single GP object that has its hyperparameters fit to the MAP value.
    Accepts keyword arguments passed to scipy's minimization routine.
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

def _fit_single_GP_MAP_bound(gp, theta0, n_tries, method, **kwargs):
    "fitting function accepting theta0 as an argument for parallelization"

    return _fit_single_GP_MAP(gp, n_tries=n_tries, theta0=theta0, method=method, **kwargs)

def _fit_MOGP_MAP(gp, n_tries=15, theta0=None, method='L-BFGS-B', **kwargs):
    """
    Fit hyperparameters using MAP for multiple GPs in parallel

    Uses Python Multiprocessing to fit GPs in parallel by calling the above routine for a single
    GP for each of the emulators in the MOGP class. Returns a MultiOutputGP object where all
    emulators have been fit to the MAP value.

    Accepts a ``processes`` argument (integer or None) as a keyword to control the number of
    subprocesses used to fit the individual GPs in parallel. Must be positive. Default is ``None``.
    """

    assert isinstance(gp, MultiOutputGP)

    try:
        processes = kwargs['processes']
        del kwargs['processes']
    except KeyError:
        processes = None

    assert int(n_tries) > 0, "n_tries must be a positive integer"

    if theta0 is None:
        theta0 = [ None ]*gp.n_emulators
    else:
        if isinstance(theta0, np.ndarray):
            if theta0.ndim == 1:
                theta0 = [theta0]*gp.n_emulators
            else:
                assert theta0.ndim == 2, "theta0 must be a 1D or 2D array"
                assert theta0.shape[0] == gp.n_emulators, "bad shape for fitting starting points"
        elif isinstance(theta0, list):
            assert len(theta0) == gp.n_emulators, "theta0 must be a list of length n_emulators"

    if not processes is None:
        processes = int(processes)
        assert processes > 0, "number of processes must be positive"

    n_tries = int(n_tries)

    # partial(fit_GP_MAP, )

    if platform.system() == "Windows":
        fit_MOGP = [fit_GP_MAP(emulator, n_tries=n_tries, theta0=t0, method=method, **kwargs)
                    for (emulator, t0) in zip(gp.emulators, theta0)]
    else:
        with Pool(processes) as p:
            fit_MOGP = p.starmap(partial(_fit_single_GP_MAP_bound, n_tries=n_tries, method=method, **kwargs),
                                 [(emulator, t0) for (emulator, t0) in zip(gp.emulators, theta0)])

    gp.emulators = fit_MOGP

    return gp