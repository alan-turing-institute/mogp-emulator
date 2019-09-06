import numpy as np
from numpy.linalg import LinAlgError
from inspect import signature
import warnings

def MH_proposal(current_params, step_sizes):
    """
    Propose an MCMC step using a Metropolis-Hastings method
    
    Proposes the next point in an MCMC sampler using the Metropolis-Hastings method.
    Inputs are the current point and a covariance matrix. The next point is drawn from
    a multivariate normal distribution centered around the current point. The covariance
    matrix must be a 2D ``n`` by ``n`` array, where ``n`` is the number of parameters, and
    must be positive definite. Returns a 1D array of length ``n`` holding the new proposed
    parameter values.
    
    :param current_params: Current value of the parameters. Must be a 1D array.
    :type current_params: ndarray
    :param step_sizes: Covariance matrix from which steps are drawn. Must be a 2D array
                       with both dimensions the same length as ``current_params``, and
                       must be positive definite.
    :type step_sizes: ndarray
    :returns: New value of parameters, a 1D array with the same length as ``current_params``
    :rtype: ndarray
    """
    
    current_params = np.array(current_params)
    step_sizes = np.array(step_sizes)
    
    assert len(current_params.shape) == 1, "current parameters must be a 1D array"
    assert len(step_sizes.shape) == 2, "step sizes must be a 2D array"
    assert step_sizes.shape[0] == step_sizes.shape[1], "step sizes must be a square array"
    assert len(current_params) == step_sizes.shape[0], "length of current parameters must match length of step sizes"
    assert np.all(np.diag(step_sizes) > 0.), "step sizes must be a positive definite matrix"
    
    return np.random.multivariate_normal(mean=current_params, cov = step_sizes)


def MCMC_step(loglikelihood, current_params, step_sizes, loglike_sign = 1.):
    """
    Method to take a weak prior Metropolis-Hastings MCMC step
    
    Take a Metropolis-Hastings MCMC step with given log-likelihood function and step sizes.
    Method uses a multivariate normal distribution centered around the current parameter
    values from which the next point is drawn, and evaluates the log-likelihood for both
    points. If the next point has a larger log-likelihood, the step is always accepted,
    and if the log-likelihood is less for the proposed step it is accepted with a
    probability based on the difference between the two. If the method encounters an
    error when evalutaing the log-likelihood, the step is rejected. Returns the next point
    and a boolean indicating whether or not the step was accepted.
    
    An optional parameter ``loglike_sign`` can be passed that must be a float with the
    value +/- 1. This is multiplied by the log-likelihood and thus allows methods that
    compute the negative log-likelihood to be used in this routine. If values other than
    +/- 1 are passed, the method will raise an error.
    
    :param loglikelihood: Log-likelihood function to be used in the MCMC step. Must be
                          callable and must accept a single argument, which is the array
                          holding the parameters. If this function computes the negative
                          log-likelihood, pass ``loglike_sign = -1.`` to the function as well.
    :type loglikelihood: function or other callable
    :param current_params: Current value of the parameters. Must be a 1D array.
    :type current_params: ndarray
    :param step_sizes: Covariance matrix from which steps are drawn. Must be a 2D array
                       with both dimensions the same length as ``current_params``, and
                       must be positive definite.
    :type step_sizes: ndarray
    :param loglike_sign: Sign for the log-likelihood function. If the provided
                         ``loglikelihood`` function computes the negative log-likelihood,
                         pass ``-1.`` for this parameter. Optional, default value is ``1.``
    :type loglike_sign: float
    :returns: Proposed next point and whether or not the point is accepted as a tuple.
              The first return item is the next point (as a 1D array with the same length\
              as ``current_params``) and the second item is a boolean indicating whether
              or not the step was accepted.
    :rtype: tuple containing a ndarray and a bool
    """
    
    assert callable(loglikelihood), "loglikelihood must be a callable function"
    assert len(signature(loglikelihood).parameters) == 1
    assert loglike_sign == 1. or loglike_sign == -1., "loglikelihood sign must be +/- 1"
    
    next_point = MH_proposal(current_params, step_sizes)

    try:
        H = loglike_sign*(-loglikelihood(current_params) + loglikelihood(next_point))
    except (FloatingPointError, AssertionError, LinAlgError):
        H = np.nan

    if H >= np.log(np.random.random()) and np.isfinite(H):
        accept = True
    else:
        accept = False

    return next_point, accept

def sample_MCMC(loglikelihood, start, step_sizes, n_samples = 1000, thin = 0, loglike_sign = 1.):
    """
    Draw MCMC samples for a given log-likelihood function with weak priors
    
    Compute an MCMC chain for a given log-likelihood function with weak priors. Function
    requires the log-likelihood function, the starting point for the MCMC chain, and an
    array describing the step sizes. Optional parameters are the number of steps to take,
    how the thin the samples, and a sign for the log-likelihood function.
    
    The log-likelihood function must be a function or other callable that accepts a single
    argument, which is a 1D array holding the current parameter values. The starting point
    must be a 1D array holding the starting parameter values, which must match the length of
    the input to the log-likelihood function. The step sizes must be a 2D array with each
    dimension having the same length as the number of parameters, and must be positive
    definite. The step size array is used as the covariance matrix for a multivariate
    normal distribution, from which steps are drawn.
    
    Optional parameters are the number of steps to be taken (must be a positive integer).
    Note that if the chain is thinned, the number of points in the final MCMC chain will
    differ from the number of steps taken.
    
    Thinning may be specified with a non-negative integer. If a positive integer is
    given, the chain will be thinned by only keeping every ``thin`` steps. Note that
    ``thin = 1`` means that the chain will not be thinned. If ``thin = 0`` is given
    (the default value), the chain will automatically be thinned by computing the
    autocorrelation of the chain for each parameter separately and estimating the value
    needed to eliminate correlations in the chain. If the autothinning method fails
    (usually occurrs if the posterior is multimodal), the chain will not be thinned
    and a warning will be given. More details on the autothinning procedure are
    described in the corresponding function.
    
    An optional parameter ``loglike_sign`` can be passed that must be a float with the
    value +/- 1. This is multiplied by the log-likelihood and thus allows methods that
    compute the negative log-likelihood to be used in this routine.
    
    Returns the final thinned MCMC chain (a 2D array, where the first dimension indicates
    the different samples and the second dimension indicates the different parameters),
    an array holding all rejected steps (also a 2D array like the MCMC chain, useful for
    diagnosing problems with convergence), the fraction of steps that are accepted (also
    useful for diagnosing problems with convergence), and the first lag autocorrelation of
    the thinned MCMC chain.
    
    :param loglikelihood: Log-likelihood function to be used in the MCMC step. Must be
                          callable and must accept a single argument, which is the array
                          holding the parameters. If this function computes the negative
                          log-likelihood, pass ``loglike_sign = -1.`` to the function as well.
    :type loglikelihood: function or other callable
    :param start: Starting value of the parameters. Must be a 1D array.
    :type start: ndarray
    :param step_sizes: Covariance matrix from which steps are drawn. Must be a 2D array
                       with both dimensions the same length as ``current_params``, and
                       must be positive definite.
    :type step_sizes: ndarray
    :param n_samples: Number of steps to be taken. Must be a positive integer. Optional,
                      default value is 1000. Note that if the chain is thinned, the
                      final MCMC chain will be shorter than this.
    :type n_samples: int
    :param thin: Integer describing how to thin the MCMC chain. A positive integer
                 indicates manual thinning by keeping every ``thin`` steps (note
                 that ``thin = 1`` means the chain will not be thinned). A value
                 of ``0`` will attempt to autothin the chain.
    :type thin: int
    :param loglike_sign: Sign for the log-likelihood function. If the provided
                         ``loglikelihood`` function computes the negative log-likelihood,
                         pass ``-1.`` for this parameter. Optional, default value is ``1.``
    :type loglike_sign: float
    :returns: MCMC chain (2D array), array of rejected points (2D array), acceptance rate
              (float), and first lag autocorrelation of the thinned MCMC chain (float)
    :rtype: tuple containing (ndarray, ndarray, float, float)
    """

    n_samples = int(n_samples)
    thin = int(thin)

    assert n_samples > 0, "number of samples must be a positive integer"
    assert thin >= 0, "thin must be a non-negative integer"
    
    start = np.array(start)
    
    assert start.ndim == 1, "starting point must be a 1d array"
    n_params = len(start)

    samples = np.zeros((n_samples, n_params))
    samples[0] = start
    rejected = []

    for i in range(n_samples - 1):
        next_point, accept = MCMC_step(loglikelihood, samples[i], step_sizes, loglike_sign)
        if accept:
            samples[i+1] = np.copy(next_point)
        else:
            samples[i+1] = np.copy(samples[i])
            rejected.append(np.array(next_point))
        
    acceptance = float(n_samples - len(rejected))/float(n_samples)

    if thin == 0:
        thin_freq = autothin_samples(samples)
    else:
        thin_freq = thin
    
    thinned = samples[::thin_freq]

    first_lag = np.zeros(n_params)

    if n_samples > 1:
        for i in range(n_params):
            autocorr = np.correlate(thinned[:,i]-np.mean(thinned[:,i]), thinned[:,i]-np.mean(thinned[:,i]), mode="full")
            if np.max(autocorr) > 0.:
                first_lag[i] = autocorr[np.argmax(autocorr)+1]/np.max(autocorr)

    return thinned, np.array(rejected), acceptance, first_lag

def autothin_samples(signal):
    """
    Automatically estimate thinning needed to obtain uncorrelated samples
    
    This function attempts to estimate the thinning needed to obtain uncorrelated samples in an
    MCMC chain. For each separate parameter, the function computes the autocorrelation and
    estimates the lag needed to obtain uncorrelated samples. This is done by recognizing that
    the standard deviation of the autocorrelation of a random signal scales inversely with the
    square root of the number of samples, so when the autocorrelation drops below three times
    this, we use this as a guess to when the signal is uncorrelated. The maximum lag across all
    parameters is returned. If the chain contains fewer than 10 points, or the autocorrelation
    never drops below the target value, the method gives a warning and
    returns 1.
    
    :param signal: MCMC chain to be thinned. Must be a 1D or 2D array. If 2D, the first dimension
                   indicates the MCMC samples, and the second dimension indicates the different
                   parameter values. If 1D, the array is assumed to hold samples of a single
                   parameter, and the array is reshaped to have a singleton second dimension.
    :type signal: ndarray
    :returns: Estimated stride needed to thin the samples to obtain uncorrelated samples. If
              the method does not succeed, gives a warning and returns ``1``.
    :rtype: int
    """
    
    signal = np.array(signal)
    if signal.ndim == 1:
        signal = np.reshape(signal, (len(signal), 1))

    assert signal.ndim == 2, "input signal to be thinned must be a 1d or 2d array"
    
    n_samples, n_params = signal.shape
    
    maxthin = 0
    
    if n_samples >= 10:
        for i in range(n_params):
            autocorr = np.correlate(signal[:,i]-np.mean(signal[:,i]), signal[:,i]-np.mean(signal[:,i]), mode="full")
            start = np.argmax(autocorr)
            if np.max(autocorr) > 0.:
                autocorr = autocorr/np.max(autocorr)
            for j in range(1, len(autocorr) - start):
                if np.abs(autocorr[start + j]) < 3./np.sqrt(float(n_samples)):
                    if j > maxthin:
                        maxthin = j
                    break
    
    if maxthin == 0:
        warnings.warn("automatic thinning failed, posterior distribution may be multimodal")
        maxthin = 1
        
    return maxthin
    