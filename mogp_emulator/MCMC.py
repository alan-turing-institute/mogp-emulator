import numpy as np
from inspect import signature

def MH_proposal(current_params, step_sizes):
    "propose new point in MCMC"
    
    current_params = np.array(current_params)
    step_sizes = np.array(step_sizes)
    
    assert len(current_params.shape) == 1, "current parameters must be a 1D array"
    assert len(step_sizes.shape) == 2, "step sizes must be a 2D array"
    assert step_sizes.shape[0] == step_sizes.shape[1], "step sizes must be a square array"
    assert len(current_params) == step_sizes.shape[0], "length of current parameters must match length of step sizes"
    assert np.all(np.diag(step_sizes) > 0.), "step sizes must be a positive definite matrix"
    
    return np.random.multivariate_normal(mean=current_params, cov = step_sizes)


def MCMC_step(loglikelihood, current_param, step_sizes, loglike_sign = 1.):
    "take an MCMC step"
    
    assert callable(loglikelihood), "loglikelihood must be a callable function"
    assert len(signature(loglikelihood).parameters) == 1
    assert loglike_sign == 1. or loglike_sign == -1., "loglikelihood sign must be +/- 1"
    
    next_point = MH_proposal(current_param, step_sizes)
    
    H = loglike_sign*(-loglikelihood(current_param) + loglikelihood(next_point))
    
    if H >= np.log(np.random.random()) and np.isfinite(H):
        accept = True
    else:
        accept = False

    return next_point, accept

def sample_MCMC(loglikelihood, start, step_size, n_samples = 1000, thin = 0, loglike_sign = 1.):
    "draw MCMC samples for a loglikelihood given weak priors"

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
        next_point, accept = MCMC_step(loglikelihood, samples[i], step_size, loglike_sign)
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
            first_lag[i] = autocorr[np.argmax(autocorr)+1]/np.max(autocorr)

    return thinned, np.array(rejected), acceptance, first_lag

def autothin_samples(signal):
    "automatically determine thinning needed to obtain uncorrelated samples"
    
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
        print("automatic thinning failed, posterior distribution may be multimodal")
        maxthin = 1
        
    return maxthin
    