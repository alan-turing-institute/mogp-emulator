import numpy as np
from scipy.spatial.distance import cdist

def calc_r(x1, x2, params):
    "calculate distance between all pairs for forming covariance matrix"
    
    params = np.array(params)
    assert params.ndim == 1, "parameters must be a vector"
    D = len(params)
    assert D >= 2, "minimum number of parameters in a covariance kernel is 2"
    
    x1 = np.array(x1)
    
    assert x1.ndim == 1 or x1.ndim == 2, "bad number of dimensions in input x1"
    
    if x1.ndim == 2:
        assert x1.shape[1] == D - 1, "bad shape for x1"
    else:
        if D == 2:
            x1 = np.reshape(x1, (len(x1), 1))
        else:
            x1 = np.reshape(x1, (1, D - 1))
            
    x2 = np.array(x2)
    
    assert x2.ndim == 1 or x2.ndim == 2, "bad number of dimensions in input x2"
    
    if x2.ndim == 2:
        assert x2.shape[1] == D - 1, "bad shape for x2"
    else:
        if D == 2:
            x2 = np.reshape(x2, (len(x2), 1))
        else:
            x2 = np.reshape(x2, (1, D - 1))
    
    exp_theta = np.exp(params)
    
    Q = cdist(np.sqrt(exp_theta[:(D - 1)])*x1, np.sqrt(exp_theta[:(D-1)])*x2, "euclidean")
    
    return Q
    
def squared_exponential(x1, x2, params):
    "compute kernel values for squared exponential"
    
    params = np.array(params)
    assert params.ndim == 1, "parameters must be a vector"
    D = len(params)
    assert D >= 2, "minimum number of parameters in a covariance kernel is 2"
    
    Q = calc_r(x1, x2, params)
    Q = np.exp(params[D - 1]) * np.exp(-0.5 * Q**2)
    
    return Q
    
def matern_5_2(x1, x2, params):
    "compute covariance using the matern 5/2 kernel"
    
    params = np.array(params)
    assert params.ndim == 1, "parameters must be a vector"
    D = len(params)
    assert D >= 2, "minimum number of parameters in a covariance kernel is 2"
    
    Q = calc_r(x1, x2, params)
    Q = np.exp(params[D - 1]) * (1. + np.sqrt(5.)*Q + 5./3.*Q**2)*np.exp(-np.sqrt(5.) * Q)
    
    return Q