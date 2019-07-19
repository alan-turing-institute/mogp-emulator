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
    
    r_matrix = cdist(np.sqrt(exp_theta[:(D - 1)])*x1, np.sqrt(exp_theta[:(D-1)])*x2, "euclidean")
    
    return r_matrix
    
def squared_exponential(x1, x2, params):
    "compute kernel values for squared exponential"
    
    params = np.array(params)
    assert params.ndim == 1, "parameters must be a vector"
    D = len(params)
    assert D >= 2, "minimum number of parameters in a covariance kernel is 2"
    
    K = calc_r(x1, x2, params)
    K = np.exp(params[D - 1]) * np.exp(-0.5 * K**2)
    
    return K

def squared_exponential_deriv(x1, x2, params):
    "compute gradient of squared_exponential kernel"
    
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
    
    n1 = x1.shape[0]
            
    x2 = np.array(x2)
    
    assert x2.ndim == 1 or x2.ndim == 2, "bad number of dimensions in input x2"
    
    if x2.ndim == 2:
        assert x2.shape[1] == D - 1, "bad shape for x2"
    else:
        if D == 2:
            x2 = np.reshape(x2, (len(x2), 1))
        else:
            x2 = np.reshape(x2, (1, D - 1))
            
    n2 = x2.shape[0]
    
    dKdtheta = np.zeros((D, n1, n2))
    
    dKdtheta[-1] = squared_exponential(x1, x2, params)
    
    for d in range(D - 1):
        dKdtheta[d] = - 0.5 * np.exp(params[d]) * cdist(np.reshape(x1[:,d], (n1, 1)),
                                                        np.reshape(x2[:,d], (n2, 1)), "sqeuclidean") * dKdtheta[-1]
    
    return dKdtheta

def matern_5_2(x1, x2, params):
    "compute covariance using the matern 5/2 kernel"
    
    params = np.array(params)
    assert params.ndim == 1, "parameters must be a vector"
    D = len(params)
    assert D >= 2, "minimum number of parameters in a covariance kernel is 2"
    
    Q = calc_r(x1, x2, params)
    Q = np.exp(params[D - 1]) * (1. + np.sqrt(5.)*Q + 5./3.*Q**2)*np.exp(-np.sqrt(5.) * Q)
    
    return Q