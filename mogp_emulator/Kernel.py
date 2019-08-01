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
    
    exp_theta = np.exp(-params[:(D - 1)])
    
    r_matrix = cdist(x1, x2, "seuclidean", V = exp_theta)
    
    return r_matrix

def calc_drdtheta(x1, x2, params):
    "compute derivative of r with respect to the hyperparameters"
    
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
    
    exp_theta = np.exp(-params[:(D - 1)])
    
    drdtheta = np.zeros((D - 1, n1, n2))
    
    r_matrix = calc_r(x1, x2, params)
    r_matrix[(r_matrix == 0.)] = 1.
    
    for d in range(D - 1):
        drdtheta[d] = 0.5 * np.exp(params[d]) / r_matrix * cdist(np.reshape(x1[:,d], (n1, 1)),
                                                                 np.reshape(x2[:,d], (n2, 1)), "sqeuclidean")
                                                                 
    return drdtheta
    
def calc_d2rdtheta2(x1, x2, params):
    "compute hessian of r with respect to hyperparameters"
    
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
    
    exp_theta = np.exp(-params[:(D - 1)])
    
    d2rdtheta2 = np.zeros((D - 1, D - 1, n1, n2))
    
    r_matrix = calc_r(x1, x2, params)
    r_matrix[(r_matrix == 0.)] = 1.
    
    for d1 in range(D - 1):
        for d2 in range(D - 1):
            if d1 == d2:
                d2rdtheta2[d1, d2] = (0.5*np.exp(params[d1]) / r_matrix *
                                      cdist(np.reshape(x1[:,d1], (n1, 1)),
                                            np.reshape(x2[:,d1], (n2, 1)), "sqeuclidean"))
            d2rdtheta2[d1, d2] -= (0.25 * np.exp(params[d1]) * np.exp(params[d2]) / r_matrix**3 *
                                   cdist(np.reshape(x1[:,d1], (n1, 1)), np.reshape(x2[:,d1], (n2, 1)), "sqeuclidean")*
                                   cdist(np.reshape(x1[:,d2], (n1, 1)), np.reshape(x2[:,d2], (n2, 1)), "sqeuclidean"))
                                                                 
    return d2rdtheta2

def squared_exponential_K(r):
    "compute K(r) for the squared exponential kernel"
    
    assert np.all(r >= 0.), "kernel distances must be positive"
    
    r = np.array(r)
    
    return np.exp(-0.5*r**2)
    
def squared_exponential_dKdr(r):
    "compute dK/dr for the squared exponential kernel"
    
    assert np.all(r >= 0.), "kernel distances must be positive"
    
    r = np.array(r)
    
    return -r*np.exp(-0.5*r**2)
    
def squared_exponential_d2Kdr2(r):
    "compute d2K/dr2 for the squared exponential kernel"
    
    assert np.all(r >= 0.), "kernel distances must be positive"
    
    r = np.array(r)
    
    return (r**2 - 1.)*np.exp(-0.5*r**2)

def squared_exponential(x1, x2, params):
    "compute kernel values for squared exponential"
    
    params = np.array(params)
    assert params.ndim == 1, "parameters must be a vector"
    D = len(params)
    assert D >= 2, "minimum number of parameters in a covariance kernel is 2"
    
    return np.exp(params[D - 1]) * squared_exponential_K(calc_r(x1, x2, params))

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
    
    dKdr = squared_exponential_dKdr(calc_r(x1, x2, params))
    
    drdtheta = calc_drdtheta(x1, x2, params)
    
    for d in range(D - 1):
        dKdtheta[d] = np.exp(params[-1]) * dKdr * drdtheta[d]
    
    return dKdtheta

def squared_exponential_hessian(x1, x2, params):
    "compute hessian of squared exponential kernel"
    
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
    
    d2Kdtheta2 = np.zeros((D, D, n1, n2))
    
    d2Kdtheta2[-1, :] = squared_exponential_deriv(x1, x2, params)
    d2Kdtheta2[:, -1] = d2Kdtheta2[-1, :]
    
    r_matrix = calc_r(x1, x2, params)
    dKdr = squared_exponential_dKdr(r_matrix)
    d2Kdr2 = squared_exponential_d2Kdr2(r_matrix)
    
    drdtheta = calc_drdtheta(x1, x2, params)
    d2rdtheta2 = calc_d2rdtheta2(x1, x2, params)
    
    for d1 in range(D - 1):
        for d2 in range(D - 1):
            d2Kdtheta2[d1, d2] = np.exp(params[-1]) * (d2Kdr2 * drdtheta[d1] * drdtheta[d2] + dKdr * d2rdtheta2[d1, d2])
    
    return d2Kdtheta2

def matern_5_2_K(r):
    "compute K(r) for the matern 5/2 kernel"
    
    assert np.all(r >= 0.), "kernel distances must be positive"
    
    r = np.array(r)
    
    return (1.+np.sqrt(5.)*r+5./3.*r**2)*np.exp(-np.sqrt(5.)*r)
    
def matern_5_2_dKdr(r):
    "compute dK/dr for the the matern 5/2 kernel"
    
    assert np.all(r >= 0.), "kernel distances must be positive"
    
    r = np.array(r)
    
    return -5./3.*r*(1.+np.sqrt(5.)*r)*np.exp(-np.sqrt(5.)*r)
    
def matern_5_2_d2Kdr2(r):
    "compute d2K/dr2 for the the matern 5/2 kernel"
    
    assert np.all(r >= 0.), "kernel distances must be positive"
    
    r = np.array(r)
    
    return 5./3.*(5.*r**2-np.sqrt(5.)*r-1.)*np.exp(-np.sqrt(5.)*r)

def matern_5_2(x1, x2, params):
    "compute covariance using the matern 5/2 kernel"
    
    params = np.array(params)
    assert params.ndim == 1, "parameters must be a vector"
    D = len(params)
    assert D >= 2, "minimum number of parameters in a covariance kernel is 2"
    
    return np.exp(params[D - 1]) * matern_5_2_K(calc_r(x1, x2, params))
    
def matern_5_2_deriv(x1, x2, params):
    "compute gradient of the covariance function for the matern 5/2 kernel"
    
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
    
    dKdtheta[-1] = matern_5_2(x1, x2, params)
    
    dKdr = matern_5_2_dKdr(calc_r(x1, x2, params))
    
    drdtheta = calc_drdtheta(x1, x2, params)
    
    for d in range(D - 1):
        dKdtheta[d] = np.exp(params[-1]) * dKdr * drdtheta[d]
    
    return dKdtheta
    
def matern_5_2_hessian(x1, x2, params):
    "compute hessian of matern 5/2 kernel"
    
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
    
    d2Kdtheta2 = np.zeros((D, D, n1, n2))
    
    d2Kdtheta2[-1, :] = matern_5_2_deriv(x1, x2, params)
    d2Kdtheta2[:, -1] = d2Kdtheta2[-1, :]
    
    r_matrix = calc_r(x1, x2, params)
    dKdr = matern_5_2_dKdr(r_matrix)
    d2Kdr2 = matern_5_2_d2Kdr2(r_matrix)
    
    drdtheta = calc_drdtheta(x1, x2, params)
    d2rdtheta2 = calc_d2rdtheta2(x1, x2, params)
    
    for d1 in range(D - 1):
        for d2 in range(D - 1):
            d2Kdtheta2[d1, d2] = np.exp(params[-1]) * (d2Kdr2 * drdtheta[d1] * drdtheta[d2] + dKdr * d2rdtheta2[d1, d2])
    
    return d2Kdtheta2