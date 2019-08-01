import numpy as np
from scipy.spatial.distance import cdist

class Kernel(object):
    "class representing stationary kernels"
    def __init__(self):
        "initialize a new kernel object"
        pass
        
    def __str__(self):
        return "Stationary Kernel"

    def _check_inputs(self, x1, x2, params):
        "common function for checking dimensions of inputs"
        
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
        
        return x1, n1, x2, n2, params, D
        

    def calc_r(self, x1, x2, params):
        "calculate distance between all pairs for forming covariance matrix"
    
        x1, n1, x2, n2, params, D = self._check_inputs(x1, x2, params)
    
        exp_theta = np.exp(-params[:(D - 1)])
    
        r_matrix = cdist(x1, x2, "seuclidean", V = exp_theta)
    
        return r_matrix

    def calc_drdtheta(self, x1, x2, params):
        "compute derivative of r with respect to the hyperparameters"
    
        x1, n1, x2, n2, params, D = self._check_inputs(x1, x2, params)
    
        exp_theta = np.exp(-params[:(D - 1)])
    
        drdtheta = np.zeros((D - 1, n1, n2))
    
        r_matrix = self.calc_r(x1, x2, params)
        r_matrix[(r_matrix == 0.)] = 1.
    
        for d in range(D - 1):
            drdtheta[d] = 0.5 * np.exp(params[d]) / r_matrix * cdist(np.reshape(x1[:,d], (n1, 1)),
                                                                     np.reshape(x2[:,d], (n2, 1)), "sqeuclidean")
                                                                 
        return drdtheta
    
    def calc_d2rdtheta2(self, x1, x2, params):
        "compute hessian of r with respect to hyperparameters"
    
        x1, n1, x2, n2, params, D = self._check_inputs(x1, x2, params)
    
        exp_theta = np.exp(-params[:(D - 1)])
    
        d2rdtheta2 = np.zeros((D - 1, D - 1, n1, n2))
    
        r_matrix = self.calc_r(x1, x2, params)
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
        
    def kernel_f(self, x1, x2, params):
        "compute kernel values"
    
        x1, n1, x2, n2, params, D = self._check_inputs(x1, x2, params)
    
        return np.exp(params[D - 1]) * self.calc_K(self.calc_r(x1, x2, params))
        
    def kernel_deriv(self, x1, x2, params):
        "compute gradient of kernel"
    
        x1, n1, x2, n2, params, D = self._check_inputs(x1, x2, params)
    
        dKdtheta = np.zeros((D, n1, n2))
    
        dKdtheta[-1] = self.kernel_f(x1, x2, params)
    
        dKdr = self.calc_dKdr(self.calc_r(x1, x2, params))
    
        drdtheta = self.calc_drdtheta(x1, x2, params)
    
        for d in range(D - 1):
            dKdtheta[d] = np.exp(params[-1]) * dKdr * drdtheta[d]
    
        return dKdtheta
        
    def kernel_hessian(self, x1, x2, params):
        "compute hessian of kernel"
    
        x1, n1, x2, n2, params, D = self._check_inputs(x1, x2, params)
    
        d2Kdtheta2 = np.zeros((D, D, n1, n2))
    
        d2Kdtheta2[-1, :] = self.kernel_deriv(x1, x2, params)
        d2Kdtheta2[:, -1] = d2Kdtheta2[-1, :]
    
        r_matrix = self.calc_r(x1, x2, params)
        dKdr = self.calc_dKdr(r_matrix)
        d2Kdr2 = self.calc_d2Kdr2(r_matrix)
    
        drdtheta = self.calc_drdtheta(x1, x2, params)
        d2rdtheta2 = self.calc_d2rdtheta2(x1, x2, params)
    
        for d1 in range(D - 1):
            for d2 in range(D - 1):
                d2Kdtheta2[d1, d2] = np.exp(params[-1]) * (d2Kdr2 * drdtheta[d1] * drdtheta[d2] + dKdr * d2rdtheta2[d1, d2])
    
        return d2Kdtheta2

    def calc_K(self, r):
        "calculate kernel as a function of r"
        
        raise NotImplementedError("base Kernel class does not implement a kernel function")
        
    def calc_dKdr(self, r):
        "calculate first kernel derivative as a function of r"
        
        raise NotImplementedError("base Kernel class does not implement a kernel derivative function")
        
    def calc_d2Kdr2(self, r):
        "calculate second kernel derivative as a function of r"
        
        raise NotImplementedError("base Kernel class does not implement kernel derivatives")

class SquaredExponential(Kernel):
    "squared exponential kernel"

    def calc_K(self, r):
        "compute K(r) for the squared exponential kernel"
    
        assert np.all(r >= 0.), "kernel distances must be positive"
    
        r = np.array(r)
    
        return np.exp(-0.5*r**2)
    
    def calc_dKdr(self, r):
        "compute dK/dr for the squared exponential kernel"
    
        assert np.all(r >= 0.), "kernel distances must be positive"
    
        r = np.array(r)
    
        return -r*np.exp(-0.5*r**2)
    
    def calc_d2Kdr2(self, r):
        "compute d2K/dr2 for the squared exponential kernel"
    
        assert np.all(r >= 0.), "kernel distances must be positive"
    
        r = np.array(r)
    
        return (r**2 - 1.)*np.exp(-0.5*r**2)
        
    def __str__(self):
        "string representation of kernel"
        return "Squared Exponential Kernel"

class Matern52(Kernel):
    "matern 5/2 kernel"
    def calc_K(self, r):
        "compute K(r) for the matern 5/2 kernel"
    
        assert np.all(r >= 0.), "kernel distances must be positive"
    
        r = np.array(r)
    
        return (1.+np.sqrt(5.)*r+5./3.*r**2)*np.exp(-np.sqrt(5.)*r)
    
    def calc_dKdr(self, r):
        "compute dK/dr for the the matern 5/2 kernel"
    
        assert np.all(r >= 0.), "kernel distances must be positive"
    
        r = np.array(r)
    
        return -5./3.*r*(1.+np.sqrt(5.)*r)*np.exp(-np.sqrt(5.)*r)
    
    def calc_d2Kdr2(self, r):
        "compute d2K/dr2 for the the matern 5/2 kernel"
    
        assert np.all(r >= 0.), "kernel distances must be positive"
    
        r = np.array(r)
    
        return 5./3.*(5.*r**2-np.sqrt(5.)*r-1.)*np.exp(-np.sqrt(5.)*r)
        
    def __str__(self):
        "string representation of kernel"
        return "Matern 5/2 Kernel"

    
