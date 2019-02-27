import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from scipy.spatial.distance import cdist
from scipy import linalg
from scipy.linalg import lapack
import logging
import warnings

class GaussianProcess(object):
    """
    Refactored version of Gaussian Process from Jose Gomez-Dans' implementation to fix some of the stability issues.
    """
    def __init__(self, *args):
        """
        Init method, can be initialized with one or two arguments
        """
        if len(args) == 1:
            raise NotImplementedError("Reading emulator from file not implemented yet")
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

    def save_emulator(self, filename):
        "Saves emulator to file"
        pass
        
    def _load_emulator(self, filename):
        "Loads emulator from file"
        pass
    
    def _jit_cholesky(self, Q, maxtries=5):
        "Performs Cholesky decomposition adding noise to the diagonal as needed. Adapted from code in GPy"
        Q = np.ascontiguousarray(Q)
        L, info = lapack.dpotrf(Q, lower=1)
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
        "Precalculates matrices needed for fitting"
        exp_theta = np.exp(self.theta)
        # Calculation of the covariance matrix Q using theta
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
        "Set parameters of emulator"
        
        theta = np.array(theta)
        assert theta.shape == (self.D + 1,), "Parameter vector must have length number of inputs + 1"
        
        self.theta = theta
        self._prepare_likelihood()
    
    def loglikelihood(self, theta):
        "Calculate the negative loglikelihood at a particular value of the hyperparameters"
        
        self._set_params(theta)

        loglikelihood = (
            0.5 * self.logdetQ
            + 0.5 * np.dot(self.targets, self.invQt)
            + 0.5 * self.n * np.log(2. * np.pi)
        )
        self.current_theta = theta
        self.current_loglikelihood = loglikelihood
        return loglikelihood
    
    def partial_devs(self, theta):
        "Calculate the partial derivatives of the negative loglikelihood wrt the hyper parameters"
        
        if not np.allclose(np.array(theta), self.theta):
            warnings.warn("Value of hyperparameters has changed, recomputing...", RuntimeWarning)
            self._set_params(theta)
        
        partials = np.zeros(self.D + 1)
        
        for d in range(self.D):
            dKdtheta = 0.5*cdist(np.reshape(self.inputs[:,d], (self.n, 1)),
                                 np.reshape(self.inputs[:,d], (self.n, 1)), "sqeuclidean")*self.Q
            partials[d] = 0.5*np.exp(self.theta[d])*(np.dot(self.invQt, np.dot(dKdtheta, self.invQt)) - np.sum(self.invQ*dKdtheta))
        
        partials[self.D] = -0.5*(np.dot(self.invQt, np.dot(self.Q, self.invQt)) - np.sum(self.invQ*self.Q))
        
        return partials
    
    def get_n(self):
        "Returns number of training examples"
        return self.n
        
    def get_D(self):
        "Returns number of input parameters"
        return self.D
        
    def __str__(self):
        """
        Returns a string representation of the emulator
        """
        return "Gaussian Process with "+str(self.n)+" training examples and "+str(self.D)+" input variables"