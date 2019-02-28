import numpy as np
from scipy.optimize import minimize
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
        
        emulator_file = None
        theta = None
        
        if len(args) == 1:
            emulator_file = args[0]
            inputs, targets, theta = self._load_emulator(emulator_file)
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
        
        if not (emulator_file is None or theta is None):
            self._set_params(theta)

    def save_emulator(self, filename):
        "Saves emulator to file"
        
        emulator_dict = {}
        emulator_dict['targets'] = self.targets
        emulator_dict['inputs'] = self.inputs
        
        try:
            emulator_dict['theta'] = self.theta
        except AttributeError:
            pass
        
        np.savez(filename, **emulator_dict)
        
    def _load_emulator(self, filename):
        "Loads emulator from file"
        
        emulator_file = np.load(filename)
        
        try:
            inputs = np.array(emulator_file['inputs'])
            targets = np.array(emulator_file['targets'])
        except KeyError:
            raise KeyError("Emulator file does not contain emulator inputs and targets")
            
        try:
            theta = np.array(emulator_file['theta'])
        except KeyError:
            theta = None
            
        return inputs, targets, theta
    
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
            self._set_params(theta)
        
        partials = np.zeros(self.D + 1)
        
        for d in range(self.D):
            dKdtheta = 0.5*cdist(np.reshape(self.inputs[:,d], (self.n, 1)),
                                 np.reshape(self.inputs[:,d], (self.n, 1)), "sqeuclidean")*self.Q
            partials[d] = 0.5*np.exp(self.theta[d])*(np.dot(self.invQt, np.dot(dKdtheta, self.invQt)) - np.sum(self.invQ*dKdtheta))
        
        partials[self.D] = -0.5*(np.dot(self.invQt, np.dot(self.Q, self.invQt)) - np.sum(self.invQ*self.Q))
        
        return partials
    
    def _learn(self, theta0, method = 'L-BFGS-B', **kwargs):
        "Minimize loglikelihood function wrt the hyperparameters, starting from theta0"
        
        self._set_params(theta0)
        
        fmin_dict = minimize(self.loglikelihood, theta0, method = method, jac = self.partial_devs, options = kwargs)
        
        if not fmin_dict['success']:
            warnings.warn("Minimization routine resulted in a warning", RuntimeWarning)
            
        return fmin_dict['x'], fmin_dict['fun']
    
    def learn_hyperparameters(self, n_tries = 15, theta0 = None, method = 'L-BFGS-B', **kwargs):
        "Fit hyperparameters by attempting to minimize the negative log-likelihood multiple times, returns best result"
    
        loglikelihood_values = []
        theta_values = []
        
        theta_startvals = 5.*np.random.rand(n_tries, self.D + 1) - 0.5
        if not theta0 is None:
            theta0 = np.array(theta0)
            assert theta0.shape == (self.D + 1,), "theta0 must be a 1D array with length D + 1"
            theta_startvals[0,:] = theta0

        for theta in theta_startvals:
            min_theta, min_loglikelihood = self._learn(theta, method, **kwargs)
            loglikelihood_values.append(min_loglikelihood)
            theta_values.append(min_theta)
            
        loglikelihood_values = np.array(loglikelihood_values)
        idx = np.argsort(loglikelihood_values)[0]
        
        self._set_params(theta_values[idx])
        
        return loglikelihood_values[idx], theta_values[idx]
    
    def predict(self, testing, do_deriv=True, do_unc=True):
        "Make predictions on a given set of inputs"
        
        testing = np.array(testing)
        if len(testing.shape) == 1:
            testing = np.reshape(testing, (1, len(testing)))
        assert len(testing.shape) == 2
                        
        n_testing, D = np.shape(testing)
        assert D == self.D
        
        exp_theta = np.exp(self.theta)

        Ktest = cdist(np.sqrt(exp_theta[: (self.D)]) * self.inputs,
                      np.sqrt(exp_theta[: (self.D)]) * testing, "sqeuclidean")

        Ktest = exp_theta[self.D] * np.exp(-0.5 * Ktest)

        mu = np.dot(Ktest.T, self.invQt)
        
        var = None
        if do_unc:
            var = exp_theta[self.D] - np.sum(Ktest * np.dot(self.invQ, Ktest), axis=0)
        
        deriv = None
        if do_deriv:
            deriv = np.zeros((n_testing, self.D))
            for d in range(self.D):
                aa = (self.inputs[:, d].flatten()[None, :] - testing[:, d].flatten()[:, None])
                c = Ktest * aa.T
                deriv[:, d] = exp_theta[d] * np.dot(c.T, self.invQt)
        return mu, var, deriv
        
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