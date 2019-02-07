# Class implementing a multi-output Gaussian Process for emulating the output of
# a series of computer simulations. Since each emulator is independent of the
# others, the emulators can be fit in parallel, significantly speeding up
# the fitting procedure.

# The class relies on an underlying implementation of a gp_emulator which you must
# have installed on your system. I used Jose Gomez-Dans' original implementation,
# but the interface for Sinan's code should be idential. However, I note that Sinan's
# code did not use the most efficient method for forming the covariance matrix, which
# takes up the bulk of the time for fitting each emulator. Therefore, I suggest
# using Jose's code for the time being, as it should give satisfactory performance
# for small problems

from multiprocessing import Pool
import numpy as np
from gp_emulator.GaussianProcess import GaussianProcess

class MultiOutputGP(object):
    """
    Implementation of a multiple-output Gaussian Process Emulator.
    """
    def __init__(self, inputs, targets):
        """
        Create a new multi-output GP Emulator
        """
        
        # check input types and shapes, reshape as appropriate for the case of a single emulator
        inputs = np.array(inputs)
        targets = np.array(targets)
        if len(targets.shape) == 1:
            targets = np.reshape(targets, (1, len(targets)))
        elif not (len(targets.shape) == 2):
            raise ValueError("targets must be either a 1D or 2D array")
        if not (len(inputs.shape) == 2):
            raise ValueError("inputs must be 2D array")
        if not (inputs.shape[0] == targets.shape[1]):
            raise ValueError("the first dimension of inputs must be the same length as the second dimension of targets (or first if targets is 1D))")

        self.emulators = [ GaussianProcess(inputs, single_target) for single_target in targets]
        
        self.n_emulators = targets.shape[0]
        self.n = inputs.shape[0]
        self.D = inputs.shape[1]
        
    def get_n_emulators(self):
        """
        Returns the number of emulators
        """
        return self.n_emulators
        
    def get_n(self):
        """
        Returns number of training examples in each emulator
        """
        return self.n
        
    def get_D(self):
        """
        Returns number of inputs for each emulator
        """
        return self.D
        
    def learn_hyperparameters(self, n_tries=15, verbose=False, x0=None, processes=None):
        """
        Fit hyperparameters for each model
        """
        
        assert int(n_tries) > 0, "n_tries must be a positive integer"
        if not x0 is None:
            x0 = np.array(x0)
            assert len(x0) == self.D + 2, "x0 must have length of number of input parameters D + 2"
        if not processes is None:
            processes = int(processes)
            assert processes > 0, "number of processes must be positive"
        
        n_tries = int(n_tries)
        
        p = Pool(processes)
        likelihood_theta_vals = p.starmap(GaussianProcess.learn_hyperparameters, [(gp, n_tries, verbose, x0) for gp in self.emulators])
        
        # re-evaluate log likelihood for each emulator to update current parameter values
        # (needed because of how multiprocessing works -- the bulk of the work is done in
        # parallel, but this step ensures that the results are gathered correctly for each
        # emulator)
        for emulator, (loglike, theta) in zip(self.emulators, likelihood_theta_vals):
            _ = emulator.loglikelihood(theta)
        return likelihood_theta_vals
        
    def predict(self, testing, do_deriv=True, do_unc=True, processes=None):
        """
        Make a prediction for a set of input vectors
        """
        testing = np.array(testing)
        assert len(testing.shape) == 2, "testing must be a 2D array"
        assert testing.shape[1] == self.D, "second dimension of testing must be the same as the number of input parameters"
        if not processes is None:
            processes = int(processes)
            assert processes > 0, "number of processes must be a positive integer"
            
        p = Pool(processes)
        
        predict_vals = p.starmap(GaussianProcess.predict, [(gp, testing) for gp in self.emulators])
        
        # repackage predictions into numpy arrays
        
        predict_unpacked, unc_unpacked, deriv_unpacked = [np.array(t) for t in zip(*predict_vals)]
        
        if not do_unc:
            unc_unpacked = None
        if not do_deriv:
            deriv_unpacked = None
            
        return predict_unpacked, unc_unpacked, deriv_unpacked
        
    def __str__(self):
        """
        Returns a string representation of the model
        """
        return ("Multi-Output Gaussian Process with:\n"+
                 str(self.get_n_emulators())+" emulators\n"+
                 str(self.get_n())+" training examples\n"+
                 str(self.get_D())+" input variables")
        