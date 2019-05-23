import numpy as np
import scipy.stats
from inspect import signature

class ExperimentalDesign(object):
    "class representing a one-shot design of experiments with uncorrelated parameters"
    def __init__(self, *args):
        "create new instance of an experimental design"
        
        if len(args) == 1:
            try:
                n_parameters = int(args[0])
                bounds = None
            except TypeError:
                try:
                    n_parameters = len(list(args[0]))
                    bounds = list(args[0])
                except TypeError:
                    raise TypeError("bad input type for ExperimentalDesign")
        elif len(args) == 2:
            try:
                n_parameters = int(args[0])
            except TypeError:
                raise TypeError("bad input type for ExperimentalDesign")
                
            if callable(args[1]):
                bounds = args[1]
            else:
                try:
                    bounds = list(args[1])
                    try:
                        if (len(bounds) == 2 and isinstance(float(bounds[0]), float) and isinstance(float(bounds[1]), float)):
                            if float(bounds[1]) <= float(bounds[0]):
                                raise ValueError("bad value for parameter bounds in ExperimentalDesign")
                            bounds = (float(bounds[0]), float(bounds[1]))
                    except TypeError:
                        pass
                except TypeError:
                    raise TypeError("bad input type for ExperimentalDesign")
        else:
            raise ValueError("bad inputs for ExperimentalDesign")
                
        if n_parameters <= 0:
            raise ValueError("number of parameters must be positive in Experimental Design")

        self.n_parameters = n_parameters
        
        if bounds is None:
            self.distributions = [scipy.stats.uniform(loc = 0., scale = 1.).ppf]*n_parameters
        elif isinstance(bounds, tuple):
            self.distributions = [scipy.stats.uniform(loc = bounds[0], scale = bounds[1]-bounds[0]).ppf]*n_parameters
        elif callable(bounds):
            if len(signature(bounds).parameters) == 1:
                self.distributions = [bounds]*n_parameters
            else:
                raise ValueError("PPF distribution provided must accept a single argument")
        else: # bounds is a list
            if not len(bounds) == n_parameters:
                raise ValueError("list of parameter distributions must have the same length")
            self.distributions = []
            for item in bounds:
                if callable(item):
                    if len(signature(item).parameters) == 1:
                        self.distributions.append(item)
                    else:
                        raise ValueError("PPF distribution provided must accept a single argument")
                else:
                    try:
                        if (len(item) == 2 and isinstance(float(item[0]), float) and isinstance(float(item[1]), float)):
                            if float(item[1]) <= float(item[0]):
                                raise ValueError("bad value for parameter bounds in ExperimentalDesign")
                            self.distributions.append(scipy.stats.uniform(loc = float(item[0]), 
                                                                          scale = float(item[1])- float(item[0])).ppf)
                        else:
                            raise ValueError("bounds for each parameter must be a tuple of two floats")
                    except TypeError:
                        raise TypeError("bounds for each parameter must be a tuple of two floats")
            
        
    def get_n_parameters(self):
        "returns number of parameters"
        return self.n_parameters
        
    def get_method(self):
        "returns method"
        try:
            return self.method
        except AttributeError:
            raise NotImplementedError("base class of ExperimentalDesign does not implement a method")
    
    def _draw_samples(self, n_samples):
        "Low level method for drawing random samples (all outputs will be between 0 and 1)"
        raise NotImplementedError
    
    def sample(self, n_samples):
        """
        draw samples from a generic experimental design. low level method creates random numbers between
        0 and 1, while this method converts them into scaled parameter values using the ppfs
        """
        
        n_samples = int(n_samples)
        assert n_samples > 0, "number of samples must be positive"
        
        sample_values = np.zeros((n_samples, self.get_n_parameters()))
        random_draws = self._draw_samples(n_samples)
        
        for (dist, index) in zip(self.distributions, range(self.get_n_parameters())):
            try:
                sample_values[:,index] = dist(random_draws[:,index])
            except:
                for sample_index in range(n_samples):
                    sample_values[sample_index, index] = dist(random_draws[sample_index,index])
        
        assert np.all(np.isfinite(sample_values)), "error due to non-finite values of parameters"
        
        return sample_values
        
    def __str__(self):
        "returns a string representation of the ExperimentalDesign object"
        try:
            method = self.get_method()+" "
        except NotImplementedError:
            method = ""
        return method+"Experimental Design with "+str(self.get_n_parameters())+" parameters"
        
class MonteCarloDesign(ExperimentalDesign):
    "class representing an experimental design drawing uncorrelated parameters using Monte Carlo sampling"
    def __init__(self, *args):
        "initialize a monte carlo experimental design"
        
        self.method = "Monte Carlo"
        super().__init__(*args)
        
    def _draw_samples(self, n_samples):
        "draw a set of n_samples from the experiment according to the given method"
        
        n_samples = int(n_samples)
        assert n_samples > 0, "number of samples must be positive"
        
        return np.random.random((n_samples, self.get_n_parameters()))
        
        
class LatinHypercubeDesign(ExperimentalDesign):
    "class representing a Latin Hypercube Design"
    def __init__(self, *args):
        "initialize a latin hypercube experimental design"
        
        self.method = "Latin Hypercube"
        super().__init__(*args)
    
    def _draw_samples(self, n_samples):
        "low level method for drawing samples from a latin hypercube design"
        
        n_samples = int(n_samples)
        assert n_samples > 0, "number of samples must be positive"
        
        n_parameters = self.get_n_parameters()

        random_samples = np.reshape(np.tile(np.arange(n_samples, dtype = np.float)/float(n_samples), n_parameters),
                                            (n_parameters, n_samples))
                    
        for row in random_samples:
            np.random.shuffle(row)
            
        random_samples =  np.transpose(random_samples) + np.random.random((n_samples, n_parameters))/float(n_samples)
        
        assert np.all(random_samples >= 0.) and np.all(random_samples <= 1.), "error in generating latin hypercube samples"
        
        return random_samples