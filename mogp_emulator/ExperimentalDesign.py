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
        raise NotImplementedError
        
    def draw_sample(self, n_samples):
        "draw a set of n_samples from the experiment according to the given method"
        raise NotImplementedError
        
    def __str__(self):
        "returns a string representation of the ExperimentalDesign object"
        return "Experimental Design with "+str(self.get_n_parameters())+" parameters"