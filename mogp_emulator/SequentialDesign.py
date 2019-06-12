import numpy as np
from inspect import signature
from .ExperimentalDesign import ExperimentalDesign

class SequentialDesign(object):
    "Base class for a sequential experimental design"
    def __init__(self, base_design, f, n_samples = None, n_init = 10, n_cand = 50, nugget = 1.):
        "create new instance of an experimental design"
        
        if not isinstance(base_design, ExperimentalDesign):
            raise TypeError("base design must be a one-shot experimental design")
        
        if not callable(f):
            raise TypeError("simulator f must be a function or other callable")
            
        if not len(signature(f).parameters) == 1:
            raise ValueError("simulator f must accept all parameters as a single input array")
        
        if (not n_samples == None) and int(n_samples) <= 0:
            raise ValueError("number of samples must be positive")
                
        if int(n_init) <= 0:
            raise ValueError("number of initial design points must be positive")
            
        if (not n_samples == None) and int(n_samples) < int(n_init):
            raise ValueError("number of samples less than initial design size")
            
        if int(n_cand) <= 0:
            raise ValueError("number of candidate design points must be positive")
            
        if float(nugget) <= 0:
            raise ValueError("nugget smoothing parameter must be positive")
        
        self.base_design = base_design
        self.f = f
        if n_samples == None:
            self.n_samples = None
        else:
            self.n_samples = int(n_samples)
        self.n_init = n_init
        self.n_cand = n_cand
        self.nugget = nugget
        
        self.current_iteration = 0
        self.initialized = False
        self.inputs = None
        self.targets = None
        self.candidates = None
    
    def get_n_parameters(self):
        "get number of parameters in design"
        pass
    
    def get_n_init(self):
        "get number of initial design points"
        pass
        
    def get_n_samples(self):
        "get total number of samples"
        pass
    
    def get_n_cand(self):
        "get number of candidate points"
        pass
        
    def get_nugget(self):
        "get nugget parameter for smoothing predictions"
        pass
        
    def get_current_iteration(self):
        "get current iteration"
        pass
        
    def get_inputs(self):
        "get current design"
        pass
        
    def get_targets(self):
        "get current value of targets"
        pass
        
    def get_candidates(self):
        "get current value of candidates"
        pass
        
    def get_base_design(self):
        "get type of base design"
        pass
    
    def generate_initial_design(self):
        "create initial design"
        pass
    
    def run_init_design(self):
        "run initial design"
        pass
        
    def _generate_candidates(self):
        "generate candidates for next iteration"
        pass
    
    def _eval_criterion(self):
        "evaluate criterion for selecting next point on candidates"
        pass
    
    def get_next_point(self):
        "evaluate candidates to determine next point"
        pass
    
    def run_sequential_design(self, n_samples = None):
        "run the entire sequential design"
        pass
        
    def __str__(self):
        "returns string representation of design"
        pass