import numpy as np
from inspect import signature
from .ExperimentalDesign import ExperimentalDesign

class SequentialDesign(object):
    "Base class for a sequential experimental design"
    def __init__(self, base_design, f, n_targets = 1, n_samples = None, n_init = 10, n_cand = 50, nugget = 1.):
        "create new instance of an experimental design"
        
        if not isinstance(base_design, ExperimentalDesign):
            raise TypeError("base design must be a one-shot experimental design")
        
        if not callable(f):
            raise TypeError("simulator f must be a function or other callable")
            
        if not len(signature(f).parameters) == 1:
            raise ValueError("simulator f must accept all parameters as a single input array")
        
        if int(n_targets) < 0:
            raise ValueError("number of targets must be positive")
        
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
        self.n_targets = int(n_targets)
        if n_samples == None:
            self.n_samples = None
        else:
            self.n_samples = int(n_samples)
        self.n_init = int(n_init)
        self.n_cand = int(n_cand)
        self.nugget = float(nugget)
        
        self.current_iteration = 0
        self.initialized = False
        self.inputs = None
        self.targets = None
        self.candidates = None
    
    def get_n_targets(self):
        "get number of targets in design"
        
        return self.n_targets
    
    def get_n_parameters(self):
        "get number of parameters in design"
        
        return self.base_design.get_n_parameters()
    
    def get_n_init(self):
        "get number of initial design points"
        
        return self.n_init
        
    def get_n_samples(self):
        "get total number of samples"
        
        return self.n_samples
    
    def get_n_cand(self):
        "get number of candidate points"
        
        return self.n_cand
        
    def get_nugget(self):
        "get nugget parameter for smoothing predictions"
        
        return self.nugget
        
    def get_current_iteration(self):
        "get current iteration"
        
        return self.current_iteration
        
    def get_inputs(self):
        "get current design"
        
        return self.inputs
        
    def get_targets(self):
        "get current value of targets"
        
        return self.targets
        
    def get_candidates(self):
        "get current value of candidates"
        
        return self.candidates
        
    def get_base_design(self):
        "get type of base design"
        
        return type(self.base_design).__name__
    
    def generate_initial_design(self):
        "create initial design"
        
        self.inputs = self.base_design.sample(self.n_init)
        self.current_iteration = self.n_init
        return self.inputs
    
    def set_initial_targets(self, targets):
        "set initial targets"
        
        if self.inputs is None:
            raise ValueError("Initial design has not been generated")
        else:
            assert self.inputs.shape == (self.n_init, self.get_n_parameters()), "inputs have not been initialized correctly"
        
        if len(np.array(targets).shape) == 1:
            targets = np.reshape(np.array(targets), (1, len(np.array(targets))))
            
        assert np.array(targets).shape == (self.n_targets, self.n_init), "initial targets must have shape (n_targets, n_init)"
        
        self.targets = np.array(targets)
        self.initialized = True
    
    def run_init_design(self):
        "run initial design"
        
        inputs = self.generate_initial_design()
        targets = np.full((self.n_targets, self.n_init), np.nan)
        
        for i in range(self.n_init):
            targets[:,i] = np.array(self.f(inputs[i,:]))
            
        assert np.all(np.isfinite(targets)), "error in initializing sequential design, function outputs may not be the correct shape"
        self.set_initial_targets(targets)
        
    def _generate_candidates(self):
        "generate candidates for next iteration"
        
        self.candidates = self.base_design.sample(self.n_cand)
    
    def _eval_metric(self):
        "evaluate metric for selecting next point on candidates"
        raise NotImplementedError("Base class for Sequential Design does not implement an evaluation metric")
    
    def get_next_point(self):
        "evaluate candidates to determine next point"
        
        self._generate_candidates()
        next_index = self._eval_metric()
        
        new_inputs = np.empty((self.current_iteration + 1, self.get_n_parameters()))
        new_inputs[:-1, :] = self.inputs
        
        next_point = self.candidates[next_index,:]
        new_inputs[-1,:] = next_point
        
        self.inputs = np.array(new_inputs)
        
        return next_point
    
    def set_next_target(self, target):
        "set value of next target"
        
        if self.inputs is None:
            raise ValueError("Initial design has not been generated")
        else:
            assert self.inputs.shape == (self.current_iteration + 1, self.get_n_parameters()), "inputs have not been correctly updated"
        
        if self.targets is None:
            raise ValueError("Initial targets have not been generated")
        else:
            assert self.targets.shape == (self.n_targets, self.current_iteration), "targets have not been correctly updated"
        
        if isinstance(target, float):
            target = np.reshape(np.array(target), (1,))
        assert target.shape == (self.n_targets,), "new target must have shape (n_targets,)"
        
        new_targets = np.empty((self.n_targets, self.current_iteration + 1))
        new_targets[:,:-1] = self.targets
        new_targets[:,-1] = np.array(target)
        
        self.targets = np.array(new_targets)
        self.current_iteration = self.current_iteration + 1
    
    def run_sequential_design(self, n_samples = None):
        "run the entire sequential design"
        pass
        
    def __str__(self):
        "returns string representation of design"
        pass