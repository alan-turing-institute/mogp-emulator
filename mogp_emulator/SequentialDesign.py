import numpy as np
from inspect import signature
from .ExperimentalDesign import ExperimentalDesign

class SequentialDesign(object):
    "Base class for a sequential experimental design"
    def __init__(self, base_design, f = None, n_targets = 1, n_samples = None, n_init = 10, n_cand = 50):
        "create new instance of an experimental design"
        
        if not isinstance(base_design, ExperimentalDesign):
            raise TypeError("base design must be a one-shot experimental design")
        
        if not f == None:
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
        
        self.base_design = base_design
        self.f = f
        self.n_targets = int(n_targets)
        if n_samples == None:
            self.n_samples = None
        else:
            self.n_samples = int(n_samples)
        self.n_init = int(n_init)
        self.n_cand = int(n_cand)
        
        self.current_iteration = 0
        self.initialized = False
        self.inputs = None
        self.targets = None
        self.candidates = None
    
    def has_function(self):
        "Determines if class contains a function for running the simulator"
        return (not self.f == None)
    
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
        
        if len(np.atleast_1d(np.array(targets)).shape) == 1:
            targets = np.reshape(np.array(targets), (1, len(np.array(targets))))
            
        assert np.array(targets).shape == (self.n_targets, self.n_init), "initial targets must have shape (n_targets, n_init)"
        
        self.targets = np.array(targets)
        self.initialized = True
    
    def run_init_design(self):
        "run initial design"
        
        assert self.has_function(), "Design must have a bound function to use run_init_design"
        
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
        
        if self.inputs is None:
            raise ValueError("Initial design has not been generated")
        else:
            assert self.inputs.shape == (self.current_iteration, self.get_n_parameters()), "inputs have not been correctly updated"
        
        if self.targets is None:
            raise ValueError("Initial targets have not been generated")
        else:
            assert self.targets.shape == (self.n_targets, self.current_iteration), "targets have not been correctly updated"
        
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
        
        target = np.atleast_1d(np.array(target))
        target = np.reshape(target, (len(target),))
        assert target.shape == (self.n_targets,), "new target must have shape (n_targets,)"
        
        new_targets = np.empty((self.n_targets, self.current_iteration + 1))
        new_targets[:,:-1] = self.targets
        new_targets[:,-1] = np.array(target)
        
        self.targets = np.array(new_targets)
        self.current_iteration = self.current_iteration + 1
        
    def run_next_point(self):
        "do one iteration of the sequential design process"
        
        assert self.has_function(), "Design must have a bound function to use run_next_point"
        
        next_point = self.get_next_point()
        next_target = np.array(self.f(next_point))
        self.set_next_target(next_target)
    
    def run_sequential_design(self, n_samples = None):
        "run the entire sequential design"
        
        assert self.has_function(), "Design must have a bound function to use run_sequential_design"
        
        if n_samples is None and self.n_samples is None:
            raise ValueError("must specify n_samples either when initializing or calling run_sequential_design")
            
        if n_samples is None:
            n_iter = self.n_samples
        else:
            n_iter = n_samples
            
        self.run_init_design()
        
        for i in range(n_iter):
            self.run_next_point()
        
    def __str__(self):
        "returns string representation of design"
        
        output_string = ""
        output_string += type(self).__name__+" with\n"
        output_string += self.get_base_design()+" base design\n"
        if self.has_function():
            output_string += "a bound simulator function\n"
        output_string += str(self.get_n_targets())+" targets\n"
        output_string += str(self.get_n_samples())+" total samples\n"
        output_string += str(self.get_n_init())+" initial points\n"
        output_string += str(self.get_n_cand())+" candidate points\n"
        output_string += str(self.get_current_iteration())+" current samples\n"
        output_string += "current inputs: "+str(self.get_inputs())+"\n"
        output_string += "current targets: "+str(self.get_targets())
        
        return output_string
        
        
class MICEDesign(SequentialDesign):
    "class representing a MICE Sequential Design"
    def __init__(self, base_design, f = None, n_targets = 1, n_samples = None, n_init = 10, n_cand = 50,
                 nugget = None, nugget_s = 1.):
        "create new instance of a MICE sequential design"
        
        if not nugget == None:
            if nugget < 0.:
                raise ValueError("nugget parameter cannot be negative")
                
        if nugget_s < 0.:
            raise ValueError("nugget smoothing parameter cannot be negative")
        
        if nugget == None:
            self.nugget = nugget
        else:
            self.nugget = float(nugget)
        self.nugget_s = float(nugget_s)
        
        super().__init__(base_design, f, n_targets, n_samples, n_init, n_cand)
        
    def get_nugget(self):
        "get value of nugget parameter"
        return self.nugget
        
    def get_nugget_s(self):
        "get value of nugget_s parameter"
        return self.nugget_s
        
    def _eval_metric(self):
        "Evaluate MICE criterion on candidate points"
        pass