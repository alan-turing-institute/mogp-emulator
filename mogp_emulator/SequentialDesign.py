import numpy as np
from scipy.spatial.distance import cdist
from inspect import signature
from .ExperimentalDesign import ExperimentalDesign
from .GaussianProcess import GaussianProcess

class SequentialDesign(object):
    """
    Base class representing a sequential experimental design
    
    This class provides the base implementation of a class for designing experiments sequentially. This
    means that rather than picking all simulation points in a single step, the points are selected one
    by one, taking into account the information obtained by determining the true parameter value at each
    design point when selecting the next one. Sequential designs can be very useful when running expensive,
    high-dimensional simulations to ensure that a limited computational budget is used effectvely.
    
    Instead of choosing all points at once, which is the case in a one-shot design, a sequential design
    does some additional computation work at each step to more carefully choose the next point. This means
    that sequential designs are better suited for very expensive simulations, where the additional
    cost of choosing the next point is small compared to the overall computational cost of running
    the simulations.
    
    A sequential design is built on top of a base design (which must be a subclass of the
    ``ExperimentalDesign`` class. In addition to the base design, the class must contain information on
    how many points are used in the initial design (i.e. the number of starting points used before starting
    the sequential steps in the design) and the number of candidate points that are considered during each
    iteration. Optionally, a function for evaluating the actual simulation can be optionally bound to the
    class instance, which allows the entire design process to be automated. If such a function is not
    provided, then the steps to run the design must be carried out manually, with the evaluated
    simulation values provided to the class at the end of each simulation in order to determine the
    next point.
    
    To use the base class to create an experimental design, a new subclass must be created that provides
    a method ``_eval_metric``, which considers all candidate points and returns the index of the best
    candidate. Otherwise, all other code provided here allows for a generic sequential design to be
    easily run and managed.
    """
    def __init__(self, base_design, f = None, n_samples = None, n_init = 10, n_cand = 50):
        """
        Create a new instance of a sequential experimental design
        
        Creates a new instance of a sequential experimental design, which sequentially chooses
        points to be evaluated from a complex simulation function. It is often used for
        expensive computational models, where the cost of running a single evaluation is
        large and must be done in series due to computational limitations, and thus some
        additional computation done at each step to select new points is small compared
        to the overall cost of running a single simulation.
        
        Sequential designs require specifying a base design using a subclass of ``ExperimentalDesign``
        as well as information on the number of points to use in each step in the design
        process. Additionally, the function to evaluated can be bound to the class to allow
        automatic evaluation of the function at each step.
        
        :param base_design: Base one-shot experimental design (must be a subclass of
                            ``ExperimentalDesign``). This contains the information on the
                            parameter space to be sampled.
        :type base_design: ExperimentalDesign
        :param f: Function to be evaluated for the design. Must take all parameter values as a single
                  input array and return a single float or an array of length 1
        :type f: function or other callable
        :param n_samples: Number of sequential design points to be drawn. If specified, this must be
                          a positive integer. Note that this is in addition to the number of initial
                          points, meaning that the total design size will be ``n_samples + n_init``. 
                          This can also be specified when running the full design. This parameter is
                          optional, and defaults to ``None`` (meaning the number of samples is set when
                          running the design, or that samples will be added manually).
        :type n_samples: int or None
        :param n_init: Number of points in the inital design before the sequential steps begin. Must
                       be a positive integer. Optional, default value is 10.
        :type n_init: int
        :param n_cand: Number of candidates to consider at each sequential design step. Must be a positive
                       integer. Optional, default value is 50.
        """
        
        if not isinstance(base_design, ExperimentalDesign):
            raise TypeError("base design must be a one-shot experimental design")
        
        if not f == None:
            if not callable(f):
                raise TypeError("simulator f must be a function or other callable")
            
            if not len(signature(f).parameters) == 1:
                raise ValueError("simulator f must accept all parameters as a single input array")
        
        if (not n_samples == None) and int(n_samples) <= 0:
            raise ValueError("number of samples must be positive")
                
        if int(n_init) <= 0:
            raise ValueError("number of initial design points must be positive")
            
        if int(n_cand) <= 0:
            raise ValueError("number of candidate design points must be positive")
        
        self.base_design = base_design
        self.f = f
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
        """
        Determines if class contains a function for running the simulator
        
        This method checks to see if a function has been provided for running the simulation.
    
        :returns: Whether or not the design has a bound function for evaluting the simulation.
        :rtype: bool
        """
        return (not self.f == None)
    
    def get_n_parameters(self):
        """
        Get number of parameters in design
        
        Returns the number of parameters in the design (note that this is specified in the base
        design that must be provided when initializing the class instance).
        
        :returns: Number of parameters in the design
        :rtype: int
        """
    
        return self.base_design.get_n_parameters()
    
    def get_n_init(self):
        """
        Get number of initial design points
        
        Returns the number of initial design points used before beginning the sequential design
        steps. Note that this means that the total number of samples to be drawn for the design
        is ``n_init + n_samples``.
        
        :returns: Number of initial design points
        :rtype: int
        """
        
        return self.n_init
        
    def get_n_samples(self):
        """
        Get number of sequential design points
        
        Returns the number of sequential design points used in the sequential design steps. This
        parameter can be ``None`` to indicate that the number of samples will be specified when
        running the design, or that the samples will be updated manually. Note that the total number
        of samples to be drawn for the design is ``n_init + n_samples``.
        
        :returns: Number of sequential design points
        :rtype: int
        """
        
        return self.n_samples
    
    def get_n_cand(self):
        """
        Get number of candidate design points
        
        Returns the number of candidate design points used in each sequential design step. Candidates
        are re-drawn at each step, so this number of points will be drawn each time and all points
        will be considered at each iteration.
        
        :returns: Number of candidate design points
        :rtype: int
        """
        
        return self.n_cand
        
    def get_current_iteration(self):
        """
        Get number of current iteration in the experimental design
        
        Returns the current iteration during the sequential design process. This is mostly useful
        if the sequential design is being updated manually to know the current iteration.
        
        :returns: Current iteration number
        :rtype: int
        """
        
        return self.current_iteration
        
    def get_inputs(self):
        """
        Get current design input points
        
        Returns a numpy array holding the current design points. The array is 2D and has shape
        ``(current_iteration, n_parameters)`` (i.e. it is resized after each iteration when a new
        design point is chosen).
        
        :returns: Current value of the design inputs
        :rtype: ndarray
        """
        
        return self.inputs
        
    def get_targets(self):
        """
        Get current design target points
        
        Returns a numpy array holding the current target points. The array is 1D and has shape
        ``(current_iteration,)`` (i.e. it is resized after each iteration when a new target point
        is added). Note that simulation outputs must be a single number, so if considering a
        simulation has multiple outputs, the user must decide how to combine them to form the
        relevant target value for deciding which point to simulate next.
        
        :returns: Current value of the target inputs
        :rtype: ndarray
        """
        
        return self.targets
        
    def get_candidates(self):
        """
        Get current candidate design input points
        
        Returns a numpy array holding the current candidate design points. The array is 2D and
        has shape ``(n_cand, n_parameters)``. It always has the same size once it is initialized,
        but the values will change acros iterations as new candidate points are considered at
        each iteration.
        
        :returns: Current value of the candidate design inputs
        :rtype: ndarray
        """
        
        return self.candidates
        
    def get_base_design(self):
        """
        Get type of base design
        
        Returns the type of the base design. The base design must be a subclass of ``ExperimentalDesign``,
        but any one-shot design method can be used to generate the initial design and the candidates.
        
        :returns: Base design type as a string
        :rtype: str
        """
        
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
            targets = np.reshape(np.array(targets), (len(np.array(targets)),))
            
        assert np.array(targets).shape == (self.n_init,), "initial targets must have shape (n_init,)"
        
        self.targets = np.atleast_1d(targets)
        self.initialized = True
    
    def run_init_design(self):
        "run initial design"
        
        assert self.has_function(), "Design must have a bound function to use run_init_design"
        
        inputs = self.generate_initial_design()
        targets = np.full((self.n_init,), np.nan)
        
        for i in range(self.n_init):
            targets[i] = np.array(self.f(inputs[i,:]))
            
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
            assert self.targets.shape == (self.current_iteration,), "targets have not been correctly updated"
        
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
            assert self.targets.shape == (self.current_iteration,), "targets have not been correctly updated"
        
        target = np.atleast_1d(np.array(target))
        target = np.reshape(target, (len(target),))
        assert target.shape == (1,), "new target must have length 1"
        
        new_targets = np.empty((self.current_iteration + 1,))
        new_targets[:-1] = self.targets
        new_targets[-1] = np.array(target)
        
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
        output_string += str(self.get_n_samples())+" total samples\n"
        output_string += str(self.get_n_init())+" initial points\n"
        output_string += str(self.get_n_cand())+" candidate points\n"
        output_string += str(self.get_current_iteration())+" current samples\n"
        output_string += "current inputs: "+str(self.get_inputs())+"\n"
        output_string += "current targets: "+str(self.get_targets())
        
        return output_string
        

class MICEFastGP(GaussianProcess):
    "class implementing Woodbury identity for fast predictions"
    def fast_predict(self, index):
        """
        Make a corrected prediction for an input vector
        """
        
        index = int(index)
        assert index >= 0 and index < self.n, "index must be 0 <= index < n" 

        indices = (np.arange(self.n) != index)
        
        exp_theta = np.exp(self.theta)

        Ktest = cdist(np.sqrt(exp_theta[: (self.D)]) * np.reshape(self.inputs[indices,:], (self.n - 1, self.D)),
                      np.sqrt(exp_theta[: (self.D)]) * np.reshape(self.inputs[index, :], (1, self.D)), "sqeuclidean")

        Ktest = exp_theta[self.D] * np.exp(-0.5 * Ktest)
        
        invQ_mod = (self.invQ[indices][:, indices] -
                    1./self.invQ[index, index]*np.outer(self.invQ[indices, index], self.invQ[indices, index]))
        
        var = exp_theta[self.D] - np.sum(Ktest * np.dot(invQ_mod, Ktest), axis=0)
        
        return var

class MICEDesign(SequentialDesign):
    "class representing a MICE Sequential Design"
    def __init__(self, base_design, f = None, n_samples = None, n_init = 10, n_cand = 50,
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
        
        super().__init__(base_design, f, n_samples, n_init, n_cand)
        
    def get_nugget(self):
        "get value of nugget parameter"
        return self.nugget
        
    def get_nugget_s(self):
        "get value of nugget_s parameter"
        return self.nugget_s
    
    def _MICE_criterion(self, data_point):
        "compute MICE criterion for a single point"
        
        data_point = int(data_point)
        
        assert data_point >= 0 and data_point < self.n_cand, "test point index is out of range"
        
        _, unc1, _ = self.gp.predict(self.candidates[data_point], do_unc = True)
        unc2 = self.gp_fast.fast_predict(data_point)
        
        mice_criter =  unc1/unc2
        
        assert np.isfinite(mice_criter), "error in computing MICE critera"
        
        return float(mice_criter)
    
    def _eval_metric(self):
        "Evaluate MICE criterion on candidate points"
        
        self.gp = GaussianProcess(self.inputs, self.targets, self.nugget)
        self.gp.learn_hyperparameters()
        
        self.gp_fast = MICEFastGP(self.candidates, np.ones(self.n_cand), self.nugget_s)
        self.gp_fast._set_params(self.gp.current_theta)
        
        results = []
        
        for point in range(self.n_cand):
            results.append(self._MICE_criterion(point))
            
        return np.argmax(results)