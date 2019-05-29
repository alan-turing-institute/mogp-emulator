import numpy as np
import scipy.stats
from inspect import signature

class ExperimentalDesign(object):
    """
    Base class representing a generic one-shot design of experiments with uncorrelated parameters
    
    This class provides the base implementation for a class for designing experiments to sample
    the parameter space of a complex model. The parameter space can be specified in a variety
    of ways, but essentially the user must provide a Probability Point Function (PPF, or inverse 
    of the Cumulative Distribution Function) for each input parameter. Each PPF function takes a 
    single numeric input and maps from the interval :math:`[0,1]` to the desired parameter
    distribution value for a given parameter, and each parameter has a separate function describing 
    its distribution. Note that this makes the assumption of no correlations between any of the
    parameter values (a future version may implement an experimental design where there are such
    parameter correlations). Once the design is initialized, a desired number of samples can be
    drawn from the design, returning an array holding the desired number of samples from the
    parameter space.
    
    Internally, the class holds the set of PPFs for all of the parameter values, and samples are
    drawn by calling the ``sample`` method. To draw the samples, a specific method ``_draw_samples``
    must be defined that generates a series of points in the :math:`[0,1]^n` hypercube, where
    :math:`n` is the number of paramters. This set of samples from the hypercube is then mapped to
    the parameter space using the given PPF functions. Thus, defining a new design protocol only
    requires defining a new ``_draw_samples`` method and redefining the ``__init__`` method to set
    the internal ``method`` attribute. By default, no ``_draw_samples`` method is defined, so the
    base ``ExperimentalDesign`` class is only intended to be used to define new protocols (trying 
    to sample from an ``ExperimentalDesign`` instance will return a ``NotImplementedError``).
    """
    def __init__(self, *args):
        """
        Create a new instance of an experimental design
        
        Creates a new instance of a design of experiments, which draws samples from the parameter
        space of a complex model. It is often used to generate data for a Gaussian Process emulator
        to fit the outputs of the complex model. This is a base class that does not implement the
        method for sampling from the distribution; to use an experimental design in practice you
        should use one of the derived classes provided or create your own.
        
        The experimental design can be initialized in several ways depending on the arguments
        provided, ranging from the simplest to the most complicated.
        
        1. Provide an integer ``n`` indicating the number of input parameters. If this is used to
           create an instance, it is assumed that all parameters are unformly distributed over
           the :math:`n`-dimensional hypercube.
        2. Provide an integer ``n`` and a tuple ``(a, b)`` of length 2 containing two numeric values
           (where :math:`a < b`). In this case, all parameters are assumed to be uniformly distributed
           over the interval :math:`[a,b]`.
        3. Provide an integer ``n`` and a function that takes a single numeric input in the interval
           :math:`[0,1]` and maps it to the parameter space. In this case, all parameters are assumed
           to follow the provided Probability Point Function.
        4. Provide a list of tuples of length 2 containing numeric values (as above, the first number
           must smaller than the second number). The design then assumes that the number of parameters
           is the length of the list, and each parameter follows a uniform distribution with the bounds
           given by the respective tuple in the given list.
        5. Provide a list of functions taking a single input (as above, each function must map the
           interval :math:`[0,1]` to the parameter space). The number of parameters in the design is the
           length of the list, and the given PPF functions define the parameter space for each input.
        
        More concretely, if one input parameter is given, you may initilize the class in any of the
        following ways:
        
        :param n_parameters: Integer specifying the number of parameters (must be positive). The
                             design will sample each parameter over the interval :math:`[0,1]`
        :type n_parameters: int
        
        or
        
        :param bounds_list: List of tuples containing two numeric values, each of which has the
                           smaller number first. Each parameter then takes a uniform distribution
                           with bounds given by each tuple.
        :type bounds_list: list
        
        or
        
        :param ppf_list: List of functions or other callable, each of which accepts one argument
                         and maps the interval :math:`[0,1]` to the parameter space. Each parameter
                         follows the distribution given by the respective PPF function.
        :type ppf_list: list
        
        and if two input parameters are given:
        
        :param n_parameters: Integer specifying the number of parameters (must be positive). The
                             design will sample each parameter over the interval :math:`[0,1]`
        :type n_parameters: int
        :param bounds: Tuple or other iterable containing two numeric values, where the smaller
                       number must come first. Each parameter then takes a uniform distribution
                       with bounds given by the numbers provided in the tuple.
        :type bounds: tuple
        
        or
        
        :param n_parameters: Integer specifying the number of parameters (must be positive). The
                             design will sample each parameter over the interval :math:`[0,1]`
        :type n_parameters: int
        :param ppf: Function or other callable, which accepts one argument and maps the interval
                    :math:`[0,1]` to the parameter space. Each parameter follows the distribution
                    given by the PPF function.
        :type ppf: function
        
        The ``scipy.stats`` package provides implementations of a wide range of distributions, with
        pre-defined PPF functions. See the Scipy user manual for more details. Note that in order to
        get a compatible PPF function that only takes a single input, you will need to set any parameters
        needed to define the distribution.
        
        Internally, the class defines any PPF functions based on the input data and collects all of
        the PPF functions in a list. The class also contains information on the method used to draw
        samples from the design.
        
        To create a usable implementation based on this class, the user must define the method
        ``_draw_samples``, which takes a positive integer input ``n_samples`` and draws ``n_samples``
        from the :math:`[0,1]^n` hypercube, where :math:`n` is the number of parameters. The user must
        also modify the ``method`` attribute of the design in order to have the ``__str__`` method work
        correctly. All other functionality should not require any changes from the base class.
        """
        
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
        """
        Returns number of parameters in the experimental design
        
        This method returns the number of parameters in the experimental design. This is set when
        initializing the object, an cannot be modified.
        
        :returns: Number of parameters in the experimental design.
        :rtype: int
        """
        return self.n_parameters
        
    def get_method(self):
        """
        Returns the method used to draw samples from the design
        
        This method returns the method used to draw samples from the experimental design. The base
        class does not implement a method, so if you try to call this on the base class the code
        will raise a ``NotImplementedError``. When deriving new designs from the base class,
        the method should be set when calling the ``__init__`` method.
        
        :returns: Method used to draw samples from the design.
        :rtype: str
        """
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
        
        assert np.all(random_draws >= 0.) and np.all(random_draws <= 1.), "error in generating random samples"
        
        for (dist, index) in zip(self.distributions, range(self.get_n_parameters())):
            try:
                sample_values[:,index] = dist(random_draws[:,index])
            except:
                for sample_index in range(n_samples):
                    sample_values[sample_index, index] = dist(random_draws[sample_index,index])
        
        assert np.all(np.isfinite(sample_values)), "error due to non-finite values of parameters"
        
        return sample_values
        
    def __str__(self):
        """
        Returns a string representation of the ExperimentalDesign object
        
        This method returns a string representation of the Experimental Design object. If a specific
        method is set for a derived class from the base class, the string will include this method
        in the string output. Otherwise, a generic string will be returned.
        
        :returns: String representation of the object
        :rtype: str
        """
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