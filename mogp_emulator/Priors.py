import numpy as np
from scipy.special import gamma

class GPPriors(object):
    """
    Class representing prior distributions on GP Hyperparameters
    
    Currently just a limited implementation of a list to remain compatible
    with the previous implementation. As new features are added to the
    GP class, this class will be fleshed out more.
    """
    def __init__(self, priors, n_params, n_mean, nugget_type):
        
        assert nugget_type in ["adaptive", "fixed", "pivot", "fit"]
        
        if priors is None:
            priors = []
        else:
            priors = list(priors)

        if not isinstance(priors, list):
            raise TypeError("priors must be a list of Prior-derived objects")

        if len(priors) == 0:
            priors = n_params*[None]

        if nugget_type in ["adaptive", "fixed"]:
            if len(priors) == n_params - 1:
                priors.append(None)

        if not len(priors) == n_params:
            raise ValueError("bad length for priors; must have length n_params")

        if nugget_type in ["adaptive", "fixed"]:
            if not priors[-1] is None:
                priors[-1] = None

        for p in priors:
            if not p is None and not issubclass(type(p), PriorDist):
                raise TypeError("priors must be a list of Prior-derived objects")

        self._priors = list(priors)
        assert n_mean >= 0
        self.n_mean = n_mean
    
    def __len__(self):
        return len(self._priors)
        
    def __getitem__(self, index):
        return self._priors[index]
        
    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if self.index < len(self._priors):
            self.index += 1 
            return self._priors[self.index - 1]
        else:
            raise StopIteration
            
    def __str__(self):
        return str(self._priors)

class PriorDist(object):
    """
    Generic Prior Distribution Object
    """
    def logp(self, x):
        """
        Computes log probability at a given value
        """
        raise NotImplementedError

    def dlogpdtheta(self, x):
        """
        Computes derivative of log probability at a given value
        """
        raise NotImplementedError

    def d2logpdtheta2(self, x):
        """
        Computes second derivative of log probability at a given value
        """
        raise NotImplementedError
        

class NormalPrior(PriorDist):
    """
    Normal Distribution Prior object

    Admits input values from -inf/+inf. 
    
    Thus, for mean function
    hyperparameters this produces a normal distribution with given mean and variance, and for
    covariance/nugget hyperparameters this produces a lognormal distribution with given log mean and
    variance.

    Take two parameters: mean and std. mean can take any numeric value, while std must be positive.
    """
    def __init__(self, mean, std):
        self.mean = mean
        assert std > 0., "std parameter must be positive"
        self.std = std

    def logp(self, x):
        """
        Computes log probability at a given value
        """
        return -0.5*((x - self.mean)/self.std)**2 - np.log(self.std) - 0.5*np.log(2.*np.pi)

    def dlogpdtheta(self, x):
        """
        Computes derivative of log probability at a given value
        """
        return -(x - self.mean)/self.std**2

    def d2logpdtheta2(self, x):
        """
        Computes second derivative of log probability at a given value
        """
        return -self.std**(-2)

class GammaPrior(PriorDist):
    r"""
    Gamma Distribution Prior object

    Admits input values from -inf/+inf assumed on a logarithmic scale, and transforms by taking the
    exponential of the input. Thus, this is assumed to be appropriate for covariance hyperparameters
    where such transformations are assumed when computing the covariance function.

    Take two parameters: shape :math:`{\alpha}` and scale :math:`{\beta}`. Both must be positive,
    and they are defined such that

    :math:`{p(x) = \frac{\beta^{-\alpha}x^{\beta - 1}}{\Gamma(/alpha)} \exp(-x/\beta)}`
    """
    def __init__(self, shape, scale):
        assert shape > 0., "shape parameter must be positive"
        self.shape = shape
        assert scale > 0., "scale parameter must be positive"
        self.scale = scale

    def logp(self, x):
        """
        Computes log probability at a given value
        """
        return (-self.shape*np.log(self.scale) - np.log(gamma(self.shape)) +
                (self.shape - 1.)*x - np.exp(x)/self.scale)

    def dlogpdtheta(self, x):
        """
        Computes derivative of log probability at a given value
        """
        return (self.shape - 1.) - np.exp(x)/self.scale

    def d2logpdtheta2(self, x):
        """
        Computes second derivative of log probability at a given value
        """
        return -np.exp(x)/self.scale

class InvGammaPrior(PriorDist):
    r"""
    Inverse Gamma Distribution Prior object

    Admits input values from -inf/+inf assumed on a logarithmic scale, and transforms by taking the
    exponential of the input. Thus, this is assumed to be appropriate for covariance hyperparameters
    where such transformations are assumed when computing the covariance function.

    Take two parameters: shape :math:`{\alpha}` and scale :math:`{\beta}`. Both must be positive,
    and they are defined such that

    :math:`{p(x) = \frac{\beta^{\alpha}x^{-\beta - 1}}{\Gamma(/alpha)} \exp(-\beta/x)}`
    """
    def __init__(self, shape, scale):
        assert shape > 0., "shape parameter must be positive"
        self.shape = shape
        assert scale > 0., "scale parameter must be positive"
        self.scale = scale

    def logp(self, x):
        """
        Computes log probability at a given value
        """
        return (self.shape*np.log(self.scale) - np.log(gamma(self.shape)) -
                (self.shape + 1.)*x - self.scale*np.exp(-x))

    def dlogpdtheta(self, x):
        """
        Computes derivative of log probability at a given value
        """
        return -(self.shape + 1.) + self.scale*np.exp(-x)

    def d2logpdtheta2(self, x):
        """
        Computes second derivative of log probability at a given value
        """
        return -self.scale*np.exp(-x)
