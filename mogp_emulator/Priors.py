import numpy as np
from scipy.optimize import root
from scipy.special import gammaln
import scipy.stats
from mogp_emulator.GPParams import CovTransform, CorrTransform

class GPPriors(object):
    """
    Class representing prior distributions on GP Hyperparameters
    
    Currently just a limited implementation of a list to remain compatible
    with the previous implementation. As new features are added to the
    GP class, this class will be fleshed out more.
    """
    def __init__(self, priors, n_params, n_mean, nugget_type):
        
        assert nugget_type in ["adaptive", "fixed", "pivot", "fit"]
        
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
        if nugget_type in ["fit", "adaptive", "fixed"]:
            self.has_nugget = True
        else:
            self.has_nugget = False
        self.n_corr = n_params - self.n_mean - 1 - int(self.has_nugget)
        
    @classmethod
    def default_priors(cls, inputs, n_mean, nugget_type, dist="invgamma"):
        "create priors with defaults for correlation length priors"
        
        assert n_mean >= 0
        
        priors = [None]*n_mean
        
        for param in np.transpose(inputs):
            priors.append(default_prior_corr(param, dist))
            
        priors.append(None)
        
        if nugget_type in ["fit", "adaptive", "fixed"]:
            priors.append(default_prior_nugget())
            
        return cls(priors, len(priors), n_mean, nugget_type)
    
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
            
    def logp(self, theta):
        
        logposterior = 0.
        
        for i in range(len(self._priors)):
            if not self._priors[i] is None:
                priorarg, priorderiv, prior2deriv = self.transform(theta.data[i], i)
                logposterior += self._priors[i].logp(priorarg)
                
        return logposterior
        
    def dlogpdtheta(self, theta):
        
        partials = np.zeros(len(self._priors))
        
        for i in range(len(self._priors)):
            if not self._priors[i] is None:
                priorarg, priorderiv, prior2deriv = self.transform(theta.data[i], i)
                partials[i] = self._priors[i].dlogpdtheta(priorarg)*priorderiv
                
        return partials
        
    def d2logpdtheta2(self, theta):
        
        hessian = np.zeros(len(self._priors))
        
        for i in range(len(self._priors)):
            if not self._priors[i] is None:
                priorarg, priorderiv, prior2deriv = self.transform(theta.data[i], i)
                hessian[i] = (self._priors[i].d2logpdtheta2(priorarg)*priorderiv**2 +
                              self._priors[i].dlogpdtheta(priorarg)*prior2deriv)
                              
        return hessian
    
    def transform(self, param, i):
        if i < self.n_mean:
            priorarg = param
            priorderiv = 1.
            prior2deriv = 0.
        elif i >= self.n_mean and i < self.n_mean + self.n_corr:
            priorarg = CorrTransform.transform(param)
            priorderiv = CorrTransform.dscaled_draw(param)
            prior2deriv = CorrTransform.d2scaled_draw2(param)
        else:
            priorarg = CovTransform.transform(param)
            priorderiv = CovTransform.dscaled_draw(param)
            prior2deriv = CovTransform.d2scaled_draw2(param)
            
        return priorarg, priorderiv, prior2deriv
    
    def inv_transform(self, priorarg, i):
        
        if i < self.n_mean:
            param = priorarg
        elif i >= self.n_mean and i < self.n_mean + self.n_corr:
            param = CorrTransform.inv_transform(priorarg)
        else:
            param = CovTransform.inv_transform(priorarg)
        
        return param
    
    def sample(self):
        sample_pt = []
        for i in range(len(self._priors)):
            if self._priors[i] is None:
                priorarg = default_sampler()
            else:
                priorarg = self.inv_transform(self._priors[i].sample(), i)
            sample_pt.append(priorarg)
        return np.array(sample_pt)
        
    def __str__(self):
        return str(self._priors)

def default_sampler():
    return float(5.*(np.random.rand() - 0.5))

def default_prior(min_val, max_val, dist="invgamma"):
    r"""
    Computes default priors given a min and max val between which
    99% of the mass should be found.
    
    Both min and max must be positive as the supported distributions
    are defined over :math:`[0, +\inf]`
    
    This stabilizes the solution, as it prevents the algorithm
    from getting stuck outside these ranges as the likelihood tends
    to be flat on those areas.
    
    Optionally, can change the distribution to be a lognormal or
    gamma distribution by specifying the ``dist`` argument.
    
    Note that the function assumes only a single input dimension is
    provided. Thus, any input array will be flattened before processing.
    
    If the root-finding algorithm fails, then the function will return
    ``None`` to revert to a flat prior.
    """
    
    if dist.lower() == "invgamma":
        dist_obj = scipy.stats.invgamma
        out_obj = InvGammaPrior
    elif dist.lower() == "gamma":
        dist_obj = scipy.stats.gamma
        out_obj = GammaPrior
    elif dist.lower() == "lognormal":
        dist_obj = scipy.stats.lognorm
        out_obj = LogNormalPrior
    else:
        raise ValueError("Bad value of distribution argument to default_prior; must be invgamma, gamma, or lognormal")
        
    if dist.lower() in ["invgamma", "gamma", "lognormal"]:
        assert min_val > 0., "min_val must be positive for InvGamma, Gamma, or LogNormal distributions"
        assert max_val > 0., "max_val must be positive for InvGamma, Gamma, or LogNormal distributions"
        
    assert min_val < max_val, "min_val must be less than max_val"
    
    def f(x):
        assert len(x) == 2
        cdf = dist_obj(np.exp(x[0]), scale=np.exp(x[1])).cdf
        return np.array([cdf(min_val) - 0.005, cdf(max_val) - 0.995])
        
    result = root(f, np.zeros(2))
    
    if not result["success"]:
        print("Default prior solver failed to converge; reverting to flat priors")
        return None
    else:
        return out_obj(np.exp(result["x"][0]), np.exp(result["x"][1]))

def max_spacing(input):
    """
    Computes the maximum spacing of a particular input
    """
    
    input = np.unique(np.array(input).flatten())
    
    if len(input) <= 1:
        return 0.
    
    input_sorted = np.sort(input)
    return input_sorted[-1] - input_sorted[0]

def min_spacing(input):
    """
    Computes the median spacing of a particular input
    """
    
    input = np.unique(np.array(input).flatten())
    
    if len(input) <= 2:
        return 0.
    
    return np.median(np.diff(np.sort(input)))

def default_prior_corr(inputs, dist="invgamma"):
    "Compute default priors on a set of inputs for the correlation length"
        
    min_val = min_spacing(inputs)
    max_val = max_spacing(inputs)
    
    if min_val == 0. or max_val == 0.:
        print("Too few unique inputs; defaulting to flat priors")
        return None
        
    return default_prior(min_val, max_val, dist)

def default_prior_nugget():
    return InvGammaPrior(shape=1., scale=1.e-8)

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
        
    def sample(self):
        raise NotImplementedError
        

class NormalPrior(PriorDist):
    """
    Normal Distribution Prior object

    Admits input values from -inf/+inf.

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
        
    def sample(self):
        return float(scipy.stats.norm.rvs(size=1, loc=self.mean, scale=self.std))

class LogNormalPrior(PriorDist):
    """
    Normal Distribution Prior object

    Admits input values from 0/+inf.

    Take two parameters: shape and scale, both of which must be positive
    """
    def __init__(self, shape, scale):
        assert shape > 0., "shape must be greater than zero"
        assert scale > 0., "scale must be greater than zero"
        self.shape = shape
        self.scale = scale
        
    def logp(self, x):
        assert x > 0
        return (-0.5*(np.log(x/self.scale)/self.shape)**2
                - 0.5*np.log(2.*np.pi) - np.log(x) - np.log(self.shape))
                
    def dlogpdtheta(self, x):
        assert x > 0.
        return -np.log(x/self.scale)/self.shape**2/x - 1./x
        
    def d2logpdtheta2(self, x):
        assert x > 0.
        return (-1./self.shape**2 + np.log(x/self.scale)/self.shape**2 + 1.)/x**2
        
    def sample(self):
        return float(scipy.stats.lognorm.rvs(size=1, s=self.shape, scale=self.scale))
    

class GammaPrior(PriorDist):
    r"""
    Gamma Distribution Prior object

    Admits input values from 0/+inf.

    Take two parameters: shape :math:`{\alpha}` and scale :math:`{\beta}`. Both must be positive,
    and they are defined such that

    :math:`{p(x) = \frac{\beta^{-\alpha}x^{\alpha - 1}}{\Gamma(/alpha)} \exp(-x/\beta)}`
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
        assert x > 0.
        return (-self.shape*np.log(self.scale) - gammaln(self.shape) +
                (self.shape - 1.)*np.log(x) - x/self.scale)

    def dlogpdtheta(self, x):
        """
        Computes derivative of log probability at a given value
        """
        assert x > 0.
        return (self.shape - 1.)/x - 1./self.scale

    def d2logpdtheta2(self, x):
        """
        Computes second derivative of log probability at a given value
        """
        assert x > 0.
        return -(self.shape - 1.)/x**2
        
    def sample(self):
        return float(scipy.stats.gamma.rvs(size=1, a=self.shape, scale=self.scale))

class InvGammaPrior(PriorDist):
    r"""
    Inverse Gamma Distribution Prior object

    Admits input values from -inf/+inf assumed on a logarithmic scale, and transforms by taking the
    exponential of the input. Thus, this is assumed to be appropriate for covariance hyperparameters
    where such transformations are assumed when computing the covariance function.

    Take two parameters: shape :math:`{\alpha}` and scale :math:`{\beta}`. Both must be positive,
    and they are defined such that

    :math:`{p(x) = \frac{\beta^{\alpha}x^{-\alpha - 1}}{\Gamma(/alpha)} \exp(-\beta/x)}`
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
        return (self.shape*np.log(self.scale) - gammaln(self.shape) -
                (self.shape + 1.)*np.log(x) - self.scale/x)

    def dlogpdtheta(self, x):
        """
        Computes derivative of log probability at a given value
        """
        return -(self.shape + 1.)/x + self.scale/x**2

    def d2logpdtheta2(self, x):
        """
        Computes second derivative of log probability at a given value
        """
        return (self.shape + 1)/x**2 - 2.*self.scale/x**3
        
    def sample(self):
        return float(scipy.stats.invgamma.rvs(size=1, a=self.shape, scale=self.scale))
