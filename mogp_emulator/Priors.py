import numpy as np
from scipy.optimize import root
from scipy.special import gammaln
from scipy.linalg import cho_factor, cho_solve
import scipy.stats
from mogp_emulator.GPParams import CovTransform, CorrTransform, GPParams
import warnings

class GPPriors(object):
    """
    Class representing prior distributions on GP Hyperparameters
    """
    def __init__(self, mean=None, corr=None, cov=None, nugget=None, n_corr=None, nugget_type="fit"):
        
        if corr is None and n_corr is None:
            raise ValueError("Must provide an argument for either corr or n_corr in GPPriors")
            
        self.mean = mean
        
        self._n_corr = n_corr
        self.corr = corr
        
        self.cov = cov
        
        assert nugget_type in ["fit", "adaptive", "fixed", "pivot"], "Bad value for nugget type in GPPriors"
        self.nugget_type = nugget_type
        self.nugget = nugget
        
    @classmethod
    def default_priors(cls, inputs, nugget_type="fit", dist="invgamma"):
        "create priors with defaults for correlation length priors and nugget"
        
        assert nugget_type in ["fit", "adaptive", "fixed", "pivot"], "Bad value for nugget type in GPPriors"
        
        if dist.lower() == "lognormal":
            dist_obj = LogNormalPrior
        elif dist.lower() == "gamma":
            dist_obj = GammaPrior
        elif dist.lower() == "invgamma":
            dist_obj = InvGammaPrior
        else:
            if not isinstance(dist, (LogNormalPrior, GammaPrior, InvGammaPrior)):
                raise TypeError("dist must be a prior distribution to contstruct default priors")
            dist_obj = dist
        
        priors = [dist_obj.default_prior_corr(param) for param in np.transpose(inputs)]
        
        priors_updated = [p if not p is None else InvGammaPrior.default_prior_corr_mode(param)
                          for (p, param) in zip(priors, np.transpose(inputs))]
        
        if nugget_type == "fit":
            nugget = InvGammaPrior.default_prior_nugget()
        else:
            nugget = None
            
        return cls(mean=None, corr=priors_updated, cov=None, nugget=nugget, nugget_type=nugget_type)
    
    @property
    def mean(self):
        "Mean is a MeanPriors object"
        return self._mean
        
    @mean.setter
    def mean(self, newmean):
        "Setter method for mean"
        
        if newmean is None:
            self._mean = MeanPriors()
        elif isinstance(newmean, MeanPriors):
            self._mean = newmean
        else:
            try:
                self._mean = MeanPriors(*newmean)
            except TypeError:
                raise ValueError("Bad value for defining a MeanPriors object in GPPriors, " +
                                 "argument must be an iterable containing the mean " +
                                 "vector and the covariance as a float/vector/matrix")
                                 
    @property
    def n_mean(self):
        """
        Number of mean parameters
        """
        return self.mean.n_params
        
    @property
    def corr(self):
        """
        Correlation Length Priors
        
        Must be a list of distributions/None. When class object is initialized, must
        either set number of correlation parameters explicitly or pass a list of
        prior objects. If only number of parameters, will generate a list of NoneTypes
        of that length (assumes weak prior information). If list provided, will use
        that and override the value of number of correlation parameters.
        
        Can change the length by setting this attribute. n_corr will automatically update.
        """
        return self._corr
        
    @corr.setter
    def corr(self, newcorr):
        if newcorr is None:
            newcorr = [None]*self.n_corr
        try:
            list(newcorr)
        except TypeError:
            raise TypeError("Correlation priors must be a list of PriorDist derived objects or None")
        assert len(newcorr) > 0, "Correlation priors must be a list of nonzero length"
        for d in newcorr:
            if not (d is None or issubclass(type(d), PriorDist)):
                raise TypeError("Correlation priors must be a list of PriorDist derived objects or None")
        self._corr = list(newcorr)
        self._n_corr = len(self._corr)
        
    @property
    def n_corr(self):
        return self._n_corr
    
    @property
    def cov(self):
        return self._cov
        
    @cov.setter
    def cov(self, newcov):
        if not (newcov is None or issubclass(type(newcov), PriorDist)):
            raise TypeError("Covariance prior must be a PriorDist derived object or None")
        self._cov = newcov
        
    @property
    def nugget(self):
        return self._nugget
        
    @nugget.setter
    def nugget(self, newnugget):
        if not (newnugget is None or issubclass(type(newnugget), PriorDist)):
            raise TypeError("Nugget prior must be a PriorDist derived object or None")
        if not newnugget is None and not self.nugget_type == "fit":
            print("Prior distributions only apply to fitted nuggets; setting nugget prior to None")
            newnugget = None
        self._nugget = newnugget
    
    def _check_theta(self, theta):
        
        if not isinstance(theta, GPParams):
            raise TypeError("theta must be a GPParams object when computing priors in GPPriors")
        assert self.n_corr == theta.n_corr, "Provided GPParams object does not have the correct number of parameters"
        if self.nugget_type == "fit":
            assert theta.n_nugget == 1, "Provided GPParams object does not have the correct number of parameters"
    
    def logp(self, theta):
        "Compute log probability given a GPParams object"
        
        self._check_theta(theta)
        
        logposterior = 0.
        
        for dist, val in zip(self._corr, theta.corr):
            if not dist is None:
                logposterior += dist.logp(val)
        
        if not self._cov is None:
            logposterior += self._cov.logp(theta.cov)
            
        if not self._nugget is None:
            logposterior += self._nugget.logp(theta.nugget)
                
        return logposterior
        
    def dlogpdtheta(self, theta):
        "Compute gradient of log probability given a GPParams object"
        
        self._check_theta(theta)
    
        partials = []
        
        for dist, val, raw in zip(self._corr, theta.corr, theta.corr_raw):
            if not dist is None:
                partials.append(dist.dlogpdtheta(val)*CorrTransform.dscaled_draw(raw))
            else:
                partials.append(0.)
        
        if not self._cov is None:
            partials.append(self._cov.dlogpdtheta(theta.cov)*
                            CovTransform.dscaled_draw(theta.cov_raw))
        else:
            partials.append(0.)
            
        if not self._nugget is None:
            partials.append(self._nugget.dlogpdtheta(theta.nugget)*
                            CovTransform.dscaled_draw(theta.nugget_raw))
        elif not self.nugget_type == "pivot":
            partials.append(0.)
                
        return np.array(partials)
        
    def d2logpdtheta2(self, theta):
        """
        Compute the second derivative of the log probability given a value of GPParams.
        """
        
        self._check_theta(theta)
        
        hessian = []
        
        for dist, val, raw in zip(self._corr, theta.corr, theta.corr_raw):
            if not dist is None:
                hessian.append(dist.d2logpdtheta2(val)*
                               CorrTransform.dscaled_draw(raw)**2 +
                               dist.dlogpdtheta(val)*
                               CorrTransform.d2scaled_draw2(raw))
            else:
                hessian.append(0.)
        
        if not self._cov is None:
            hessian.append(self._cov.d2logpdtheta2(theta.cov)*
                           CovTransform.dscaled_draw(theta.cov_raw)**2 +
                           self._cov.dlogpdtheta(theta.cov)*
                           CovTransform.d2scaled_draw2(theta.cov_raw))
        else:
            hessian.append(0.)
            
        if not self._nugget is None:
            hessian.append(self._nugget.d2logpdtheta2(theta.nugget)*
                           CovTransform.dscaled_draw(theta.nugget_raw)**2 +
                           self._nugget.dlogpdtheta(theta.nugget)*
                           CovTransform.d2scaled_draw2(theta.nugget_raw))
        elif not self.nugget_type == "pivot":
            hessian.append(0.)
                
        return np.array(hessian)
    
    def sample(self):
        """
        Draw a set of samples from the prior distributions associated with this GPPriors object.
        Used in fitting to initialize the minimization algorithm.
        """
        
        sample_pt = []
        
        for dist in self._corr:
            if dist is None:
                smpl = default_sampler()
            else:
                smpl = CorrTransform.inv_transform(dist.sample())
            sample_pt.append(smpl)

        if self._cov is None:
            smpl = default_sampler()
        else:
            smpl = CovTransform.inv_transform(self._cov.sample())
        sample_pt.append(smpl)
        
        if not self.nugget_type == "pivot":
            if self._nugget is None:
                smpl = default_sampler()
            else:
                smpl = CovTransform.inv_transform(self._nugget.sample())
            sample_pt.append(smpl)

        return np.array(sample_pt)
        
    def __str__(self):
        return str(self._priors)

class MeanPriors(object):
    """
    Object holding mean priors (mean vector and covariance float/vector/matrix
    assuming a multivariate normal distribution)
    """
    def __init__(self, mean=None, cov=None):
        if mean is None:
            self.mean = None
            if not cov is None:
                warnings.warn("Both mean and cov need to be set to form a valid nontrivial " +
                              "MeanPriors object. mean is not provided, so ignoring the " +
                              "provided cov.")
            self.cov = None
            self.Lb = None
        else:
            self.mean = np.reshape(np.array(mean), (-1,))
            if cov is None:
                raise ValueError("Both mean and cov need to be set to form a valid MeanPriors object")
            self.cov = np.array(cov)
            self.Lb = None
            if self.cov.ndim == 0:
                assert self.cov > 0., "covariance term must be greater than zero in MeanPriors"
            elif self.cov.ndim == 1:
                assert len(self.cov) == len(self.mean), "mean and variances must have the same length in MeanPriors"
                assert np.all(self.cov > 0.), "all variances must be greater than zero in MeanPriors"
            elif self.cov.ndim == 2:
                assert self.cov.shape[0] == len(self.mean), "mean and covariances must have the same shape in MeanPriors"
                assert self.cov.shape[1] == len(self.mean), "mean and covariances must have the same shape in MeanPriors"
                assert np.all(np.diag(self.cov) > 0.), "all covariances must be greater than zero in MeanPriors"
                self.Lb = cho_factor(self.cov)
            else:
                raise ValueError("Bad shape for the covariance in MeanPriors")
    
    @property
    def n_params(self):
        "Number of parameters associated with the mean"
        if self.mean is None:
            return 0
        else:
            return len(self.mean)
    
    def inv_cov(self):
        "compute the inverse of the covariance matrix"
        if self.cov is None:
            return 0.
        elif self.cov.ndim < 2:
            inv_cov = np.zeros((len(self.mean), len(self.mean)))
            np.fill_diagonal(inv_cov, np.broadcast_to(1./self.cov, (len(self.mean),)))
            return inv_cov
        else:
            return cho_solve(self.Lb, np.eye(len(self.mean)))
        
    def inv_cov_b(self):
        "compute the inverse of the covariance matrix times the mean vector"
        if self.cov is None:
            return 0.
        elif self.cov.ndim < 2:
            return self.mean/self.cov
        else:
            return cho_solve(self.Lb, self.mean)
            
    def log_det_cov(self):
        "Compute the log of the determinant of the covariance"
        if self.cov is None:
            return 0.
        elif self.cov.ndim < 2:
            return np.sum(np.log(np.broadcast_to(self.cov, (len(self.mean),))))
        else:
            return 2.*np.sum(np.log(np.diag(self.Lb[0])))
            
    def __str__(self):
        return "MeanPriors with mean = {} and cov = {}".format(self.mean, self.cov)            

class PriorDist(object):
    """
    Generic Prior Distribution Object
    """
    @classmethod
    def default_prior(cls, min_val, max_val):
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
    
        if cls == InvGammaPrior:
            dist_obj = scipy.stats.invgamma
        elif cls == GammaPrior:
            dist_obj = scipy.stats.gamma
        elif cls == LogNormalPrior:
            dist_obj = scipy.stats.lognorm
        else:
            raise ValueError("Default prior must be invgamma, gamma, or lognormal")
        
        assert min_val > 0., "min_val must be positive for InvGamma, Gamma, or LogNormal distributions"
        assert max_val > 0., "max_val must be positive for InvGamma, Gamma, or LogNormal distributions"
        
        assert min_val < max_val, "min_val must be less than max_val"
    
        def f(x):
            assert len(x) == 2
            cdf = dist_obj(np.exp(x[0]), scale=np.exp(x[1])).cdf
            return np.array([cdf(min_val) - 0.005, cdf(max_val) - 0.995])
        
        result = root(f, np.zeros(2))
    
        if not result["success"]:
            print("Prior solver failed to converge")
            return None
        else:
            return cls(np.exp(result["x"][0]), np.exp(result["x"][1]))

    @classmethod
    def default_prior_corr(cls, inputs):
        "Compute default priors on a set of inputs for the correlation length"
        
        min_val = min_spacing(inputs)
        max_val = max_spacing(inputs)
    
        if min_val == 0. or max_val == 0.:
            print("Too few unique inputs; defaulting to flat priors")
            return None
        
        return cls.default_prior(min_val, max_val)

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

    Admits input values from 0/+inf.

    Take two parameters: shape :math:`{\alpha}` and scale :math:`{\beta}`. Both must be positive,
    and they are defined such that

    :math:`{p(x) = \frac{\beta^{\alpha}x^{-\alpha - 1}}{\Gamma(/alpha)} \exp(-\beta/x)}`
    """
    @classmethod
    def default_prior_mode(cls, min_val, max_val):
        """
        Fits an inverse gamma with a mode that is
        the geometric mean of the min/max values and 99.5% of the mass is below the max value
        """
    
        assert min_val > 0.
        assert max_val > 0.
        assert min_val < max_val, "min_val must be less than max_val"
    
        mode = np.sqrt(min_val*max_val)

        def f(x):
            a = np.exp(x)
            return scipy.stats.invgamma(a, scale=(1. + a)*mode).cdf(max_val) - 0.995

        result = root(f, 0.)

        if not result["success"]:
            print("Prior solver failed to converge")
            return None
        else:
            a = np.exp(result["x"])
            return cls(a, scale=(1. + a)*mode)
    
    @classmethod
    def default_prior_corr_mode(cls, inputs):
        "Compute default priors on a set of inputs for the correlation length"
        
        min_val = min_spacing(inputs)
        max_val = max_spacing(inputs)
    
        if min_val == 0. or max_val == 0.:
            print("Too few unique inputs; defaulting to flat priors")
            return None
        
        return cls.default_prior_mode(min_val, max_val)
    
    @classmethod
    def default_prior_nugget(cls, min_val=1.e-8, max_val=1.e-6):
        "Compute a default prior for the nugget"
        return cls.default_prior_mode(min_val, max_val)
    
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
        """
        Draws a random sample from the distribution
        """
        return float(scipy.stats.invgamma.rvs(size=1, a=self.shape, scale=self.scale))

def default_sampler():
    """
    Generate a default random sample
    
    If no prior information is provided, this method is used to generate a
    starting value for the optimization routine. Any variable where
    prior information is available will use that to pick a random start
    point drawn from the prior distribution. If that is not available,
    this method is called to produce a random draw from a flat prior.
    
    :returns: Random number drawn from a flat prior (note that samples
              from this method are *not* transformed).
    :rtype: float
    """
    
    return float(5.*(np.random.rand() - 0.5))


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
