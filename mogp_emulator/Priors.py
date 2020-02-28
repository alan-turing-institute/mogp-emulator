import numpy as np
from scipy.special import gamma

class Prior(object):
    def logp(self, x):
        raise NotImplementedError

    def dlogpdtheta(self, x):
        raise NotImplementedError

    def d2logpdtheta2(self, x):
        return 0.

class NormalPrior(Prior):
    def __init__(self, mean, std):
        self.mean = mean
        assert std > 0., "scale must be positive"
        self.std = std

    def logp(self, x):
        return -0.5*((x - self.mean)/self.std)**2 - np.log(self.std) - 0.5*np.log(2.*np.pi)

    def dlogpdtheta(self, x):
        return -(x - self.mean)/self.std**2

class InvGammaPrior(Prior):
    def __init__(self, shape, scale):
        assert shape > 0., "shape parameter a must be positive"
        self.shape = shape
        assert scale > 0., "scale must be positive"
        self.scale = scale

    def logp(self, x):
        return (self.shape*np.log(self.scale) - np.log(gamma(self.shape)) -
                (self.shape + 1.)*x - self.scale*np.exp(-x))

    def dlogpdtheta(self, x):
        return -(self.shape + 1.) + self.scale*np.exp(-x)
