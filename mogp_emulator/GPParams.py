import numpy as np

class GPParams:
    """
    Class representing parameters for a GaussianProcess object
    
    This class serves as a wrapper to a numpy array holding the parameters for a ``GaussianProcess``
    object. Because the parameters for a GP are transformed in different ways (depending on
    convention and positivity constraints), the raw parameter values are often fairly
    opaque. This class provides more clarity on the raw and transformed parameter values
    used in a particular GP.
    """
    
    def __init__(self, n_mean=0, n_corr=1, nugget=True, data=None):
        """
        Create a new parameters object, optionally holding a set of parameter values
        If no data provided, data will be ``None`` to distingush GPs that have not
        yet been fit from those that have.
        """
        assert n_mean >= 0, "Number of mean parameters must be nonnegative"
        self.n_mean = n_mean
        assert n_corr >= 1, "Number of correlation parameters must be positive"
        self.n_corr = n_corr
        self.n_cov = 1
        if nugget:
            self.n_nugget = 1
        else:
            self.n_nugget = 0
        if data is None:
            self.data = None
        else:
            data = np.array(data)
            assert data.shape == (self.n_mean + self.n_corr + self.n_cov + self.n_nugget,), "Bad shape for data in GPParams"
            self.data = np.copy(data)

    @property
    def n_params(self):
        "Total number of parameters"
        return self.n_mean + self.n_corr + self.n_cov + self.n_nugget
        
    @property
    def mean(self):
        if self.data is None:
            return None
        else:
            return self.data[:self.n_mean]
        
    @mean.setter
    def mean(self, new_mean):
        if self.data is None:
            raise ValueError("Must initialize parameters before setting individual values")
        new_mean = np.reshape(np.array(new_mean), (-1,))
        assert new_mean.shape == (self.n_mean,), "Bad shape for new mean parameters"
        self.data[:self.n_mean] = np.copy(new_mean)

    @property
    def corr_raw(self):
        if self.data is None:
            return None
        else:
            return self.data[self.n_mean:(self.n_mean+self.n_corr)]
        
    @corr_raw.setter
    def corr_raw(self, new_corr):
        if self.data is None:
            raise ValueError("Must initialize parameters before setting individual values")
        new_corr = np.reshape(np.array(new_corr), (-1,))
        assert new_corr.shape == (self.n_corr,), "Bad shape for new correlation lengths; expected array of length {}".format(self.n_corr)
        self.data[self.n_mean:(self.n_mean+self.n_corr)] = new_corr

    @property
    def corr(self):
        if self.data is None:
            return None
        else:
            return np.exp(-2.*self.corr_raw)
        
    @corr.setter
    def corr(self, new_corr):
        new_corr = np.array(new_corr)
        assert np.all(new_corr > 0.), "Correlation parameters must all be positive"
        self.corr_raw = -0.5*np.log(new_corr)

    @property
    def cov_raw(self):
        if self.data is None:
            return None
        else:
            return self.data[(self.n_mean+self.n_corr):(self.n_mean+self.n_corr+1)]
    
    @cov_raw.setter
    def cov_raw(self, new_cov):
        if self.data is None:
            raise ValueError("Must initialize parameters before setting individual values")
        new_cov = np.reshape(np.array(new_cov), (-1,))
        assert new_cov.shape == (1,), "New covariance value must be a float or array of length 1"
        self.data[(self.n_mean+self.n_corr):(self.n_mean+self.n_corr+1)] = np.copy(new_cov)

    @property
    def cov(self):
        if self.data is None:
            return None
        else:
            return np.exp(self.cov_raw)
        
    @cov.setter
    def cov(self, new_cov):
        new_cov = np.reshape(np.array(new_cov), (-1,))
        assert new_cov[0] > 0., "Covariance parameter must be positive"
        self.cov_raw = np.log(new_cov)

    @property
    def nugget_raw(self):
        if self.data is None:
            return None
        elif self.n_nugget == 0:
            return np.array([])
        else:
            return self.data[-1]

    @nugget_raw.setter
    def nugget_raw(self, new_nugget):
        if self.data is None:
            raise ValueError("Must initialize parameters before setting individual values")
        if self.n_nugget == 0:
            raise AttributeError("GPParams object does not include a nugget parameter")
        new_nugget = np.reshape(np.array(new_nugget), (-1,))
        assert new_nugget.shape == (1,), "New nugget value must be a float or array of length 1"
        self.data[-1] = new_nugget[0]

    @property
    def nugget(self):
        if self.data is None:
            return None
        else:
            return np.exp(self.nugget_raw)
        
    @nugget.setter
    def nugget(self, new_nugget):
        new_nugget = np.reshape(np.array(new_nugget), (-1,))
        assert new_nugget[0] >= 0., "New nugget value must be non-negative" 
        self.nugget_raw = np.log(new_nugget)

    def get_data(self):
        return self.data

    def set_data(self, new_params):
        if new_params is None:
            self.data = None
        else:
            new_params = np.array(new_params)
            assert new_params.shape == (self.n_params,), "Bad shape for new data; expected {} parameters".format(self.n_params)
            self.data = np.copy(new_params)
        
    def same_shape(self, other):
        "Test if two GPParams objects have the same shape"
        
        if isinstance(other, np.ndarray):
            return other.shape == (self.n_params,)
        elif isinstance(other, GPParams):
            return (self.n_mean == other.n_mean and
                    self.n_corr == other.n_corr and
                    self.n_nugget == other.n_nugget and
                    self.n_params == other.n_params)
        else:
            raise ValueError("other must be a numpy array or another GPParams object in GPParams.same_shape")

    def __str__(self):
        outstr = "GPParams with:"
        if self.data is None:
            outstr += " data = None"
        else:
            outstr += "\nmean = {}".format(self.mean)
            outstr += "\ncorrelation = {}".format(self.corr)
            outstr += "\ncovariance = {}".format(self.cov)
            if self.n_nugget == 1:
                outstr += "\nnugget = {}".format(self.nugget)
        return outstr