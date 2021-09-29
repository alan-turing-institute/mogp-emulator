import numpy as np
        
class CorrTransform(object):
    r"Class representing correlation length transforms"
    @staticmethod
    def transform(r):
        "convert raw parameter to scaled"
        return np.exp(-0.5*r)
    @staticmethod
    def inv_transform(s):
        "convert scaled parameter to raw"
        return -2.*np.log(s)
    @staticmethod
    def dscaled_draw(s):
        return -0.5*s
    @staticmethod
    def d2scaled_draw2(s):
        return 0.25*s
    
class CovTransform(object):
    r"Class representing covariance/nugget transforms"
    @staticmethod
    def transform(r):
        "convert raw parameter to scaled"
        return np.exp(r)
    @staticmethod
    def inv_transform(s):
        "convert scaled parameter to raw"
        return np.log(s)
    @staticmethod
    def dscaled_draw(s):
        return s
    @staticmethod
    def d2scaled_draw2(s):
        return s

class GPParams(object):
    r"""
    Class representing parameters for a GaussianProcess object
    
    This class serves as a wrapper to a numpy array holding the parameters for a
    ``GaussianProcess`` object. Because the parameters for a GP are transformed
    in different ways (depending on convention and positivity constraints), the
    raw parameter values are often fairly opaque. This class provides more clarity
    on the raw and transformed parameter values used in a particular GP.
    
    The class is a wrapper around the numpy array holding the raw data values.
    When initializing a new ``GPParams`` object, the data can optionally be
    specified to initialize the array. Otherwise, a default value of ``None``
    will be used to indicate that the GP has not been fit.
    
    :param n_mean: The number of parameters in the mean function. Optional, default
                   is 0 (zero or fixed mean function). Must be a non-negative
                   integer.
    :type n_mean: int
    :param n_corr: The number of correlation length parameters. Optional, default
                   is 1. This must be the same as the number of inputs for a
                   particular GP. Must be a positive integer.
    :type n_corr: int
    :param nugget: Boolean that specifies if the emulator has a nugget parameter.
                   Optional, default is ``True``. The value of this depends on how
                   the nugget is fit for a particular GP (pivoted Cholesky
                   decomposition does not require a nugget, while all others do).
    :type nugget: bool
    :param data: Data to use for the parameters. Must be a 1-D numpy array with a
                 length of ``n_mean + n_corr + 1 + int(nugget)`` (i.e. the number
                 of parameters) or ``None``. Optional, default is ``None``
                 (indicates no parameter values have been set).
    :type data: ndarray or None
    
    To access scaled values of the parameters, use the provided attributes: ::
    
        >>> import numpy as np
        >>> from mogp_emulator.GPParams import GPParams
        >>> gpp = GPParams(n_mean=2, n_corr=2, nugget=True)
        >>> gpp.set_data(np.zeros(6))
        >>> gpp.mean
        array([0., 0.])
        >>> gpp.corr
        array([1., 1.])
        >>> gpp.cov
        1.
        >>> gpp.nugget
        1.0
        >>> gpp.set_data(np.ones(6))
        >>> gpp.mean
        array([1., 1.])
        >>> gpp.corr
        array([0.13533528, 0.13533528])
        >>> gpp.cov
        2.71828183
        >>> gpp.nugget
        2.718281828459045
    
    The transformations between the raw parameters :math:`\theta` and the
    transformed ones :math:`(\beta, l, \sigma^2, \eta^2)` are as follows:
    
    1. **Mean:** No transformation; :math:`{\beta = \theta}`
    2. **Correlation:** The raw values are transformed via
       :math:`{l = \exp(-0.5\theta)}` such that the transformed values are the
       correlation length associated with the given input.
    3. **Covariance:** The raw value is transformed via
       :math:`{\sigma^2 = \exp(\theta)}` so that the transformed value is
       the covariance.
    4. **Nugget:** The raw value is transformed via
       :math:`{\eta^2 = \exp(\theta)}` so that the transformed value is
       the variance associated with the nugget noise.
    """
    
    def __init__(self, n_mean=0, n_corr=1, nugget=True, mean_data=None, data=None):
        r"""
        Create a new parameters object, optionally holding a set of parameter values
        If no data provided, data will be ``None`` to distingush GPs that have not
        yet been fit from those that have. Note that if n_mean is zero, mean_data
        will be ignored even if an array is provided.
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
        
        if self.n_mean == 0:
            self.mean = np.array([])
        elif mean_data is None:
            self.mean = None
        else:
            self.mean = mean_data
        if data is None:
            self.data = None
        else:
            data = np.array(data)
            assert data.shape == (self.n_corr + self.n_cov + self.n_nugget,), "Bad shape for data in GPParams"
            self.data = np.copy(data)

    @property
    def n_params(self):
        r"""
        Total number of fitting parameters
        
        The ``n_params`` attribute gives the total number of fitting
        parameters, which is the length of the underlying numpy data
        array (all parameters save the mean function). Cannot be changed
        without initializing a new array.
        
        :returns: Number of parameters for this GP emulator
        :rtype int:
        """
        return self.n_corr + self.n_cov + self.n_nugget
        
    @property
    def mean(self):
        r"""
        Mean parameters
        
        The ``mean`` property returns the part of the data array associated with
        the mean function. Returns a numpy array of length ``(n_mean,)``
        or ``None`` if the mean data has not been initialized.
        
        Can be set with a new numpy array of the correct length.
        
        :returns: Numpy array holding the mean parameters
        :rtype: ndarray
        """
        return self.mean_data
        
    @mean.setter
    def mean(self, new_mean):
        if new_mean is None:
            self.mean_data = None
        else: 
            new_mean = np.reshape(np.array(new_mean), (-1,))
            assert new_mean.shape == (self.n_mean,), "Bad shape for new mean parameters"
            self.mean_data = np.copy(new_mean)

    @property
    def corr_raw(self):
        r"""
        Raw correlation length parameters
        
        The ``corr_raw`` property returns the part of the data array, without
        transformation, associated with the correlation lengths. Returns a
        numpy array of length ``(n_corr,)`` or ``None`` if the
        data array has not been initialized.
        
        Can be set with a new numpy array of the correct length. If the data
        array has not been initialized then setting individual parameter
        values cannot be done.
        
        :returns: Numpy array holding the raw correlation parameters
        :rtype: ndarray
        """
        if self.data is None:
            return None
        else:
            return self.data[:self.n_corr]
        
    @corr_raw.setter
    def corr_raw(self, new_corr):
        if self.data is None:
            raise ValueError("Must initialize parameters before setting individual values")
        new_corr = np.reshape(np.array(new_corr), (-1,))
        assert new_corr.shape == (self.n_corr,), "Bad shape for new correlation lengths; expected array of length {}".format(self.n_corr)
        self.data[:self.n_corr] = new_corr

    @property
    def corr(self):
        r"""
        Transformed correlation length parameters
        
        The ``corr`` property returns the part of the data array, with
        transformation, associated with the correlation lengths.
        Transformation is done via via :math:`{l = \exp(-0.5\theta)}`.
        Returns a numpy array of length ``(n_corr,)`` or ``None`` if the
        data array has not been initialized.
        
        Can be set with a new numpy array of the correct length. New
        parameters must satisfy the positivity constraint and 
        all be :math:`> 0`. If the data array has not been initialized,
        then setting individual parameter values cannot be done.
        
        :returns: Numpy array holding the transformed correlation parameters
        :rtype: ndarray
        """
        if self.data is None:
            return None
        else:
            return CorrTransform.transform(self.corr_raw)
        
    @corr.setter
    def corr(self, new_corr):
        new_corr = np.array(new_corr)
        assert np.all(new_corr > 0.), "Correlation parameters must all be positive"
        self.corr_raw = CorrTransform.inv_transform(new_corr)

    @property
    def cov_raw(self):
        r"""
        Raw covariance parameter
        
        The ``cov_raw`` property returns the covariance, without
        transformation. Returns a float or ``None`` if the data
        array has not been initialized.
        
        Can be set with a new float or numpy array of the correct
        length. If the data array has not been initialized, then
        setting individual parameter values cannot be done.
        
        :returns: Raw covariance parameter
        :rtype: float
        """
        if self.data is None:
            return None
        else:
            return self.data[self.n_corr:(self.n_corr+1)][0]
    
    @cov_raw.setter
    def cov_raw(self, new_cov):
        if self.data is None:
            raise ValueError("Must initialize parameters before setting individual values")
        new_cov = np.reshape(np.array(new_cov), (-1,))
        assert new_cov.shape == (1,), "New covariance value must be a float or array of length 1"
        self.data[self.n_corr:(self.n_corr+1)] = np.copy(new_cov)

    @property
    def cov(self):
        r"""
        Transformed covariance parameter
        
        The ``cov`` property returns the covariance transformed according to
        :math:`\sigma^2=\exp(\theta)`. Returns a float or ``None`` if the
        data array has not been initialized.
        
        Can be set with a new float :math:`>0` or numpy array of length 1
        holding a positive float. If the data array has not been initialized,
        then setting individual parameter values cannot be done.
        
        :returns: Transformed covariance parameter
        :rtype: float or None
        """
        if self.data is None:
            return None
        else:
            return CovTransform.transform(self.cov_raw)
        
    @cov.setter
    def cov(self, new_cov):
        new_cov = np.reshape(np.array(new_cov), (-1,))
        assert new_cov[0] > 0., "Covariance parameter must be positive"
        self.cov_raw = CovTransform.inv_transform(new_cov)

    @property
    def nugget_raw(self):
        r"""
        Raw nugget parameter
        
        The ``nugget_raw`` property returns the nugget, without
        transformation. Returns a float or ``None`` if the emulator
        parameters have not been initialized or if the emulator
        does not have a nugget.
        
        Can be set with a new float or numpy array of the correct length.
        If the data array has not been initialized or the emulator
        has no nugget, then the setter will raise an error.
        
        :returns: Raw nugget parameter
        :rtype: float or None
        """
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
        r"""
        Transformed nugget parameter
        
        The ``nugget`` property returns the nugget transformed via
        :math:`\eta^2=\exp(\theta)`. Returns a float or ``None`` if the
        emulator parameters have not been initialized or if the emulator
        does not have a nugget.
        
        Can be set with a new float :math:`>0` or numpy array of length 1.
        If the data array has not been initialized or the emulator has no
        nugget, then the nugget setter will return an error.
        
        :returns: Transformed nugget parameter
        :rtype: float or None
        """
        if self.data is None:
            return None
        else:
            return CovTransform.transform(self.nugget_raw)
        
    @nugget.setter
    def nugget(self, new_nugget):
        new_nugget = np.reshape(np.array(new_nugget), (-1,))
        assert new_nugget[0] >= 0., "New nugget value must be non-negative" 
        self.nugget_raw = CovTransform.inv_transform(new_nugget)

    def get_data(self):
        r"""
        Returns current value of raw parameters as a numpy array
        
        Provides access to the underlying data array for the ``GPParams``
        object. Returns a numpy array or ``None`` if the parameters
        of this object have not yet been set.
        
        :returns: Numpy array holding the raw parameter values
        :rtype: ndarray or None
        """
        return self.data

    def set_data(self, new_params):
        r"""
        Set a new value of the raw parameters
        
        Allows the data underlying the ``GPParams`` object to be set
        at once. Can also be used to reset the underlying data to
        ``None`` for the full array, indicating that no parameters
        have been set.
        
        :param new_params: New parameter values as a numpy array with shape
                           ``(n_params,)`` or None.
        :type new_params: ndarray or None
        :returns: None
        """
        if new_params is None:
            self.data = None
        else:
            new_params = np.array(new_params)
            assert self.same_shape(new_params), "Bad shape for new data; expected {} parameters".format(self.n_params)
            self.data = np.copy(new_params)
        
    def same_shape(self, other):
        """
        Test if two GPParams objects have the same shape
        
        Method to check if a new ``GPParams`` object or numpy array has the
        same length as the current object. If a numpy array, assumes
        that the array represents the fitting parameters and that the
        mean parameters will be handled separately. If a ``GPParams``
        object, it will also check if the number of mean parameters
        match.
        
        Returns a boolean if the number of mean (if a ``GPParams`` object
        only), correlation, and nugget parameters (for a numpy array or
        a ``GPParams`` object) are the same in both object.
        
        :param other: Additional instance of ``GPParams`` or ndarray to be
                      compared with the current one.
        :type other: GPParams or ndarray
        :returns: Boolean indicating if the underlying arrays have the same
                  shape.
        :rtype: bool
        """
        
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
        "Returns a string representation of the GPParams class"
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