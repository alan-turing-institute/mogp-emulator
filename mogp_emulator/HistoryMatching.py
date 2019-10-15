import numpy as np
from mogp_emulator import GaussianProcess

class HistoryMatching(object):
    r"""
    Class containing tools to implement history matching between the outputs of a
    gaussian process and an observation. 
  
    Provided methods:
      __init__ : constructor for the class. Sets up the various parameters that 
        will be used for further calculations, either as placeholders or with 
        default values. Can be called with any of the parameters that can be set 
        using the set_ methods as keyword arguments.
  
      get_implausibility : the primary use method for the class. Requires that an
        observation is provided, as well as either a set of expectations or both a
        GP and a set of coordinates for which expectations are to be computed. The
        variances of the expectations are included in the GP output, and the 
        variance of the observation can be included with it when it is set. Any 
        number of further variances can be given as arguments for 
        get_implausibility.
  
      get_NROY : returns a series of indices corresponding to entries in self.I 
        that are Not Ruled Out Yet by the specified observation and implausiblity
        threshold. If self.I has not been computed, calls get_implausiblity to do
        this. get_NROY also sets self.RO - the corresponding list of indices that
        HAVE been ruled out
  
      get_RO : returns a series of indices corresponding to entries in self.I 
        that are Not Ruled Out Yet by the specified observation and implausiblity
        threshold. If self.NROY and self.RO have not been computed, calls get_NROY
        to do this.
  
      set_gp : allows a gaussian process to be specified by setting the self.gp
        variable. 
  
      set_obs : allows an observation value, and, optionally, its associated 
        variance, to be specified by setting the self.obs variable.
  
      set_coords : allows a set of coordinates for which expectation values are to
        be calculated to be specified by setting the self.coords value.
  
      set_expectations : allows a set of expectation values (explicitely assumed 
        to be consistent with the output of a 
        mogp_emulator.GaussianProcess.predict() method).
  
      set_threshold : allows an implausibility threshold to be specified by 
        setting the self.threshold value. By default, this is set to 3.0. 
  
      status : prints a summary of the current status of the class object, 
        including the values stored in self.gp, .obs, .ndim, .ncoords, and 
        .threshold, and the shapes/sizes ofvalues stored in self.obs, .coords,
        .expectations, .I, .NROY, and .RO.
  
      check_gp : returns a boolean that is True if the provided quantity is 
        consistent with the requirements for a gaussian process, i.e. is of type
        GaussianProcess.
  
      check_obs : returns a boolean that is True if the provided quantity is 
        consistent with the requirements for the observation quantity, i.e. is a 
        numerical value that can be converted to a float, or a list of up to two 
        such values.
  
      check_coords : returns a boolean that is True if the provided quantity is 
        consistent with the requirements for the coordinates quantity, i.e. is a 
        list or ndarray of fewer than 3 dimensions.
  
      check_expectations : returns a boolean that is True if the provided quantity
        is consistent with the output of the predict method of a GaussianProcess
        object, i.e. that it is a tuple of length 3 containing dnarrays.
  
      check_threshold : returns a boolean that is True if the provided quantity is 
        consistent with the requirements for a threshold value, i.e. that it is a 
        single numerical value that can be converted to a float.
  
      check_ncoords : returns a boolean that is True if the provided quantity is 
        consistent with the requirements for a ncoords value, i.e. that it is a 
        single numerical value that can be converted to a float.
  
      update : checks that sufficient information exists to computen dim and/or 
        ncoords and, if so, computes these values. This also serves to update 
        these values if the information on which they are based is changed.
  
    Example - implausibility computation for a 1D GP:
    >>> import math
    >>> import numpy as np
    >>> from HistoryMatching import HistoryMatching
    >>> from mogp_emulator import GaussianProcess
    >>> 
    >>> # Set up a gaussian process
    >>> x_training = np.array([[0.],[10.],[20.],[30.],[43.],[50.]])
    >>> def get_y_training(x):
    >>>     n_points = len(x)
    >>>     f = np.zeros(n_points)
    >>>     for i in range(n_points):
    >>>         f[i] = np.sin(2*math.pi*x[i] / 50) 
    >>>     return f
    >>> y_training = get_y_training(x_training)
    >>> 
    >>> gp = GaussianProcess(x_training, y_training)
    >>> np.random.seed(47)
    >>> gp.learn_hyperparameters()
    >>> 
    >>> # Define the observation to which to compare
    >>> obs = [-0.8, 0.0004]
    >>> 
    >>> # Coordinates to predict
    >>> n_rand = 2000
    >>> coords = np.sort(np.random.rand(n_rand),axis=0)*(56)-3
    >>> coords = coords[:,None]
    >>>
    >>> # Generate Expectations
    >>> expectations = gp.predict(coords)
    >>> 
    >>> # Calculate implausibility
    >>> hm = HistoryMatching(obs=obs, expectations=expectations)
    >>> I = hm.get_implausibility()
  
    """
  
    def __init__(self, gp=None, obs=None, coords=None, expectations=None, threshold=3.):
        r"""
        Create a new instance of history matching.  
    
        The primary function of this method is to initialise the self.gp, .obs, 
        .coords, .expectations, .ndim, .ncoords, .threshold, .I, .NROY, and .RO 
        variables as placeholders. The optional keyword arguments can also be used 
        to set user-defined values for varius quantities in a single call to 
        HistoryMtching(). Setting sufficient of these to allow the number of 
        dimensions and/or number of coordinates to be computed also causes these to
        be set at this stage.
    
        Optional keyword arguments:
          GaussianProcess
          obs
          coords
          expectations
        """
    
        # Place-holder values for user-defined quantities that can be passed in
        # as args.
        self.gp = None
        self.obs = None
        self.coords = None
        self.expectations = None
          
        # Place-holder or default values for optional or derived quantities.
        self.ndim = None
        self.ncoords = None
        self.threshold = None
        self.I = None
        self.NROY = None
        self.RO = None
    
        # If suitable keyword arguments are provided, update the relevent 
        # variable(s)
        if self.check_gp(gp): self.set_gp(gp) 
        if self.check_obs(obs): self.set_obs(obs)
        if self.check_coords(coords): self.set_coords(coords)
        if self.check_expectations(expectations): self.set_expectations(expectations)
        if self.check_threshold(threshold): self.set_threshold(threshold)
    
        self.update()
    
    
    def get_implausibility(self, *args):
        r"""
        Carries out the implausibility calculation given by:
    
        LaTeX:
        I_i(\bar{x_0}) = \frac{|z_i - E(f_i(\bar{x_0}))|}{\sqrt{Var[z_i - E(f_i(\bar{x_0}))]}}
    
        to return an implausibility value for each of the provided coordinates.
    
        Requires that the observation parameter is set, and that at least one of the
        following conditions is met:
          a) The coords and gp parameters are set
          b) The expectations parameter is set
    
        All parameters to be passed into the function are assumed to be variances. 
        These can be in either of 2 forms:
          a) single values that correspond to variances on the observation or that
        are a constant across all expectation values.
          b) lists containing a variance parameter for each expectation value.
          
        As the implausibility calculation linearly sums variances, the result is
        agnostic to the precise provenance of any provided variances. Variances 
        may therefore be provided in any order.
    
        """
    
        # Confirm that observation parameter is set
        if not self.check_obs(self.obs):
            raise Exception(
              "implausibility calculation requires that the observation value is " + 
              "set. This can be done using the set_obs method.")
      
        # Check that we have exactly 1 valid combination of parameters
        UseCoordGP = False
        UseExpectations = False
        if (self.check_coords(self.coords) and self.check_gp(self.gp)):
            UseCoordGP = True
        if self.check_expectations(self.expectations):
            UseExpectations = True
        if UseCoordGP and UseExpectations:
            raise Exception(
              "Multiple valid parameter combinations are set. Previously set " +
              "parameters can be removed by setting them to None")
      
        # Confirm that the ncoords parameter is set
        if not self.check_ncoords(self.ncoords):
            raise Exception(
              "ncoords is not set despite a valid parameter combination being found.")
      
        # From this point on, the calculation works off the self.expectations value.
        # If this is not set, calculate it from the other parameters
        if UseCoordGP:
            self.expectations = self.gp.predict(self.coords)
      
        # Read in args, check for validity and construct iterable lists.
        varvals = []
        varlists = None 
    
        if len(args) > 0:
            for a in args:
                # For each argument, check whether it's a single value or a list/array 
                # of multiple values
                if isinstance(a, list):    # vars as list: convert to ndarray and adjoin
                    if len(a) != self.ncoords: 
                        raise Exception(
                          "bad input for get_implausibility - expected variance quantities"+
                          " containing 1 or", self.ncoords, 
                          "values, found quantities containing", len(a))
                    if varlists is None:
                        varlists = np.reshape(np.asarray(a), (-1,1))
                    else:
                        varlists = np.concatenate(
                          (varlists, np.reshape(np.asarray(a), (-1,1))), 
                          axis=1)
                elif isinstance(a, np.ndarray):    # vars as ndarray: reshape and adjoin
                    if len(a.shape) == 1:
                        a = np.reshape(a, (-1,1))
                    elif (len(a.shape) > 2 or a.shape[1] != 1):
                        raise Exception(
                          "bad input for get_implausibility - expected variance quantities"+
                          " as single numerical values, lists, or dnarrays of shape (n,) " +
                          "or (n,1), found dnarray of shape", a.shape)
                    elif a.shape[0] != self.ncoords:
                        raise Exception(
                          "bad input for get_implausibility - expected variance quantities"+
                          " containing 1 or", self.ncoords, 
                          "values, found quantities containing", a.shape[0])
                    if varlists is None:
                        varlists = a
                    else:
                        varlists = np.concatenate((varlists, a), axis=1)
                else:
                    try:                    # vars as individual values: append to varvals
                        varvals.append(float(a))
                    except:
                        raise Exception(
                          "bad input for get_implausibility - expected variance quantities"+
                          " as single numerical values, lists, or ndarrays, found variable"+
                          " of type", type(a))
        
        assert np.all(np.array(varvals) >= 0.), "all variances must be positive"
        if varlists is not None:
            assert np.all(np.array(varlists) >= 0.), "all variances must be positive"
        
        # Compute implausibility for each expectation value
        self.I = np.zeros(self.ncoords)
        for i, E in enumerate(self.expectations[0]):
            Vs = self.expectations[1][i]                  # variance on expectation
            Vs += sum(varvals)                            # fixed variance values
            Vs += self.obs[1]                             # variance on observation
            if varlists is not None:                      # individual variance values
                Vs += sum(varlists[i,:]) 
            self.I[i] = (abs(self.obs[0]-E) / np.sqrt(Vs))
      
        if UseCoordGP:
            self.expectations = None
      
        return self.I    
          
    
    def get_NROY(self, *args):
        r"""
        Returns a list of indices for self.I that correspond to entries that are not
        yet ruled out.
        """
        if self.I is None: self.get_implausibility(args)
    
        self.NROY = []
        self.RO = []
        for i, I in enumerate(self.I):
            if I <= self.threshold:
                self.NROY.append(i)
            else:
                self.RO.append(i)
            
        return self.NROY
    
    
    def get_RO(self, *args):
        r"""
        Returns a list of indices for self.I that correspond to entries that are 
        ruled out.
        """
        if self.RO is None: self.get_NROY(args)
    
        return self.RO
    
    
    def set_gp(self, gp):
        r"""
        Sets the self.gp variable to the provided gp argument if it passes the 
        check_gp requirements.
        """
        if not self.check_gp(gp): 
            raise TypeError("bad input for set_gp - expects a GaussianProcess object.")
        self.gp = gp
              
    
    def set_obs(self, obs):
        r"""
        Sets the self.obs variable to the provided obs argument if it passes the 
        check_obs requirements.
        """
        if not self.check_obs(obs):
            raise TypeError("bad input for set_obs")
        if isinstance(obs, list):
            if len(obs)==1:
                self.obs = [float(obs[0]), 0.]
            else:
                self.obs = [float(a) for a in obs]
        else:
            self.obs = [float(obs), 0.]
      
      
    def set_coords(self, coords):
        r"""
        Sets the self.coords variable to the provided coords argument if it passes 
        the check_coords requirements.
        """
        # need to allow coords == None, as otherwise can't reset values
        if not self.check_coords(coords) and (not coords == None):
            raise TypeError(
              "bad input for set_coords - expected coords in the form of a list or " +
              "1D or 2D ndarray of numerical values")
        if isinstance(coords, np.ndarray):
            if len(coords.shape) == 1:
                self.coords = np.reshape(coords, [-1,1])
            elif len(coords.shape) == 2:
                self.coords = coords
            else:
                raise Exception(
                  "error in exception handling - an ndarray of >2 dimensions somehow " +
                  "got through check_coords")
        elif isinstance(coords, list):
            self.coords = np.reshape(np.asarray(coords), [-1,1])
        elif coords == None:
            self.coords = None
        else:
            raise Exception(
              "error in exception handling - an argument of illegal type somehow " +
              "got through check_coords")
        self.update()
    
      
    def set_expectations(self, expectations):
        r"""
        Sets the self.expectations variable to the provided expectations argument if
        it passes the check_expectations requirements.
        """
        # need to allow expectations == None, as otherwise can't reset values
        if not self.check_expectations(expectations) and (not expectations == None):
            raise TypeError(
              "bad input for set_expectations - expected a Tuple of 3 ndarrays.")
        self.expectations = expectations
        self.update()
    
    
    def set_threshold(self, threshold):
        r"""
        Sets the self.threshold variable to the provided threshold argument if it 
        passes the check_threshold requirements.
        """
        if not self.check_threshold(threshold):
            raise TypeError("bad input for set_expectations - expected a float")
        self.threshold = float(threshold)
    
    
    def status(self):
        r"""
        Prints a summary of the current status of the class object.
        
        Prints a summary of the current status of the class object, including the 
        values stored in self.gp, .obs, .ndim, .ncoords, and .threshold, and the 
        shapes/sizes ofvalues stored in self.obs, .coords, .expectations, .I, .NROY,
        and .RO.
        """
    
        print(str(self))
    
    
    def check_gp(self, gp):
        r"""
        Checks if the provided argument is consistent with expectations for a GP.
    
        Returns a boolean that is True if the provided quantity is consistent with 
        the requirements for a gaussian process, i.e. is of type GaussianProcess.
        """
        if gp is None: return False
        if isinstance(gp, GaussianProcess): return True
        return False
              
    
    def check_obs(self, obs):
        r"""
        Checks if the provided argument is consistent with expectations for 
        observations.
    
        Returns a boolean that is True if the provided quantity is consistent with 
        the requirements for the observation quantity, i.e. is a numerical value 
        that can be converted to a float, or a list of up to two such values.
        """
        # TODO: allow tuple arguments as well as lists.
        if obs is None: return False
        if isinstance(obs, list):
            if len(obs) > 2: 
                raise ValueError(
                  "bad input type for HistoryMatching - the specified " + 
                  "observation parameter cannot contain more than 2 entries "+
                  "(value, [variance])")
            if len(obs) <= 2:
                try:
                    test = [float(a) for a in obs]
                except TypeError:
                    raise TypeError(
                      "bad input type for HistoryMatching - the specified " + 
                      "observation parameter must contain only numerical " +
                      "values")
            if len(obs) == 2:
                assert float(obs[1]) >= 0., "variance in observations cannot be negative"
            return True
        else:
            try:
                test = float(obs)
            except (TypeError, ValueError):
                raise TypeError(
                  "bad input type for HistoryMatching - the specified " + 
                  "observation parameter must contain only numerical values")
            return True
        return False
    
    
    def check_coords(self, coords):
        r"""
        Checks if the provided argument is consistent with expectations for 
        coordinates.
    
        Returns a boolean that is True if the provided quantity is consistent with 
        the requirements for the coordinates quantity, i.e. is a list or ndarray of 
        fewer than 3 dimensions.
        """
        # TODO: implement check that arrays or lists contain only numerical values
        if coords is None: return False
    
        if isinstance(coords, list): return True
        if isinstance(coords, np.ndarray):
            if len(coords.shape)<=2: return True
        return False
    
          
    def check_expectations(self, expectations):
        r"""
        Checks if the provided argument is consistent with expectations for 
        Gaussian Process Expectations.
    
        Returns a boolean that is True if the provided quantity is consistent with 
        the output of the predict method of a GaussianProcess object, i.e. that it 
        is a tuple of length 3 containing dnarrays.
        """
        if expectations is None: return False
        if not isinstance(expectations, tuple):
            return False
        if len(expectations)!=3:
            return False
        if not all ((
          isinstance(expectations[0], np.ndarray), 
          isinstance(expectations[1], np.ndarray),
          isinstance(expectations[2], np.ndarray)
        )):
            raise TypeError(
              "bad input type for HistoryMatching - expected expectation values in " +
              "the form of a Tuple of ndarrays.")
        assert np.all(expectations[1] >= 0.), "all variances must be nonnegative"
        return True
    
    
    def check_threshold(self, threshold):
        r"""
        Checks if the provided argument is consistent with expectations for a 
        threshold value.
    
        Returns a boolean that is True if the provided quantity is consistent with 
        the requirements for a threshold value, i.e. that it is a single numerical 
        value that can be converted to a float.
        """
        if threshold is None: return False
        try:
            test = float(threshold)
            assert test >= 0., "threshold must be non-negative"
            return True
        except TypeError:
            return False
      
      
    def check_ncoords(self, ncoords):
        r"""
        Checks if the provided argument is consistent with expectations for the 
        number of coordinates for which expectation values are to be / have been
        computed.
    
        Returns a boolean that is True if the provided quantity is consistent with 
        the requirements for a ncoords value, i.e. that it is a single numerical 
        value that can be converted to a float.
        """
        if ncoords is None: return False
        try:
            test = float(ncoords)
            return True
        except TypeError:
            return False
      
      
    def update(self):
        r"""
        Checks that sufficient information exists to computen dim and/or ncoords 
        and, if so, computes these values. This also serves to update these values 
        if the information on which they are based is changed.
        """
    
        if self.check_coords(self.coords):
            self.ndim = self.coords.shape[1]
            self.ncoords = self.coords.shape[0]
        elif self.check_expectations(self.expectations):
            self.ncoords = self.expectations[0].shape[0]
        

    def __str__(self):
        r"""
        Returns string representation of HistoryMatching object
        """

        if self.coords is None: 
            coordstr = None
        else: 
            coordstr = self.coords.shape
      
        if self.expectations is None: expstr = None
        else: expstr = (
          str(type(self.expectations)) + " of len " + str(len(self.expectations)) + 
          " containing arrays of shape: " + str(self.expectations[0].shape))
    
        if self.I is None: Istr = None
        elif isinstance(self.I, np.ndarray): Istr = (
          str(type(self.I)) + " of shape " + str(self.I.shape)
        )
    
        if self.NROY is None: NROYstr = None
        elif isinstance(self.NROY, list): NROYstr = (
          str(type(self.NROY)) + " of length " + len(self.NROY)
        )
    
        if self.RO is None: ROstr = None
        elif isinstance(self.RO, list): ROstr = (
          str(type(self.RO)) + " of length " + len(self.RO)
        )
    
        return (
          "History Matching tools created with:\n" + 
          "Gaussian Process: {}\n" +
          "Observations: {}\n" +
          "Coords: {}\n" +
          "Expectations: {}\n" +
          "No. of Input Dimensions: {}\n" +
          "No. of Descrete Expectation Values: {}\n" +
          "I_threshold: {}\n" + 
          "I: {}\n" +
          "NROY: {}\n" + 
          "RO: {}\n").format(self.gp, self.obs, coordstr, expstr,
                             self.ndim, self.ncoords, self.threshold,
                             Istr, NROYstr, ROstr)