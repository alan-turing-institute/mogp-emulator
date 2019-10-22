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
        using the ``set_`` methods as keyword arguments.

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

      set_expectations : allows a set of expectation values (explicitly assumed
        to be consistent with the output of a
        ``mogp_emulator.GaussianProcess.predict()`` method).

      set_threshold : allows an implausibility threshold to be specified by
        setting the self.threshold value. By default, this is set to 3.0.

      status : prints a summary of the current status of the class object,
        including the values stored in self.gp, .obs, .ndim, .ncoords, and
        .threshold, and the shapes/sizes of values stored in self.obs, .coords,
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

    Example - implausibility computation for a 1D GP::

        import math
        import numpy as np
        from HistoryMatching import HistoryMatching
        from mogp_emulator import GaussianProcess

        # Set up a gaussian process
        x_training = np.array([[0.],[10.],[20.],[30.],[43.],[50.]])
        def get_y_training(x):
            n_points = len(x)
            f = np.zeros(n_points)
            for i in range(n_points):
                f[i] = np.sin(2*math.pi*x[i] / 50)
            return f
        y_training = get_y_training(x_training)

        gp = GaussianProcess(x_training, y_training)
        np.random.seed(47)
        gp.learn_hyperparameters()

        # Define the observation to which to compare
        obs = [-0.8, 0.0004]

        # Coordinates to predict
        n_rand = 2000
        coords = np.sort(np.random.rand(n_rand),axis=0)*(56)-3
        coords = coords[:,None]

        # Generate Expectations
        expectations = gp.predict(coords)

        # Calculate implausibility
        hm = HistoryMatching(obs=obs, expectations=expectations)
        I = hm.get_implausibility()


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

        :param gp: (Optional) ``GaussianProcess`` object used to make predictions in
                   history matching. Optional, can instead provide predictions directly
                   via the ``expectations`` argument. Default is ``None``
        :type gp: GaussianProcess or None
        :param obs: (Optional) Observations against which the predictions will be
                    compared. If provided, must either be a float (assumes no
                    uncertainty in the observations) or a list of two floats
                    holding the observation and its variance. Required for history
                    matching, but can be provided when calling ``get_implausibility``.
                    Default is ``None``.
        :type obs: float, list, or None
        :param coords: (Optional) Inputs at which the emulator values will be computed
                       to compare the GP output to the observations. If provided, must
                       be a numpy array matching the input for the GP (must be a 2D
                       array with dimensions ``(n, D)`` where ``n`` is the number of
                       points to be considered and ``D`` is the number of parameters
                       for the GP). Only required if ``expectations`` is not provided.
                       Default is ``None``.
        :type coords: ndarray
        :param expectations: (Optional) Tuple of 3 numpy arrays of the form expected
                             from GP predictions. The first array must hold the predicted
                             mean at all query points, and the second array must hold
                             the variances from the GP predictions at all query points.
                             The third is not used, so can simply be a dummy array.
                             Can instead provide a GP object and the points to query
                             to have the predictions made internally. Default is None.
        :type expectations: tuple holding 3 ndarrays
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
        if self.check_gp(gp):
            self.set_gp(gp)
        if self.check_obs(obs):
            self.set_obs(obs)
        if self.check_coords(coords):
            self.set_coords(coords)
        if self.check_expectations(expectations):
            self.set_expectations(expectations)
        if self.check_threshold(threshold):
            self.set_threshold(threshold)

        self.update()


    def get_implausibility(self, *args):
        r"""
        Compute Implausibility measure for all query points

        Carries out the implausibility calculation given by:

        .. math::
            {I_i(\bar{x_0}) = \frac{|z_i - E(f_i(\bar{x_0}))|}
            {\sqrt{Var\left[z_i - E(f_i(\bar{x_0}))\right]}}}

        to return an implausibility value for each of the provided coordinates.

        Requires that the observation parameter is set, and that at least one of the
        following conditions are met:

          a) The coords and gp parameters are set
          b) The expectations parameter is set

        All parameters to be passed into the function are assumed to be variances.
        These can be in either of 2 forms:

          a) single values that correspond to variances on the observation or that
             are a constant across all expectation values.
          b) lists containing a variance parameter for each expectation value.

        These additional variances can be used to include measurement uncertainty in
        ``obs`` (though the ``obs`` object can also accept a measurement variance)
        or a more general model discrepancy that describes the prior beliefs regarding
        how well the model matches reality. In practice, the model discrepancy is
        essential to have predictions that are not overly confident, however it can
        be hard to estimate (see further discussion in Brynjarsdottir and O\'Hagan,
        2014).

        As the implausibility calculation linearly sums variances, the result is
        agnostic to the precise provenance of any provided variances. Variances
        may therefore be provided in any order.

        :param args: List of additional variances to include in the calculation.
                      These must all be positive floats.
        """

        # Confirm that observation parameter is set
        if not self.check_obs(self.obs):
            raise Exception("implausibility calculation requires that the observation value is " +
                            "set. This can be done using the set_obs method.")

        # Check that we have exactly 1 valid combination of parameters
        UseCoordGP = False
        UseExpectations = False
        if (self.check_coords(self.coords) and self.check_gp(self.gp)):
            UseCoordGP = True
        if self.check_expectations(self.expectations):
            UseExpectations = True
        if UseCoordGP and UseExpectations:
            raise Exception("Multiple valid parameter combinations are set. Previously set " +
                            "parameters can be removed by setting them to None")

        # Confirm that the ncoords parameter is set
        if not self.check_ncoords(self.ncoords):
            raise Exception("ncoords is not set despite a valid parameter combination "+
                            "being found.")

        # From this point on, the calculation works off the self.expectations value.
        # If this is not set, calculate it from the other parameters
        if UseCoordGP:
            self.expectations = self.gp.predict(self.coords)

        # Read in args, check for validity and construct iterable lists.
        varvals = []
        varlists = None

        for a in args:
            # For each argument, check whether it's a single value or a list/array
            # of multiple values
            if isinstance(a, list):    # vars as list: convert to ndarray and adjoin
                if len(a) != self.ncoords:
                    raise Exception("bad input for get_implausibility - expected variance" +
                                    "quantities containing 1 or" + str(self.ncoords) +
                                    "values, found quantities containing" + str(len(a)))
                if varlists is None:
                    varlists = np.reshape(np.asarray(a), (-1, 1))
                else:
                    varlists = np.concatenate((varlists, np.reshape(np.asarray(a), (-1, 1))),
                                              axis=1)
            elif isinstance(a, np.ndarray):    # vars as ndarray: reshape and adjoin
                if len(a.shape) == 1:
                    a = np.reshape(a, (-1, 1))
                elif (len(a.shape) > 2 or a.shape[1] != 1):
                    raise Exception("bad input for get_implausibility - expected variance " +
                                    "quantities as single numerical values, lists, or " +
                                    "ndarrays of shape (n,) or (n, 1), found ndarray of " +
                                    "shape" + str(a.shape))
                elif a.shape[0] != self.ncoords:
                    raise Exception("bad input for get_implausibility - expected variance " +
                                    "quantities containing 1 or" + str(self.ncoords)+
                                    "values, found quantities containing"+ str(a.shape[0]))
                if varlists is None:
                    varlists = a
                else:
                    varlists = np.concatenate((varlists, a), axis=1)
            else:
                try:                    # vars as individual values: append to varvals
                    varvals.append(float(a))
                except:
                    raise Exception("bad input for get_implausibility - expected variance " +
                                    "quantities as single numerical values, lists, or " +
                                    "ndarrays, found variable of type" + str(type(a)))

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
                Vs += sum(varlists[i, :])
            self.I[i] = (abs(self.obs[0]-E) / np.sqrt(Vs))

        if UseCoordGP:
            self.expectations = None

        return self.I


    def get_NROY(self, *args):
        r"""
        Return set of indices that are not yet ruled out

        Returns a list of indices for ``self.I`` that correspond to entries that are not
        yet ruled out. Points that are ruled out have an implausibility metric that
        exceeds the threshold (can be set when initializing history matching or using
        the ``set_threshold`` method). If the implausibility metric has not yet been
        computed for the desired points, it is calculated.

        :param args: Inputs for ``get_implausibility`` if it has not yet been computed.
        :returns: List of integer indices that have not yet been ruled out.
        :rtype: list
        """
        if self.I is None: self.get_implausibility(*args)

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
        Return set of indices that have been ruled out

        Returns a list of indices for ``self.I`` that correspond to entries that have
        been ruled out. Points that are ruled out have an implausibility metric that
        exceeds the threshold (can be set when initializing history matching or using
        the ``set_threshold`` method). If the implausibility metric has not yet been
        computed for the desired points, it is calculated.

        :param args: Inputs for ``get_implausibility`` if it has not yet been computed.
        :returns: List of integer indices that have been ruled out.
        :rtype: list
        """
        if self.RO is None:
            self.get_NROY(*args)

        return self.RO


    def set_gp(self, gp):
        r"""
        Set the Gaussian Process to use with history matching

        Sets the ``self.gp`` variable to the provided ``GaussianProcess`` argument.

        :param gp: ``GaussianProcess`` object to use for history matching.
        :type gp: GaussianProcess
        :returns: None
        """
        if not self.check_gp(gp):
            raise TypeError("bad input for set_gp - expects a GaussianProcess object.")
        self.gp = gp


    def set_obs(self, obs):
        r"""
        Set the observations to be used for history matching

        Sets the ``self.obs`` variable to the provided ``obs`` argument. The object
        must pass the ``check_obs`` requirements, meaning that it must be a
        float (assumes that the observation has no error associated with it) or a
        list containing two floats (representing an observation and its variance).
        Note that the variance must be non-negative.

        :param obs: Observations to be used for history matching. Must be a float
                    or a list of two floats.
        :type obs: float or list
        :returns: None
        """
        if not self.check_obs(obs):
            raise TypeError("bad input for set_obs")
        if isinstance(obs, list):
            if len(obs) == 1:
                self.obs = [float(obs[0]), 0.]
            else:
                self.obs = [float(a) for a in obs]
        else:
            self.obs = [float(obs), 0.]


    def set_coords(self, coords):
        r"""
        Set the query points to be used in history matching

        Sets the ``self.coords`` variable to the provided query points ``coords`` if
        it passes the ``check_coords`` requirements, or be ``None`` to remove a set
        of existing query points. ``coords`` must be a numpy array matching the
        inputs to the provided ``GaussianProcess`` object.

        :param coords: Numpy array holding query points (array with shape ``(n, D)``,
                       where ``n`` is the number of query points and ``D`` is the
                       number of inputs to the emulator), or ``None``
        :type coords: ndarray or None
        :returns: None
        """
        # need to allow coords == None, as otherwise can't reset values
        if not self.check_coords(coords) and (not coords is None):
            raise TypeError("bad input for set_coords - expected coords in the form " +
                            "of a list or 1D or 2D ndarray of numerical values")
        if isinstance(coords, np.ndarray):
            if len(coords.shape) == 1:
                self.coords = np.reshape(coords, [-1, 1])
            elif len(coords.shape) == 2:
                self.coords = coords
            else:
                raise Exception("error in exception handling - an ndarray of >2 " +
                                "dimensions somehow got through check_coords")
        elif isinstance(coords, list):
            self.coords = np.reshape(np.asarray(coords), [-1, 1])
        elif coords == None:
            self.coords = None
        else:
            raise Exception("error in exception handling - an argument of illegal " +
                            "type somehow got through check_coords")
        self.update()


    def set_expectations(self, expectations):
        r"""
        Set the expected output of the simulator to be used in history matching

        Sets the ``self.expectations`` variable to the provided ``expectations``
        argument if it passes the ``check_expectations`` requirements, or ``None``
        can be used to remove an existing set of expectations. Expectations
        must be a tuple of 3 numpy arrays, where the first holds the predicted
        means for all query points and the second holds the predicted variances
        for all query points. The third arrray is not used in the computation,
        but is included as it is an expected output of the GP ``predict`` method.

        :param expectations: GP predictions at all query points. Must be a tuple
                             of 3 numpy arrays, or None to remove existing
                             expectations.
        :type expectations: tuple of 3 ndarrays or None
        :returns: None
        """
        # need to allow expectations == None, as otherwise can't reset values
        if not self.check_expectations(expectations) and (not expectations == None):
            raise TypeError("bad input for set_expectations - expected a Tuple " +
                            "of 3 ndarrays.")
        self.expectations = expectations
        self.update()


    def set_threshold(self, threshold):
        r"""
        Set the threshold value for history matching

        Sets the ``self.threshold variable`` to the provided threshold argument
        if it  passes the ``check_threshold`` requirements. The threshold must
        be a non-negative float.

        :param threshold: New value for threshold, must be a non-negative float.
        :type threshold: float
        :returns: None
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
        and .RO. No inputs, no return value.

        :returns: None
        """

        print(str(self))


    def check_gp(self, gp):
        r"""
        Checks if the provided argument is consistent with expectations for a GP.

        Returns a boolean that is True if the provided quantity is consistent with
        the requirements for a gaussian process, i.e. is of type GaussianProcess.

        :param gp: Input GP object to be checked.
        :type gp: GaussianProcess
        :returns: Boolean indicating if provided object is a ``GaussianProcess``
        :rtype: bool
        """
        if gp is None: return False
        if isinstance(gp, GaussianProcess):
            return True
        return False


    def check_obs(self, obs):
        r"""
        Checks if the provided argument is consistent with expectations for
        observations.

        Returns a boolean that is ``True`` if the provided quantity is consistent with
        the requirements for the observation quantity, i.e. is a numerical value
        that can be converted to a float, or a list of up to two such values. Also
        checks if the provided variance is non-negative.

        :param obs: Input for observations to be checked
        :type obs: float or list
        :returns: Boolean indicating if provided observations are acceptable
        :rtype: bool
        """
        # TODO: allow tuple arguments as well as lists.
        if obs is None:
            return False
        if isinstance(obs, list):
            if len(obs) > 2:
                raise ValueError("bad input type for HistoryMatching - the specified " +
                                 "observation parameter cannot contain more than 2 entries "+
                                 "(value, [variance])")
            if len(obs) <= 2:
                try:
                    test = [float(a) for a in obs]
                except TypeError:
                    raise TypeError("bad input type for HistoryMatching - the specified " +
                                    "observation parameter must contain only numerical " +
                                    "values")
            if len(obs) == 2:
                assert float(obs[1]) >= 0., "variance in observations cannot be negative"
            return True
        else:
            try:
                test = float(obs)
            except (TypeError, ValueError):
                raise TypeError("bad input type for HistoryMatching - the specified " +
                                "observation parameter must contain only numerical values")
            return True
        return False


    def check_coords(self, coords):
        r"""
        Checks if the provided argument is consistent with expectations for
        coordinates.

        Returns a boolean that is ``True`` if the provided quantity is consistent with
        the requirements for the coordinates quantity, i.e. is a list or ndarray of
        fewer than 3 dimensions.

        :param coords: Input to check for consistency with coordinates.
        :type coords: ndarray
        :returns: Boolean indicating if coords is consistent
        :rtype: bool
        """
        # TODO: implement check that arrays or lists contain only numerical values
        if coords is None:
            return False

        if isinstance(coords, list):
            return True
        if isinstance(coords, np.ndarray):
            if len(coords.shape) <= 2:
                return True
        return False


    def check_expectations(self, expectations):
        r"""
        Checks if the provided argument is consistent with expectations for
        Gaussian Process Expectations.

        Returns a boolean that is ``True`` if the provided quantity is consistent with
        the output of the predict method of a GaussianProcess object, i.e. that it
        is a tuple of length 3 containing ndarrays.

        :param expectations: Input to check for consistency with expectations
        :type expectations: tuple of 3 numpy arrays
        :returns: Boolean indicating if expectations is consistent
        :rtype: bool
        """
        if expectations is None:
            return False
        if not isinstance(expectations, tuple):
            return False
        if len(expectations) != 3:
            return False
        if not all((isinstance(expectations[0], np.ndarray),
                    isinstance(expectations[1], np.ndarray),
                    (isinstance(expectations[2], np.ndarray) or
                     expectations[2] is None))):
            raise TypeError("bad input type for HistoryMatching - expected expectation " +
                            "values in the form of a Tuple of ndarrays.")
        assert np.all(expectations[1] >= 0.), "all variances must be non-negative"
        return True


    def check_threshold(self, threshold):
        r"""
        Check value of threshold

        Checks if the provided argument is consistent with expectations for a
        threshold value.

        Returns a boolean that is ``True`` if the provided quantity is consistent with
        the requirements for a threshold value, i.e. that it is a non-negative numerical
        value that can be converted to a float.

        :param threshold: threshold to be tested
        :type threshold: float
        :returns:  Boolean indicating if the provided argument is consistent with
                   a threshold
        :rtype: bool
        """
        if threshold is None:
            return False
        try:
            test = float(threshold)
            assert test >= 0., "threshold must be non-negative"
            return True
        except TypeError:
            return False


    def check_ncoords(self, ncoords):
        r"""
        Check value of ncoords

        Checks if the provided argument is consistent with expectations for the
        number of coordinates for which expectation values are to be / have been
        computed.

        Returns a boolean that is True if the provided quantity is consistent with
        the requirements for a ncoords value, i.e. that it is a single numerical
        value that can be converted to a float.

        :param ncoords: number of query points for ``HistoryMatching``
        :type ncoords: int
        :returns: Boolean indicating if ``ncoords`` is consistent with requirements
        :rtype: bool
        """
        if ncoords is None:
            return False
        try:
            test = float(ncoords)
            return True
        except TypeError:
            return False


    def update(self):
        r"""
        Update History Matching object

        Checks that sufficient information exists to computen dim and/or ncoords
        and, if so, computes these values. This also serves to update these values
        if the information on which they are based is changed. No inputs, no return
        value.

        :returns: None
        """

        if self.check_coords(self.coords):
            self.ndim = self.coords.shape[1]
            self.ncoords = self.coords.shape[0]
        elif self.check_expectations(self.expectations):
            self.ncoords = self.expectations[0].shape[0]


    def __str__(self):
        r"""
        Returns string representation of ``HistoryMatching`` object

        Returns a string representation of a ``HistoryMatching`` object,
        used when printing information to the interpreter or when the
        ``status`` method is called.

        :returns: string representation of the HistoryMatching object.
        :rtype: str
        """

        if self.coords is None:
            coord_str = None
        else:
            coord_str = self.coords.shape

        if self.expectations is None:
            exp_str = None
        else:
            exp_str = (str(type(self.expectations)) + " of len " +
                       str(len(self.expectations)) + " containing arrays of shape: " +
                       str(self.expectations[0].shape))

        if self.I is None:
            I_str = None
        elif isinstance(self.I, np.ndarray):
            I_str = (str(type(self.I)) + " of shape " + str(self.I.shape))

        if self.NROY is None:
            NROY_str = None
        elif isinstance(self.NROY, list):
            NROY_str = (str(type(self.NROY)) + " of length " + len(self.NROY))

        if self.RO is None:
            RO_str = None
        elif isinstance(self.RO, list):
            RO_str = (str(type(self.RO)) + " of length " + len(self.RO))

        return ("History Matching tools created with:\n" +
                "Gaussian Process: {}\n" +
                "Observations: {}\n" +
                "Coords: {}\n" +
                "Expectations: {}\n" +
                "No. of Input Dimensions: {}\n" +
                "No. of Descrete Expectation Values: {}\n" +
                "I_threshold: {}\n" +
                "I: {}\n" +
                "NROY: {}\n" +
                "RO: {}\n").format(self.gp, self.obs, coord_str, exp_str,
                                   self.ndim, self.ncoords, self.threshold,
                                   I_str, NROY_str, RO_str)
