# Class implementing a multi-output Gaussian Process for emulating the output of
# a series of computer simulations. Since each emulator is independent of the
# others, the emulators can be fit in parallel, significantly speeding up
# the fitting procedure.

# The class relies on an underlying implementation of a gp_emulator which you must
# have installed on your system. I used Jose Gomez-Dans' original implementation,
# but the interface for Sinan's code should be idential. However, I note that Sinan's
# code did not use the most efficient method for forming the covariance matrix, which
# takes up the bulk of the time for fitting each emulator. Therefore, I suggest
# using Jose's code for the time being, as it should give satisfactory performance
# for small problems

from multiprocessing import Pool
import numpy as np
from gp_emulator.GaussianProcess import GaussianProcess

class MultiOutputGP(object):
    """
    Implementation of a multiple-output Gaussian Process Emulator.
    
    This class provides an interface to fit a Gaussian Process Emulator to multiple targets
    using the same input data. The class creates all of the necessary sub-emulators from
    the input data and provides interfaces to the ``learn_hyperparameters`` and ``predict``
    methods of the sub-emulators. Because the emulators are all fit independently, the
    class provides the option to use multiple processes to fit the emulators and make
    predictions in parallel.
    
    The emulators are stored internally in a list. Other useful information stored is the
    numer of emulators ``n_emulators``, number of training examples ``n``, and number of
    input parameters ``D``. These other variables are made available externally through
    the ``get_n_emulators``, ``get_n``, and ``get_D`` methods.
    
    Example: ::
    
        >>> import numpy as np
        >>> from mogp_emulator import MultiOutputGP
        >>> x = np.array([[1., 2., 3.], [4., 5., 6.]])
        >>> y = np.array([[4., 6.], [5., 7.]])
        >>> mogp = MultiOutputGP(x, y)
        >>> print(mogp)
        Multi-Output Gaussian Process with:
        2 emulators
        2 training examples
        3 input variables
        >>> mogp.get_n_emulators()
        2
        >>> mogp.get_n()
        2
        >>> mogp.get_D()
        3
        >>> np.random.seed(47)
        >>> mogp.learn_hyperparameters()
        After 15, the minimum cost was 5.322784e+00
        After 15, the minimum cost was 5.140462e+00
        [(5.1404621594033895,
          array([-6.95976295, -4.99805894, -5.21415165,  3.23718116, -0.61961335])),
         (5.322783716197344,
          array([-9.80965415, -5.53659105, -6.29521694,  3.58162789,  0.06580016]))]
        >>> x_predict = np.array([[2., 3., 4.], [7., 8., 9.]])
        >>> mogp.predict()
        (array([[4.76282574, 6.36038561],
                [5.78892678, 6.9389214 ]]), array([[0.30425374, 2.21021771],
                [0.56260549, 2.67487348]]), array([[[0.03785875, 0.26923002, 0.21690803],
                 [0.00478238, 0.03400961, 0.02740021]],
                [[0.00309533, 0.22206213, 0.10399381],
                 [0.00107731, 0.07728762, 0.03619453]]]))
    
    Note that there will frequently be a ``RuntimeWarning`` during the fitting of hyperparameters,
    due to the random initial conditions which sometimes leads to a poorly conditioned matrix
    inversion. This should only be a concern if the final minimum cost is 9999 (meaning that
    all attempts to minimize the negative log-likelihood resulted in an error).
        
    """
    def __init__(self, *args):
        """
        Create a new multi-output GP Emulator
        
        Creates a new multi-output GP Emulator from either the input data and targets to
        be fit or a file holding the input/targets and learned parameter values.
        
        Arguments passed to the ``__init__`` method must be either two arguments which
        are numpy arrays ``inputs`` and ``targets``, described below, or a single argument
        which is the filename (string or file handle) of a previously saved emulator.
        
        ``inputs`` is a 2D array-like object holding the input data, whose shape is
        ``n`` by ``D``, where ``n`` is the number of training examples to be fit and ``D``
        is the number of input variables to each simulation. Because the model assumes all
        outputs are drawn from the same identical set of simulations (i.e. the normal use
        case is to fit a series of computer simulations with multiple outputs from the same
        input), the input to each emulator is identical.
        
        ``targets`` is the target data to be fit by the emulator, also held in an array-like
        object. This can be either a 1D or 2D array, where the last dimension must have length
        ``n``. If the ``targets`` array is of shape ``(n_emulators,n)``, then the emulator fits
        a total of ``n_emulators`` to the different target arrays, while if targets has shape
        ``(n,)``, a single emulator is fit.
        
        If two input arguments ``inputs`` and ``targets`` are given:
        :param inputs: Numpy array holding emulator input parameters. Must be 2D with shape
                       ``n`` by ``D``, where ``n`` is the number of training examples and
                       ``D`` is the number of input parameters for each output.
        :type inputs: ndarray
        :param targets: Numpy array holding emulator targets. Must be 2D or 1D with length
                       ``n`` in the final dimension. The first dimension is of length
                       ``n_emulators`` (defaults to a single emulator if the input is 1D)
        :type targets: ndarray
        
        If one input argument ``emulator_file`` is given:
        :param emulator_file: Filename for saved emulator parameters (using the ``save_emulator``
                              method)
        :type emulator_file: str
        
        :returns: New ``MultiOutputGP`` instance
        :rtypr: MultiOutputGP
        """
        
        emulator_file = None
        
        if len(args) == 1:
            emulator_file = args[0]
            inputs, targets, theta = self._load_emulators(emulator_file)
        elif len(args) == 2:
            inputs, targets = args
        else:
            raise ValueError("Bad number of inputs to create a MultiOutputGP (must be 2 arrays or a single filename)")
        
        # check input types and shapes, reshape as appropriate for the case of a single emulator
        inputs = np.array(inputs)
        targets = np.array(targets)
        if len(targets.shape) == 1:
            targets = np.reshape(targets, (1, len(targets)))
        elif not (len(targets.shape) == 2):
            raise ValueError("targets must be either a 1D or 2D array")
        if not (len(inputs.shape) == 2):
            raise ValueError("inputs must be 2D array")
        if not (inputs.shape[0] == targets.shape[1]):
            raise ValueError("the first dimension of inputs must be the same length as the second dimension of targets (or first if targets is 1D))")

        self.emulators = [ GaussianProcess(inputs, single_target) for single_target in targets]
        
        self.n_emulators = targets.shape[0]
        self.n = inputs.shape[0]
        self.D = inputs.shape[1]
        
        if not (emulator_file is None or theta is None):
            self._set_params(theta)
        
    def _load_emulators(self, filename):
        """
        Load saved emulators and parameter values from file
        
        Method takes the filename of saved emulators (using the ``save_emulators`` method).
        The saved emulator may or may not contain the fitted parameters. If there are no
        parameters found in the emulator file, the method returns ``None`` for the
        parameters.
        
        :param filename: File where the emulator parameters. Can be a string filename
                         or an open file handle.
        :type filename: str or file
        :returns: inputs, targets, and (optionally) fitted parameter values from the
                  saved emulator file
        :rtype: tuple containing 3 ndarrays or 2 ndarrays and None (if no theta values
                are found in the emulator file)
        """

        emulator_file = np.load(filename)
        
        try:
            inputs = np.array(emulator_file['inputs'])
            targets = np.array(emulator_file['targets'])
        except KeyError:
            raise KeyError("Emulator file does not contain emulator inputs and targets")
            
        try:
            theta = np.array(emulator_file['theta'])
        except KeyError:
            theta = None
            
        return inputs, targets, theta
        
    def save_emulators(self, filename):
        """
        Write emulators to disk
        
        Method saves emulators to disk using the given filename or file handle. The (common)
        inputs to all emulators are saved, and all targets are collected into a single numpy
        array (this saves the data in the same format used in the two-argument ``__init__``
        method). If the model has been assigned parameters, either manually or by fitting,
        those parameters are saved as well. Once saved, the emulator can be read by passing
        the file name or handle to the one-argument ``__init__`` method.
        
        :param filename: Name of file (or file handle) to which the emulators will be save.
        :type filename: str or file
        :returns: None
        """
        
        emulators_dict = {}
        emulators_dict['targets'] = np.array([emulator.targets for emulator in self.emulators])
        emulators_dict['inputs'] = self.emulators[0].inputs
        
        try:
            emulators_dict['theta'] = np.array([emulator.current_theta for emulator in self.emulators])
        except AttributeError:
            pass
        
        np.savez(filename, **emulators_dict)
        
    def get_n_emulators(self):
        """
        Returns the number of emulators
        
        :returns: Number of emulators in the object
        :rtype: int
        """
        return self.n_emulators
        
    def get_n(self):
        """
        Returns number of training examples in each emulator
        
        :returns: Number of training examples in each emulator in the object
        :rtype: int
        """
        return self.n
        
    def get_D(self):
        """
        Returns number of inputs for each emulator
        
        :returns: Number of inputs for each emulator in the object
        :rtype: int
        """
        return self.D
        
    def _set_params(self, theta):
        """
        Method for setting the hyperparameters for all emulators
        
        This method is used to reset the value of the hyperparameters for the emulators and
        update the log-likelihood. It is used after fitting the hyperparameters or when loading
        an emulator from file. Input ``theta`` must be array-like with shape
        ``(n_emulators, D + 2)``, where ``n_emulators`` is the number of emulators and ``D``
        is the number of input parameters. If the number of emulators is 1, then ``theta``
        having shape ``(D + 2,)`` is allowed.
        
        :param theta: Parameter values to be used for the emulators. Must be array-like and
                      have shape ``(n_emulators, D + 2)`` (if there is only a single
                      emulator, then shape ``(D + 2,)`` is allowed)
        :type theta: ndarray
        :returns: None
        """
        theta = np.array(theta)
        if self.n_emulators == 1 and theta.shape == (self.D + 2,):
            theta = np.reshape(theta, (1, self.D + 2))
        assert theta.shape == (self.n_emulators, self.D + 2), "theta must have shape n_emulators x (D + 2)"
        
        for emulator, theta_val in zip(self.emulators, theta):
            _ = emulator.loglikelihood(theta_val)
        
    def learn_hyperparameters(self, n_tries=15, verbose=False, x0=None, processes=None):
        """
        Fit hyperparameters for each model
        
        Fit the hyperparameters for each emulator. Options that can be specified include
        the number of different initial conditions to try during the optimization step,
        the level of verbosity of output during the fitting, the initial values of the
        hyperparameters to use when starting the optimization step, and the number of
        processes to use when fitting the models. Since each model can be fit independently
        of the others, parallelization can significantly improve the speed at which
        the models are fit.
        
        Returns a list holding ``n_emulators`` tuples, each of which contains the minimum
        negative log-likelihood and a numpy array holding the optimal parameters found for
        each model.
        
        Note that fitting the hyperparameters will frequently result in a ``RuntimeWarning``.
        This is because the fitting routine tries several different sets of initial 
        conditions to ensure that the minimization routines does not get stuck in a
        local minimum, and often certain sets of initial conditions lead to a poorly
        conditions matrix inversion. This should only be a concern if the final
        negative log-likelihood for any emulator is 9999, which means that all attempts
        to fit that emulator resulting in a warning.
        
        :param n_tries: (optional) The number of different initial conditions to try when
                        optimizing over the hyperparameters (must be a positive integer,
                        default = 15)
        :type n_tries: int
        :param verbose: (optional) Flag indicating whether or not to print detailed
                        information on the fitting to the screen (default = False)
        :type verbose: bool
        :param x0: (optional) Initial value of the hyperparameters to use in the optimization
                   routine (must be array-like with a length of ``D + 2``, where ``D`` is
                   the number of input parameters to each model). Default is ``None``.
        :type x0: ndarray or None
        :param processes: (optional) Number of processes to use when fitting the model.
                          Must be a positive integer or ``None`` to use the number of
                          processors on the computer (default is ``None``)
        :type processes: int or None
        :returns: List holding ``n_emulators`` tuples of length 2. Each tuple contains
                  the minimum negative log-likelihood for that particular emulator and a
                  numpy array of length ``D + 2`` holding the corresponding hyperparameters
        :rtype: list
        """
        
        assert int(n_tries) > 0, "n_tries must be a positive integer"
        if not x0 is None:
            x0 = np.array(x0)
            assert len(x0) == self.D + 2, "x0 must have length of number of input parameters D + 2"
        if not processes is None:
            processes = int(processes)
            assert processes > 0, "number of processes must be positive"
        
        n_tries = int(n_tries)
        
        with Pool(processes) as p:
            likelihood_theta_vals = p.starmap(GaussianProcess.learn_hyperparameters,
                                          [(gp, n_tries, verbose, x0) for gp in self.emulators])
        
        # re-evaluate log likelihood for each emulator to update current parameter values
        # (needed because of how multiprocessing works -- the bulk of the work is done in
        # parallel, but this step ensures that the results are gathered correctly for each
        # emulator)
        
        loglike_unpacked, theta_unpacked = [np.array(t) for t in zip(*likelihood_theta_vals)]
        
        self._set_params(theta_unpacked)

        return likelihood_theta_vals
        
    def predict(self, testing, do_deriv=True, do_unc=True, processes=None):
        """
        Make a prediction for a set of input vectors
        
        Makes predictions for each of the emulators on a given set of input vectors. The
        input vectors must be passed as a ``(n_predict, D)`` or ``(D,)`` shaped array-like
        object, where ``n_predict`` is the number of different prediction points under
        consideration and ``D`` is the number of inputs to the emulator. If the prediction
        inputs array has shape ``(D,)``, then the method assumes ``n_predict == 1``. 
        The prediction points are passed to each emulator and the predictions are collected
        into a ``(n_emulators, n_predict)`` shaped numpy array as the first return value
        from the method.
        
        Optionally, the emulator can also calculate the uncertainties in the predictions 
        and the derivatives with respect to each input parameter. If the uncertainties are
        computed, they are returned as the second output from the method as a
        ``(n_emulators, n_predict)`` shaped numpy array. If the derivatives are computed,
        they are returned as the third output from the method as a
        ``(n_emulators, n_predict, D)`` shaped numpy array.
        
        As with the fitting, this computation can be done independently for each emulator
        and thus can be done in parallel.
        
        :param testing: Array-like object holding the points where predictions will be made.
                        Must have shape ``(n_predict, D)`` or ``(D,)`` (for a single prediction)
        :type testing: ndarray
        :param do_deriv: (optional) Flag indicating if the derivatives are to be computed.
                         If ``False`` the method returns ``None`` in place of the derivative
                         array. Default value is ``True``.
        :type do_deriv: bool
        :param do_unc: (optional) Flag indicating if the uncertainties are to be computed.
                         If ``False`` the method returns ``None`` in place of the uncertainty
                         array. Default value is ``True``.
        :type do_unc: bool
        :param processes: (optional) Number of processes to use when making the predictions.
                          Must be a positive integer or ``None`` to use the number of
                          processors on the computer (default is ``None``)
        :type processes: int or None
        :returns: Tuple of numpy arrays holding the predictions, uncertainties, and derivatives,
                  respectively. Predictions and uncertainties have shape ``(n_emulators, n_predict)``
                  while the derivatives have shape ``(n_emulators, n_predict, D)``. If
                  the ``do_unc`` or ``do_deriv`` flags are set to ``False``, then those arrays
                  are replaced by ``None``.
        :rtype: tuple
        """
        testing = np.array(testing)
        if testing.shape == (self.D,):
            testing = np.reshape(testing, (1, self.D))
        assert len(testing.shape) == 2, "testing must be a 2D array"
        assert testing.shape[1] == self.D, "second dimension of testing must be the same as the number of input parameters"
        if not processes is None:
            processes = int(processes)
            assert processes > 0, "number of processes must be a positive integer"
            
        with Pool(processes) as p:
            predict_vals = p.starmap(GaussianProcess.predict, [(gp, testing) for gp in self.emulators])
        
        # repackage predictions into numpy arrays
        
        predict_unpacked, unc_unpacked, deriv_unpacked = [np.array(t) for t in zip(*predict_vals)]
        
        if not do_unc:
            unc_unpacked = None
        if not do_deriv:
            deriv_unpacked = None
            
        return predict_unpacked, unc_unpacked, deriv_unpacked
        
    def __str__(self):
        """
        Returns a string representation of the model
        
        :returns: A string representation of the model (indicates number of sub-components
                  and array shapes)
        :rtype: str
        """
        return ("Multi-Output Gaussian Process with:\n"+
                 str(self.get_n_emulators())+" emulators\n"+
                 str(self.get_n())+" training examples\n"+
                 str(self.get_D())+" input variables")
        