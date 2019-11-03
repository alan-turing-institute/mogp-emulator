"""
extends GaussianProcess with an (optional) GPU implementation
"""

import os.path
import ctypes
import ctypes.util
import numpy as np
import numpy.ctypeslib as npctypes

from . import GaussianProcess

_ndarray_1d = npctypes.ndpointer(dtype=np.float64,
                                 ndim=1,
                                 flags='C_CONTIGUOUS')

_ndarray_2d = npctypes.ndpointer(dtype=np.float64,
                                 ndim=2,
                                 flags='C_CONTIGUOUS')

class UnavailableError(RuntimeError):
    """Exception type to use when a GPU, or the GPU library, is unavailable"""
    pass

def _find_mogp_gpu(verbose = True):
    """
    Look for the library that provides GPU support.  It expects to
    find the library in "../mogp_gpu/lib/libgpgpu.so" relative to the
    directory where this source file is located.  If found, a
    ctypes.CDLL object wrapper around the library is returned,
    otherwise None.
    """
    gpgpu_path = os.path.join(os.path.dirname(__file__),
                              os.pardir,
                              "mogp_gpu",
                              "lib",
                              "libgpgpu.so")
    if verbose:
        print("GaussianProcessGPU._find_mogp_gpu: Looking for GPGPU library at: " + gpgpu_path)

    if not os.path.isfile(gpgpu_path):
        if verbose:
            print("GaussianProcessGPU._find_mogp_gpu: Could not find libgpgpu.so, which is needed for GPU "
                  "support.  Falling back to CPU-only.  Please consult "
                  "the build instructions.")
    else:
        if verbose:
            print("GaussianProcessGPU._find_mogp_gpu: Found gpgpu library at " + gpgpu_path)
        try:
            mogp_gpu = ctypes.CDLL(gpgpu_path)
            hello = mogp_gpu.gplib_hello_world
            hello.restype = ctypes.c_double
            assert(hello() == 0.1134)
            return mogp_gpu

        except OSError as err:
            if verbose:
                print("GaussianProcessGPU._find_mogp_gpu: There was a problem loading the library.  The error "
                      "was: " + str(err))

        except AssertionError:
            if verbose:
                print("GaussianProcessGPU._find_mogp_gpu: The library was loaded, but the simple check failed.")


class GPGPULibrary(object):
    """Encapsulates the functionality provided by the library, bound as
    ctypes calls.

    .. py:method:: make_gp 
    .. py:method:: _make_gp(source : string, filename, symbol='file') -> 


    """
    def __init__(self):
        """Instantiating a member of this class attempts to find the library,
        and provides instance methods that call into it."""
        self.lib = _find_mogp_gpu()
        if self.lib:
            self._make_gp = self.lib.gplib_make_gp
            self._make_gp.restype  = ctypes.c_void_p # handle
            self._make_gp.argtypes = [ctypes.c_uint, # N
                                      ctypes.c_uint, # Ninput
                                      _ndarray_1d,   # theta
                                      _ndarray_2d,   # xs
                                      _ndarray_1d]   # ts

            self._destroy_gp = self.lib.gplib_destroy_gp
            self._destroy_gp.restype  = None
            self._destroy_gp.argtypes = [ctypes.c_void_p]

            self._predict = self.lib.gplib_predict
            self._predict.restype  = ctypes.c_double
            self._predict.argtypes = [ctypes.c_void_p, _ndarray_1d]

            self._predict_variance = self.lib.gplib_predict_variance
            self._predict_variance.restype  = ctypes.c_double
            self._predict_variance.argtypes = [ctypes.c_void_p,
                                               _ndarray_1d,
                                               _ndarray_1d]

            self._predict_batch = self.lib.gplib_predict_batch
            self._predict_batch.restype  = None
            self._predict_batch.argtypes = [ctypes.c_void_p,
                                            ctypes.c_int,
                                            _ndarray_2d,
                                            _ndarray_1d]

            self._predict_variance_batch = self.lib.gplib_predict_variance_batch
            self._predict_variance_batch.restype  = None
            self._predict_variance_batch.argtypes = [ctypes.c_void_p,
                                                     ctypes.c_int,
                                                     _ndarray_2d,
                                                     _ndarray_1d,
                                                     _ndarray_1d]

            self._update_theta = self.lib.gplib_update_theta
            self._update_theta.restype  = None
            self._update_theta.argtypes = [ctypes.c_void_p, _ndarray_1d]

            self._get_invQ = self.lib.gplib_get_invQ
            self._get_invQ.restype = None
            self._get_invQ.argtypes = [ctypes.c_void_p, _ndarray_2d]

            self._get_invQt = self.lib.gplib_get_invQt
            self._get_invQt.restype = None
            self._get_invQt.argtypes = [ctypes.c_void_p, _ndarray_1d]

            self._get_logdetQ = self.lib.gplib_get_logdetQ
            self._get_logdetQ.restype = ctypes.c_double
            self._get_logdetQ.argtypes = [ctypes.c_void_p]

            self._status = self.lib.gplib_status
            self._status.restype  = ctypes.c_int
            self._status.argtypes = [ctypes.c_void_p]

            self._error_string = self.lib.gplib_error_string
            self._error_string.restype  = ctypes.c_char_p
            self._error_string.argtypes = [ctypes.c_void_p]
            
    def have_gpu(self):
        return (self.lib is not None)
    
    def make_gp(self, *args):
        return GPGPU(self, *args)


class GPGPU(object):
    """Represents a single GaussianProcess object on the GPU.

    A higher-level class than GPGPULibrary, this class wraps the raw methods
    provided by :class:`mogp_emulator.GaussianProcessGPU.GPGPULibrary` with some
    checks.
    """
    def __init__(self, lib_wrapper, theta, xs, ts):
        self._lib_wrapper = lib_wrapper

        self.N = xs.shape[0]
        self.Ninput = xs.shape[1]

        assert(xs.ndim     == 2)
        assert(ts.shape    == (self.N,))
        assert(theta.shape == (self.Ninput + 1,))
        
        self._ready = False        
        
        if self._lib_wrapper.have_gpu():
            self.handle = self._lib_wrapper._make_gp(self.N, self.Ninput, theta, xs, ts)
            if self._lib_wrapper._status(self.handle) == 0:
                self._ready = True
            else:
                print("An error occured when trying to create a Gaussian "
                      "process object on the device.  The following response "
                      "was returned: "
                      + self._lib_wrapper._error_string(self.handle).decode())

    def ready(self):
        return self._ready

    def __del__(self):
        if (hasattr(self, "lib_wrapper") and hasattr(self, "handle")
            and self.gp_okay()):
            self.lib_wrapper._destroy_gp(self.handle)

    def _call_gpu(self, call, name, *args):
        if self.ready():
            result = call(*args)
            if (self._lib_wrapper._status(self.handle) == 0):
                return result
            else:
                msg = ("Error when calling " + name + " on the GPU: "
                       + self._lib_wrapper._error_string(self.handle).decode())
                print(msg)
                raise RuntimeError(msg)
        else:
            raise UnavailableError("GPU interface unavailable")

    def predict(self, xnew):
        single_prediction = (xnew.ndim == 1)

        if single_prediction:
            assert(xnew.shape == (self.Ninput,))
            xnew = xnew[np.newaxis,:]

        assert(xnew.shape[1] == self.Ninput)
        Npredict = xnew.shape[0]
        result = np.zeros(Npredict)
        self._call_gpu(
            lambda Npredict, xnew, result: self._lib_wrapper._predict_batch(
                self.handle, Npredict, xnew, result),
            __name__, Npredict, xnew, result)

        if single_prediction:
            return result[0]
        else:
            return result

    def predict_variance(self, xnew):
        single_prediction = (xnew.ndim == 1)

        if single_prediction:
            assert(xnew.shape == (self.Ninput,))
            xnew = xnew[np.newaxis,:]
            
        assert(xnew.shape[1] == self.Ninput)
        Npredict = xnew.shape[0]
        result = np.zeros(Npredict)
        variance = np.zeros(Npredict)
        self._call_gpu(
            lambda Npredict, xnew, result, variance: \
            self._lib_wrapper._predict_variance_batch(
                self.handle, Npredict, xnew, result, variance),
            __name__, Npredict, xnew, result, variance)

        if single_prediction:
            return (result[0], variance[0])
        else:
            return (result, variance)

    def update_theta(self, theta):
        assert(theta.shape == (self.Ninput + 1,))

        return self._call_gpu(
            lambda theta: self._lib_wrapper._update_theta(self.handle, theta),
            __name__, theta)

    def get_invQ(self):
        invQ = np.zeros((self.N, self.N))
        self._call_gpu(
            lambda invQ: self._lib_wrapper._get_invQ(self.handle, invQ),
            __name__, invQ)
        return invQ

    def get_invQt(self):
        invQt = np.zeros((self.N,))
        self._call_gpu(
            lambda invQt: self._lib_wrapper._get_invQt(self.handle, invQt),
            __name__, invQt)
        return invQt

    def get_logdetQ(self):
        return self._call_gpu(
            lambda: self._lib_wrapper._get_logdetQ(self.handle),
            __name__)


class GaussianProcessGPU(object):
    """This class implements the same interface as
    :class:`mogp_emulator.GaussianProcess.GaussianProcess`, but with
    particular methods overridden to use a GPU if it is available.
    """

    ## the (class-level) interface to the library
    _gpu_library = GPGPULibrary()
    
    def __init__(self, *args):
        """Create a new GP emulator, using a GPU if it is available.
        
        The arguments passed as `*args` must be either:

        - A GaussianProcess object (to which this should behave
          identically, save for performing its computations on the
          GPU)
        - Arguments that are acceptable to
          :func:`mogp_emulator.GaussianProcess.GaussianProcess.__init__` (documented there).

        """
        self.nugget = None
        self.theta = None
        
        if len(args) == 1 and type(args[0]) == GaussianProcess:
            ## gp = args[0]
            ## self.inputs = gp.inputs
            ## ...
            
            raise NotImplementedError("Haven't yet implemented construction "
                                      "from GaussianProcess")

        elif len(args) == 1:
            raise NotImplementedError("Haven't yet implemented construction "
                                      "from a stored emulator")

        elif len(args) == 2 or len(args) == 3:
            self.inputs = np.array(args[0])
            if self.inputs.ndim == 1:
                self.inputs = np.expand_dims(self.inputs, axis=0)
            if not self.inputs.ndim == 2:
                raise ValueError("Inputs must by a 2D array")
            self.targets = np.array(args[1])
            if not self.targets.ndim == 1:
                raise ValueError("Targets must be a 1D array")
            if not self.targets.shape[0] == self.inputs.shape[0]:
                raise ValueError("First dimensions of inputs and targets must be the same length")

            if len(args) == 3:
                self.nugget = args[2]
                if not self.nugget == None:
                    self.nugget = float(self.nugget)
                    if self.nugget < 0.:
                        raise ValueError("nugget parameter must be onnegative or None")

        else:       
            raise ValueError("Init method of GaussianProcessGPU requires two arguments: "
                             "(input array and target array)")

        self.n = self.inputs.shape[0]
        self.D = self.inputs.shape[1]

        ## attempt to create the GP on the device
        self.gpgpu = None        
        self._device_gp_create()

    def save_emulator(self, filename):
        raise NotImplementedError("Saving GaussianProcessGPU not implemented yet")
        
    def device_ready(self):
        return self.gpgpu and self.gpgpu.ready()

    def get_n(self):
        return self.n

    def get_D(self):
        return self.D

    @property
    def invQ(self):
        return self.gpgpu.get_invQ()
    
    @property
    def invQt(self):
        return self.gpgpu.get_invQt()

    @property
    def logdetQ(self):
        return self.gpgpu.get_logdetQ()

    def get_params(self):
        return self.theta

    def get_nugget(self):
        return self.nugget

    def set_nugget(self, nugget):
        if not nugget == None:
            nugget = float(nugget)
            assert nugget >= 0., "noise parameter must be nonnegative"
        self.nugget = nugget

    def _device_gp_create(self):
        """Attempt to update the various GP data on the device"""
        if self.theta != None:
            theta = self.theta
        else:
            theta = np.zeros(self.get_D() + 1)

        assert theta.shape == (self.get_D() + 1,)

        self.gpgpu = self._gpu_library.make_gp(theta, self.inputs, self.targets)

    def _set_params(self, theta):
        self.theta = np.array(theta)
        if self.device_ready():
            self.gpgpu.update_theta(self.theta)

    def loglikelihood(self, theta):
        self._set_params(theta)
            
    def predict(self, testing, do_deriv = True, do_unc = True, *args, **kwargs):
        """Make a prediction for a set of input vectors

        See :func:`mogp_emulator.GaussianProcess.predict`, which provides the
        same interface and functionality.

        Currently, the GPU implementation does not provide variance or
        derivative computations.  Therefore, if `do_deriv` or `do_unc` are
        ``True``, the prediction will fall back to the CPU implementation.

        :type testing: ndarray

        :param testing: As for `meth`:mogp_emulator.GaussianProcess.predict:.
                        Array-like object holding the points where predictions
                        will be made.  Must have shape ``(n_predict, D)`` or
                        ``(D,)`` (for a single prediction)

        :type do_deriv: Bool

        :param do_deriv: (optional, default `True`) As for
                         `meth`:mogp_emulator.GaussianProcess.predict:.  Flag
                         indicating if the uncertainties are to be computed.  If
                         ``False`` the method returns ``None`` in place of the
                         uncertainty array. Default value is ``True``.

        :type do_unc:

        :param do_unc: (optional, default `True`) As for
                      `meth`:mogp_emulator.GaussianProcess.predict:.  Flag
                      indicating if the uncertainties are to be computed.  If
                      ``False`` the method returns ``None`` in place of the
                      uncertainty array. Default value is ``True``.

        :throws: RuntimeError: When the GPU library (or a suitable GPU) is not
                               available for the requested task, a runtime
                               exception is (re)thrown.  This could be the
                               derived "UnavailableError" exception, a
                               "NotImplementedError", or a plain "RuntimeError"
                               indicating some other runtime failure.
        """
        if not do_deriv and not args and not kwargs:
            if do_unc:
                mean, var = self.gpgpu.predict_variance(testing)
                return (mean, var, None)
            else:
                mean = self.gpgpu.predict(testing)
                return (mean, None, None)
        else:
            raise NotImplementedError(
                "GaussianProcessGPU.predict does not support the options "
                "requested for prediction (consider using GaussianProcess "
                "instead)")

    ## __setstate__ and __getstate__ for pickling: don't pickle "gpgpu",
    ## since the ctypes members won't pickle, reinitialize after.
    ## (Pickling is required to use multiprocessing.)
    def __setstate__(self, state):
        self.__dict__ = state
        self._device_gp_create()
        if self.theta is not None:
            self._set_params(self.theta)

    def __getstate__(self):
        copy_dict = self.__dict__
        del copy_dict["gpgpu"]
        copy_dict["gpgpu"] = None
        return copy_dict
        
