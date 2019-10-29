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
                                      _ndarray_1d,   # ts
                                      _ndarray_2d,   # Q,
                                      _ndarray_2d,   # invQ,
                                      _ndarray_1d]   # invQt

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
            self._update_theta.argtypes = [ctypes.c_void_p,
                                           _ndarray_2d,
                                           _ndarray_1d,
                                           _ndarray_1d]
            
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
    """A higher-level class than GPGPULibrary, which wraps the raw methods
    provided by :class:`mogp_emulator.GaussianProcessGPU.GPGPULibrary` with
    some checks.
    """
    def __init__(self, lib_wrapper, theta, xs, ts, Q, invQ, invQt):
        self._lib_wrapper = lib_wrapper

        self.N = xs.shape[0]
        self.Ninput = xs.shape[1]

        assert(xs.ndim     == 2)
        assert(ts.shape    == (self.N,))
        assert(theta.shape == (self.Ninput + 1,))
        assert(Q.shape     == (self.N, self.N))
        assert(invQ.shape  == (self.N, self.N))
        assert(invQt.shape == (self.N,))
        
        self._ready = False        
        
        if self._lib_wrapper.have_gpu():
            self.handle = self._lib_wrapper._make_gp(self.N, self.Ninput, theta,
                                                     xs, ts, Q, invQ, invQt)
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

    def update_theta(self, invQ, theta, invQt):
        assert(invQ.shape == (self.N, self.N))
        assert(theta.shape == (self.Ninput + 1,))
        assert(invQt.shape == (self.N,))

        return self._call_gpu(
            lambda invQ, theta, invQt: self._lib_wrapper._update_theta(
                self.handle, invQ, theta, invQt),
            __name__, invQ, theta, invQt)


class GaussianProcessGPU(GaussianProcess):
    """This class derives from
    :class:`mogp_emulator.GaussianProcess.GaussianProcess`, but with
    particular methods overridden to use a GPU if it is available.

    """
    mogp_gpu = GPGPULibrary()
    
    def __init__(self, *args):
        """Create a new GP emulator, using a GPU if it is available.
        
        The arguments passed as `*args` must be either:

        - A GaussianProcess object (to which this should behave
          identically, save for performing its computations on the
          GPU)
        - Arguments that are acceptable to
          :func:`mogp_emulator.GaussianProcess.GaussianProcess.__init__` (documented there).

        """
        ## Flag to indicate whether the GPGPU object reported that it
        ## was ready.  This is cumbersome, but needed because we
        ## cannot pickle GPGPU objects
        self.device_ready = False
        
        if len(args) == 1 and type(args[0]) == GaussianProcess:
            self.gp = args[0]
            raise NotImplementedError("Haven't yet implemented construction "
                                      "from GaussianProcess")
        else:
            self.gp = super().__init__(*args)

    def _device_gp_create(self):
        """Attempt to update the various GP data on the device"""
        inputs = self.inputs
        if inputs.ndim == 1:
            inputs = expand_dims(inputs, axis=1)

        self.gpgpu = self.mogp_gpu.make_gp(
            self.get_params(), inputs, self.targets, self.Q, self.invQ,
            self.invQt)
        if (self.gpgpu.ready()):
            self.device_ready = True
            
    def _set_params(self, theta):
        super()._set_params(theta)
        if not self.device_ready:
            self._device_gp_create()
        self.gpgpu.update_theta(self.invQ, self.theta, self.invQt)

    def predict(self, testing, do_deriv = True, do_unc = True, require_gpu = True, *args, **kwargs):
        """Make a prediction for a set of input vectors

        See :func:`mogp_emulator.GaussianProcess.predict`, which provides the
        same interface and functionality.

        Currently, the GPU implementation does not provide variance or
        derivative computations.  Therefore, if `do_deriv` or `do_unc` are
        ``True``, the prediction will fall back to the CPU implementation.

        :type require_gpu: Bool

        :param require_gpu: (optional, default True).  If this parameter is
                            `True`, the GPU library must be used, and if this is
                            not possible, an exception derived from RuntimeError
                            is (re)thrown.  If this parameter is `False`, the
                            CPU implementation will (quietly) be used as a
                            fallback.

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
                               exception is (re)thrown when `require_gpu` is
                               True.  This could be the derived
                               "UnavailableError" exception, a
                               "NotImplementedError", or a plain "RuntimeError"
                               indicating some other runtime failure.
        """
        if not do_deriv and not args and not kwargs:
            try:
                if do_unc:
                    result, var = self.gpgpu.predict_variance(testing)
                    return (result, var, None)
                else:
                    return (self.gpgpu.predict(testing), None, None)
            except RuntimeError as ex:
                if require_gpu:
                    raise ex
        elif require_gpu:
            raise NotImplementedError(
                "GaussianProcessGPU.predict does not support the options "
                "requested for prediction (`require_gpu` was set - or "
                "defaulted - to True: to fall back to the non-GPU "
                "implementation in this case, set this parameter to False)")

        return super().predict(testing, do_deriv, do_unc, *args, **kwargs)            
    
    ## __setstate__ and __getstate__ for pickling: don't pickle "gpgpu",
    ## since the ctypes members won't pickle, reinitialize after.
    ## (Pickling is required to use multiprocessing.)
    def __setstate__(self, state):
        self.__dict__ = state
        # possibly set back to True by subsequent call to _device_update()
        self.device_ready = False
        self._device_gp_create()
        if self.theta is not None:
            self._set_params(self.theta)

    def __getstate__(self):
        copy_dict = self.__dict__
        del copy_dict["gpgpu"]
        return copy_dict
        
