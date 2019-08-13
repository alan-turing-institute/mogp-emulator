## extends GaussianProcess with an (optional) GPU implementation

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
        print("Looking for GPGPU library at: " + gpgpu_path)

    if not os.path.isfile(gpgpu_path):
        if verbose:
            print("Could not find libgpgpu.so, which is needed for GPU "
                  "support.  Falling back to CPU-only.  Please consult "
                  "the build instructions.")
    else:
        if verbose:
            print("Found gpgpu library at " + gpgpu_path)
        try:
            mogp_gpu = ctypes.CDLL(gpgpu_path)
            hello = mogp_gpu.gplib_hello_world
            hello.restype = ctypes.c_double
            assert(hello() == 0.1134)

            # selfcheck = mogp_gpu.gplib_check_cublas
            # selfcheck.restype = ctypes.c_int
            # assert(selfcheck() == 0)

            return mogp_gpu
        except OSError as err:
            if verbose:
                print("There was a problem loading the library.  The error "
                      "was: " + err)
        except AssertionError:
            if verbose:
                print("The library was loaded, but the simple check failed.")


class GPGPULibrary(object):    
    def __init__(self):        
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

            self._status = self.lib.gplib_status
            self._status.restype = ctypes.c_int
            self._status.argtypes = [ctypes.c_void_p]

            self._error_string = self.lib.gplib_error_string
            self._error_string.restype = ctypes.c_char_p
            self._error_string.argtypes = [ctypes.c_void_p]
            
    def have_gpu(self):
        return (self.lib is not None)
    
    def make_gp(self, *args):
        return GPGPU(self, *args)


class GPGPU(object):
    def __init__(self, lib_wrapper, theta, xs, ts, Q, invQ, invQt):
        self._lib_wrapper = lib_wrapper
        self._ready = False

        if self._lib_wrapper.have_gpu():
            self.N = xs.shape[0]
            self.Ninput = xs.shape[1]

            assert(xs.ndim     == 2)
            assert(ts.shape    == (self.N,))
            assert(theta.shape == (self.Ninput + 1,))
            assert(Q.shape     == (self.N, self.N))
            assert(invQ.shape  == (self.N, self.N))
            assert(invQt.shape == (self.N,))

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

    def predict(self, xnew):
        if self.ready():
            assert(xnew.shape == (self.Ninput,))
            result = self._lib_wrapper._predict(self.handle, xnew)
            if (self._lib_wrapper._status(self.handle) == 0):
                return result
            else:
                msg = ("Error when calling predict on the GPU: " +
                       self._lib_wrapper._error_string(self.handle).decode())
                print(msg)
                raise RuntimeError(msg)
        else:
            raise RuntimeError("GPU interface unavailable")


class GaussianProcessGPU(GaussianProcess):
    mogp_gpu = GPGPULibrary()
    
    def __init__(self, *args):
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
        self._device_gp_create()

    def predict(self, testing, do_deriv = True, do_unc = True, use_gpu = True):
        if use_gpu and not do_deriv and not do_unc:
            try:
                return (self.gpgpu.predict(testing), None, None)
            except RuntimeError:
                pass
        return super().predict(testing, do_deriv, do_unc)

    ## __setstate__ and __getstate__ for pickling: don't pickle "gpgpu",
    ## since the ctypes members won't pickle, reinitialize after.
    ## (Pickling is required to use multiprocessing.)
    def __setstate__(self, state):
        self.__dict__ = state
        # possibly set back to True by subsequent call to _device_update()
        self.device_ready = False
        self._device_gp_create()

    def __getstate__(self):
        copy_dict = self.__dict__
        del copy_dict["gpgpu"]
        return copy_dict
        
