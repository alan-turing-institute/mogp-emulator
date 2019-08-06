## extends GaussianProcess with an (optional) GPU implementation

import os.path
import ctypes
import ctypes.util
import numpy as np
import numpy.ctypeslib as npctypes

from . import GaussianProcess

def _find_mogp_gpu():
    """
    Look for the library that provides GPU support.  It expects to
    find the library in "../mogp_gpu/lib/libgpgpu.so" relative to the
    directory where this source file is located.  If found, a
    ctypes.CDLL object wrapper around the library is returned,
    otherwise None.
    """
    path = os.path.join(os.path.dirname(__file__),
                        os.pardir,
                        "mogp_gpu",
                        "lib",
                        "libgpgpu.so")
    try:
        mogp_gpu = ctypes.CDLL(path)
        hello = mogp_gpu.gplib_hello_world
        hello.restype = ctypes.c_double
        # check the response from the library
        assert(hello() == 0.1134)
        return mogp_gpu
    except OSError:
        print("Could not find libgpgpu.so, which is needed for GPU "
              "support.  Falling back to CPU-only.  Please consult "
              "the build instructions.")

_ndarray_1d = npctypes.ndpointer(dtype=np.float64,
                                 ndim=1,
                                 flags='C_CONTIGUOUS')

_ndarray_2d = npctypes.ndpointer(dtype=np.float64,
                                 ndim=2,
                                 flags='C_CONTIGUOUS')

def _make_gp_gpu_argtypes():
    return [ctypes.c_uint, # N
            ctypes.c_uint, # Ninput
            _ndarray_1d, # theta
            _ndarray_2d, # xs
            _ndarray_1d, # ts
            _ndarray_2d, # Q,
            _ndarray_2d, # invQ,
            _ndarray_1d] # invQt
            
class GaussianProcessGPU(object):
    ## Perform various checks (that the library exists, and that we
    ## can access a GPU through it), and set this variable to True if
    ## they pass
    gpu_enabled = False
    mogp_gpu = _find_mogp_gpu()
    
    make_gp_gpu = mogp_gpu.gplib_make_gp
    make_gp_gpu.restype = ctypes.c_void_p
    make_gp_gpu.argtypes = _make_gp_gpu_argtypes()

    gplib_predict = mogp_gpu.gplib_predict
    gplib_predict.restype = ctypes.c_double
    gplib_predict.argtypes = [ctypes.c_void_p, _ndarray_1d]
    
    def _gpu_predict(self, testing):
        return self.gplib_predict(self.gpgpu_handle, testing)

    if mogp_gpu:
        gpu_enabled = True
    
    def __init__(self, *args):
        if len(args) == 1 and type(args[0]) == GaussianProcess:
            print("constructor with gp argument")
            self.gp = args[0]
        else:
            self.gp = GaussianProcess(*args)

    def _set_params(self, theta):
        self.gp._set_params(theta)
        inputs = self.gp.inputs
        if inputs.ndim == 1:
            inputs = expand_dims(inputs, axis=1)
            
        self.gpgpu_handle = GaussianProcessGPU.make_gp_gpu(
            self.gp.get_n(), self.gp.get_D(), self.gp.get_params(),
            inputs, self.gp.targets, self.gp.Q, self.gp.invQ,
            self.gp.invQt)

    def predict(self, testing, do_deriv = True, do_unc = True, use_gpu = None):
        if not use_gpu and self.gpu_enabled:
            return (self._gpu_predict(testing), None, None)
        else:
            return gp.predict(testing, do_deriv, do_unc)

    ## Delegate other properties to gp
    def __getattr__(self, name):
        if 'gp' not in vars(self):
            raise AttributeError
        return getattr(self.gp, name)    
            
    ## __setstate__ and __getstate__ for pickling: only pickle the gp,
    ## then reinitialize because the ctypes members won't pickle.
    ## (Pickling is required to use multiprocessing.)
    def __setstate__(self, state):
        self.__init__(state)

    def __getstate__(self):
        return self.gp
