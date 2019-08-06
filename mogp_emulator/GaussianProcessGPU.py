## extends GaussianProcess with an (optional) GPU implementation

import os.path
import ctypes
import ctypes.util

from . import GaussianProcess

def _find_mogp_gpu():
    """
    Look for the library that provides GPU support.  It expects to
    find the library in "../mogp_gpu/libgpgpu.so" relative to the
    directory where this source file is located.  If found, a
    ctypes.CDLL object wrapper around the library is returned,
    otherwise None.
    """
    path = os.path.join(os.path.dirname(__file__),
                        os.pardir,
                        "mogp_gpu",
                        "libgpgpu.so")
    try:
        mogp_gpu = ctypes.CDLL(path)
        hello = mogp_gpu.hello
        hello.restype = ctypes.c_double
        # check the response from the library
        print(hello())
        return mogp_gpu
    except OSError:
        print("Could not find libgpgpu.so, which is needed for GPU "
              "support.  Falling back to CPU-only.  Please consult "
              "the build instructions.")

class GaussianProcessGPU(object):
    ## Perform various checks (that the library exists, and that we
    ## can access a GPU through it), and set this variable to True if
    ## they pass
    gpu_enabled = False
    mogp_gpu = _find_mogp_gpu()
    if mogp_gpu:
        gpu_enabled = True
    
    def __init__(self, *args):
        if len(args) == 1 and type(args[0]) == GaussianProcess:
            print("constructor with gp argument")
            self.gp = args[0]
        else:
            self.gp = GaussianProcess(*args)

    def predict(self, testing, do_deriv = True, do_unc = True, use_gpu = None):
        if use_gpu and gpu_enabled:
            _gpu_predict()
        else:
            gp.predict(testing, do_deriv, do_unc)

    def _gpu_predict():
        pass
            
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
