"""
extends GaussianProcess with an (optional) GPU implementation
"""

import os
import numpy as np
from mogp_emulator.MeanFunction import MeanFunction, MeanBase
from mogp_emulator.Kernel import Kernel, SquaredExponential, Matern52
from mogp_emulator.Priors import Prior
from scipy import linalg
from scipy.optimize import OptimizeResult

import libgpgpu

from . import GaussianProcess



class UnavailableError(RuntimeError):
    """Exception type to use when a GPU, or the GPU library, is unavailable"""
    pass


class GaussianProcessGPU(object):
    """This class implements the same interface as
    :class:`mogp_emulator.GaussianProcess.GaussianProcess`, but with
    particular methods overridden to use a GPU if it is available.
    """

    def __init__(self, inputs, targets, mean=None, kernel=SquaredExponential(), priors=None,
                 nugget="adaptive", inputdict = {}, use_patsy=True):
        inputs = np.array(inputs)
        if inputs.ndim == 1:
            inputs = np.reshape(inputs, (-1, 1))
        assert inputs.ndim == 2

        targets = np.array(targets)
        assert targets.ndim == 1
        assert targets.shape[0] == inputs.shape[0]

        if mean:
            raise ValueError("GPU implementation requires mean to be None")

        if isinstance(kernel, str):
            if kernel == "SquaredExponential":
                kernel = SquaredExponential()
            else:
                raise ValueError("GPU implementation requires kernel to be SquaredExponential")
        elif kernel and kernel != SquaredExponential():
                raise ValueError("GPU implementation requires kernel to be SquaredExponential()")
        self._densegp_gpu = libgpgpu.DenseGP_GPU(inputs, targets)


    def fit(self, theta):

        theta = np.array(theta)
        self._densegp_gpu.update_theta(theta)
