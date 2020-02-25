from .GaussianProcess import GaussianProcess
from .prediction import (default_container, predict_single_expectation,
                         VectorFloat)
import numpy as np


class GaussianProcessFPGA(GaussianProcess):
    def __init__(self, kernel_path, *args):
        super().__init__(*args)

        self.cl_container = default_container(kernel_path)

    def _predict_single(self, testing, do_deriv=True, do_unc=True):
        testing = np.array(testing)
        if len(testing.shape) == 1:
            testing = np.reshape(testing, (1, len(testing)))
        assert len(testing.shape) == 2

        n_testing, D = np.shape(testing)
        assert D == self.D

        var = None
        deriv = None

        if (do_deriv is False) and (do_unc is False):
            expectation = np.empty(n_testing)
            expectation = VectorFloat(expectation)
            testing = VectorFloat(testing.flatten())
            inputs = VectorFloat(self.inputs.flatten())
            scale = VectorFloat(self.theta[:-1])
            invQt = VectorFloat(self.invQt.flatten())

            predict_single_expectation(
                inputs, self.n, self.D,
                testing, n_testing,
                scale, self.theta[-1],
                invQt,
                expectation,
                self.cl_container
                )

            expectation = np.array(expectation)

        return expectation, var, deriv
