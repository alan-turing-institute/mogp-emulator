from .GaussianProcess import GaussianProcess
from .prediction import (default_container, predict_single_expectation,
                         predict_single_variance, predict_single_deriv,
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

        variance = None
        deriv = None

        testing = VectorFloat(testing.flatten())
        inputs = VectorFloat(self.inputs.flatten())
        scale = VectorFloat(self.theta[:-1])
        invQt = VectorFloat(self.invQt.flatten())

        expectation = np.empty(n_testing)
        expectation = VectorFloat(expectation)

        if (do_deriv is False) and (do_unc is False):
            predict_single_expectation(
                inputs, self.n, self.D,
                testing, n_testing,
                scale, self.theta[-1],
                invQt,
                expectation,
                self.cl_container
                )
        elif (do_deriv is False) and (do_unc is True):
            L = VectorFloat(self.L.flatten())

            variance = VectorFloat(np.empty(n_testing))

            predict_single_variance(
                inputs, self.n, self.D,
                testing, n_testing,
                scale, self.theta[-1],
                invQt, L,
                expectation, variance,
                self.cl_container
                )
            variance = np.array(variance)
        elif (do_deriv is True) and (do_unc is True):
            L = VectorFloat(self.L.flatten())

            variance = VectorFloat(np.empty(n_testing))
            deriv = VectorFloat(np.empty((n_testing, self.D)).flatten())

            predict_single_deriv(
                inputs, self.n, self.D,
                testing, n_testing,
                scale, self.theta[-1],
                invQt, L,
                expectation, variance, deriv,
                self.cl_container
                )
            variance = np.array(variance)
            deriv = np.array(deriv)
            deriv.reshape((n_testing, self.D))
        else:
            raise NotImplementedError

        expectation = np.array(expectation)

        return expectation, variance, deriv
