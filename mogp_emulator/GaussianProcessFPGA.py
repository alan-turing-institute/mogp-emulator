from .GaussianProcess import GaussianProcess
from .prediction import create_cl_container, predict_single, VectorFloat
import numpy as np


class GaussianProcessFPGA(GaussianProcess):
    def __init__(self, kernel_path, *args):
        super().__init__(*args)

        self.cl_container = create_cl_container(kernel_path)

    def _predict_single(self, testing, do_deriv=True, do_unc=True):
        testing = np.array(testing)
        if len(testing.shape) == 1:
            testing = np.reshape(testing, (1, len(testing)))
        assert len(testing.shape) == 2

        n_testing, D = np.shape(testing)
        assert D == self.D

        # Convert arguments to VectorFloat objects
        testing = VectorFloat(testing.flatten())
        inputs = VectorFloat(self.inputs.flatten())
        scale = VectorFloat(self.theta[:-1])
        invQt = VectorFloat(self.invQt.flatten())
        L = VectorFloat(self.L.flatten())

        # Initialise results
        expectation = VectorFloat(np.empty(n_testing))
        variance = VectorFloat(np.empty(n_testing))
        deriv = VectorFloat(np.empty((n_testing, self.D)).flatten())

        # Call kernel wrapper
        predict_single(
            inputs, self.n, self.D,
            testing, n_testing,
            scale, self.theta[-1],
            invQt, L,
            expectation, variance, deriv,
            self.cl_container,
            do_unc, do_deriv
            )

        # Cast results to np.arrays
        expectation = np.array(expectation)

        if do_unc:
            variance = np.array(variance)
        else:
            variance = None

        if do_deriv:
            deriv = np.array(deriv)
            deriv.reshape((n_testing, self.D))
        else:
            deriv = None

        return expectation, variance, deriv
