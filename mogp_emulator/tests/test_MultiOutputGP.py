import numpy as np
import pytest
from numpy.testing import assert_allclose
from ..GaussianProcess import GaussianProcess, PredictResult
from ..MultiOutputGP import MultiOutputGP
from ..MeanFunction import ConstantMean, LinearMean, MeanFunction
from ..Kernel import Matern52
from scipy import linalg

@pytest.fixture
def x():
    return np.array([[1., 2., 3.], [4., 5., 6.]])

@pytest.fixture
def y():
    return np.array([[2., 4.], [3., 5.]])

def test_MultiOutputGP_init(x, y):
    "Test function for correct functioning of the init method of GaussianProcess"

    gp = MultiOutputGP(x, y)
    assert_allclose(x, gp.emulators[0].inputs)
    assert_allclose(x, gp.emulators[1].inputs)
    assert_allclose(y[0], gp.emulators[0].targets)
    assert_allclose(y[1], gp.emulators[1].targets)
    assert gp.D == 3
    assert gp.n == 2
    assert gp.n_emulators == 2

@pytest.fixture
def dx():
    return 1.e-6

def test_MultiOutputGP_predict(x, y, dx):
    "test the predict method of GaussianProcess"

    gp = MultiOutputGP(x, y, nugget=0.)
    theta = np.ones(gp.emulators[0].n_params)

    gp.emulators[0].fit(theta)
    gp.emulators[1].fit(theta)

    x_test = np.array([[2., 3., 4.]])

    mu, var, deriv = gp.predict(x_test)

    for i in range(2):

        K = gp.emulators[i].kernel.kernel_f(x, x, theta[:-1])
        Ktest = gp.emulators[i].kernel.kernel_f(x_test, x, theta[:-1])

        mu_expect = np.dot(Ktest, gp.emulators[i].invQt)
        var_expect = np.exp(theta[-2]) - np.diag(np.dot(Ktest, np.linalg.solve(K, Ktest.T)))

        D = gp.D

        deriv_expect = np.zeros((1, D))

        for j in range(D):
            dx_array = np.zeros(D)
            dx_array[j] = dx
            deriv_expect[0, j] = (gp.emulators[i].predict(x_test)[0] - gp.emulators[i].predict(x_test - dx_array)[0])/dx

        assert_allclose(mu[i], mu_expect)
        assert_allclose(var[i], var_expect)
        assert_allclose(deriv[i], deriv_expect, atol=1.e-6, rtol=1.e-6)

    nugget = 1.
    gp = MultiOutputGP(x, y, nugget=nugget)
    theta = np.ones(gp.emulators[0].n_params)

    gp.emulators[0].fit(theta)
    gp.emulators[1].fit(theta)

    x_test = np.array([[2., 3., 4.]])

    mu, var, deriv = gp.predict(x_test)

    for i in range(2):

        K = gp.emulators[i].kernel.kernel_f(x, x, theta[:-1]) + np.eye(gp.emulators[i].n)*nugget
        Ktest = gp.emulators[i].kernel.kernel_f(x_test, x, theta[:-1])

        var_expect = np.exp(theta[-2]) + nugget - np.diag(np.dot(Ktest, np.linalg.solve(K, Ktest.T)))

        assert_allclose(var[i], var_expect)

    mu, var, deriv = gp.predict(x_test, include_nugget=False)

    for i in range(2):

        K = gp.emulators[i].kernel.kernel_f(x, x, theta[:-1]) + np.eye(gp.emulators[i].n)*nugget
        Ktest = gp.emulators[i].kernel.kernel_f(x_test, x, theta[:-1])

        var_expect = np.exp(theta[-2]) - np.diag(np.dot(Ktest, np.linalg.solve(K, Ktest.T)))

        assert_allclose(var[i], var_expect)
