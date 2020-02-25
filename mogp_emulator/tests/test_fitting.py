import numpy as np
import pytest
from numpy.testing import assert_allclose
from ..GaussianProcess import GaussianProcess
from ..fitting import fit_GP_MLE

def test_fit_GP_MLE():
    "test the fit_GP_MLE function"

    # test correct basic functioning

    x = np.linspace(0., 1.)
    y = x**2

    gp = GaussianProcess(x, y)

    theta_exp = np.array([ 1.6030532031342832, -2.090511425471982 , -0.7803960307137198])
    loglike_exp = -296.0297245831661
    np.random.seed(4335)

    gp = fit_GP_MLE(gp)

    assert isinstance(gp, GaussianProcess)
    assert_allclose(gp.theta, theta_exp)
    assert_allclose(gp.current_loglike, loglike_exp)

    # same test, but pass args and kwargs rather than gp

    np.random.seed(4335)

    gp = fit_GP_MLE(x, y, mean="0.", use_patsy=False, method="L-BFGS-B")
    assert isinstance(gp, GaussianProcess)
    assert_allclose(gp.theta, theta_exp)
    assert_allclose(gp.current_loglike, loglike_exp)

    # minimization fails

    with pytest.raises(RuntimeError):
        fit_GP_MLE(gp, n_tries=1, theta0=-10000.*np.ones(3))

    gp = GaussianProcess(x, y, nugget=0.)

    with pytest.raises(RuntimeError):
        fit_GP_MLE(gp, n_tries=1)

    # bad inputs

    with pytest.raises(TypeError):
        fit_GP_MLE(x)

    with pytest.raises(TypeError):
        fit_GP_MLE()

    with pytest.raises(AssertionError):
        fit_GP_MLE(gp, n_tries=-1)

    with pytest.raises(AssertionError):
        fit_GP_MLE(gp, theta0=np.ones(1))
