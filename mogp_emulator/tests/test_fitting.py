import numpy as np
import pytest
from numpy.testing import assert_allclose
from ..GaussianProcess import GaussianProcess
from ..MultiOutputGP import MultiOutputGP
from ..fitting import fit_GP_MAP, _fit_single_GP_MAP, _fit_MOGP_MAP

def test_fit_GP_MAP():
    "test the fit_GP_MAP function"

    # test correct basic functioning

    x = np.linspace(0., 1.)
    y = x**2

    gp = GaussianProcess(x, y)

    theta_exp = np.array([ 1.6030532031342832, -2.090511425471982 , -0.7803960307137198])
    logpost_exp = -296.0297245831661
    np.random.seed(4335)

    gp = fit_GP_MAP(gp)

    assert isinstance(gp, GaussianProcess)
    assert_allclose(gp.theta, theta_exp)
    assert_allclose(gp.current_logpost, logpost_exp)

    # same test, but pass args and kwargs rather than gp

    np.random.seed(4335)

    gp = fit_GP_MAP(x, y, mean="0.", use_patsy=False, method="L-BFGS-B")
    assert isinstance(gp, GaussianProcess)
    assert_allclose(gp.theta, theta_exp)
    assert_allclose(gp.current_logpost, logpost_exp)

    # minimization fails

    with pytest.raises(RuntimeError):
        fit_GP_MAP(gp, n_tries=1, theta0=-10000.*np.ones(3))

    gp = GaussianProcess(x, y, nugget=0.)

    with pytest.raises(RuntimeError):
        fit_GP_MAP(gp, n_tries=1)

    with pytest.raises(RuntimeError):
        fit_GP_MAP(gp, theta0 = np.array([800., 0., 0.]), n_tries=1)

    # bad inputs

    with pytest.raises(TypeError):
        fit_GP_MAP(x)

    with pytest.raises(TypeError):
        fit_GP_MAP()

    with pytest.raises(AssertionError):
        fit_GP_MAP(gp, n_tries=-1)

    with pytest.raises(AssertionError):
        fit_GP_MAP(gp, theta0=np.ones(1))


def test_fit_GP_MAP_MOGP():
    "test the fit_GP_MAP function with multiple outputs"

    x = np.linspace(0., 1.)
    y = np.zeros((2, 50))
    y[0] = x**2
    y[1] = 2. + x**3

    gp = MultiOutputGP(x, y)

    theta_exp = np.zeros((2, 3))
    theta_exp[0] = np.array([ 1.6030532031342832, -2.090511425471982 , -0.7803960307137198])
    theta_exp[1] = np.array([ 1.414112951818647 , -0.5974688573393576,  0.6857536842773265])
    logpost_exp = np.zeros(2)
    logpost_exp[0] = -296.0297245831661
    logpost_exp[1] = -250.06025683867367
    np.random.seed(4335)

    gp = fit_GP_MAP(gp, processes=1)

    assert isinstance(gp, MultiOutputGP)
    for i in range(2):
        assert_allclose(gp.emulators[i].theta, theta_exp[i])
        assert_allclose(gp.emulators[i].current_logpost, logpost_exp[i])

    # same test, but pass args and kwargs rather than gp

    np.random.seed(4335)

    gp = fit_GP_MAP(x, y, mean="0.", use_patsy=False, method="L-BFGS-B", processes=1)
    assert isinstance(gp, MultiOutputGP)
    for i in range(2):
        assert_allclose(gp.emulators[i].theta, theta_exp[i])
        assert_allclose(gp.emulators[i].current_logpost, logpost_exp[i])

    # pass processes argument

    np.random.seed(4335)

    gp = fit_GP_MAP(x, y, mean="0.", use_patsy=False, method="L-BFGS-B", processes=1)
    assert isinstance(gp, MultiOutputGP)

    # pass various theta0 arguments

    gp = fit_GP_MAP(x, y, theta0=np.zeros(3))

    gp = fit_GP_MAP(x, y, theta0=np.zeros((2, 3)))

    gp = fit_GP_MAP(x, y, theta0=[None, np.zeros(3)])

    # minimization fails

    with pytest.raises(RuntimeError):
        fit_GP_MAP(gp, n_tries=1, theta0=-10000.*np.ones(3))

    gp = MultiOutputGP(x, y, nugget=0.)

    with pytest.raises(RuntimeError):
        fit_GP_MAP(gp, n_tries=1)

    with pytest.raises(RuntimeError):
        fit_GP_MAP(gp, theta0 = np.array([800., 0., 0.]), n_tries=1)

    # bad inputs

    with pytest.raises(TypeError):
        fit_GP_MAP(x)

    with pytest.raises(TypeError):
        fit_GP_MAP()

    with pytest.raises(AssertionError):
        fit_GP_MAP(gp, n_tries=-1)

    with pytest.raises(AssertionError):
        fit_GP_MAP(gp, theta0=np.ones(1))

    with pytest.raises(AssertionError):
        fit_GP_MAP(gp, theta0=np.zeros((3, 3)))

    with pytest.raises(AssertionError):
        fit_GP_MAP(gp, theta0=np.zeros((2, 1)))

    with pytest.raises(AssertionError):
        fit_GP_MAP(gp, theta0=np.zeros((1, 1, 1)))

    with pytest.raises(AssertionError):
        fit_GP_MAP(gp, theta0=[None, None, None])

    with pytest.raises(AssertionError):
        fit_GP_MAP(gp, theta0=[np.zeros(1), np.zeros(3)], processes=1)

def test_fit_single_GP_MAP():
    "test the method to run the minimization algorithm on a GP class"

    x = np.linspace(0., 1.)
    y = x**2

    gp = GaussianProcess(x, y)

    theta_exp = np.array([ 1.6030532031342832, -2.090511425471982 , -0.7803960307137198])
    logpost_exp = -296.0297245831661
    np.random.seed(4335)

    gp = _fit_single_GP_MAP(gp)

    assert isinstance(gp, GaussianProcess)
    assert_allclose(gp.theta, theta_exp)
    assert_allclose(gp.current_logpost, logpost_exp)

    # minimization fails

    with pytest.raises(RuntimeError):
        _fit_single_GP_MAP(gp, n_tries=1, theta0=-10000.*np.ones(3))

    gp = GaussianProcess(x, y, nugget=0.)

    with pytest.raises(RuntimeError):
        _fit_single_GP_MAP(gp, n_tries=1)

    with pytest.raises(RuntimeError):
        _fit_single_GP_MAP(gp, theta0 = np.array([800., 0., 0.]), n_tries=1)

    # bad inputs

    with pytest.raises(AssertionError):
        _fit_single_GP_MAP(x)

    with pytest.raises(AssertionError):
        fit_GP_MAP(gp, n_tries=-1)

    with pytest.raises(AssertionError):
        fit_GP_MAP(gp, theta0=np.ones(1))

def test_fit_MOGP_MAP_MOGP():
    "test the fit_GP_MAP function with multiple outputs"

    x = np.linspace(0., 1.)
    y = np.zeros((2, 50))
    y[0] = x**2
    y[1] = 2. + x**3

    gp = MultiOutputGP(x, y)

    theta_exp = np.zeros((2, 3))
    theta_exp[0] = np.array([ 1.6030532031342832, -2.090511425471982 , -0.7803960307137198])
    theta_exp[1] = np.array([ 1.414112951818647 , -0.5974688573393576,  0.6857536842773265])
    logpost_exp = np.zeros(2)
    logpost_exp[0] = -296.0297245831661
    logpost_exp[1] = -250.06025683867367
    np.random.seed(4335)

    gp = _fit_MOGP_MAP(gp, processes=1)

    assert isinstance(gp, MultiOutputGP)
    for i in range(2):
        assert_allclose(gp.emulators[i].theta, theta_exp[i])
        assert_allclose(gp.emulators[i].current_logpost, logpost_exp[i])

    # pass processes argument

    np.random.seed(4335)

    gp = _fit_MOGP_MAP(gp, processes=1)
    assert isinstance(gp, MultiOutputGP)

    # pass various theta0 arguments

    gp = fit_GP_MAP(gp, theta0=np.zeros(3))

    gp = fit_GP_MAP(gp, theta0=np.zeros((2, 3)))

    gp = fit_GP_MAP(gp, theta0=[None, np.zeros(3)])

    # minimization fails

    with pytest.raises(RuntimeError):
        _fit_MOGP_MAP(gp, n_tries=1, theta0=-10000.*np.ones(3))

    gp = MultiOutputGP(x, y, nugget=0.)

    with pytest.raises(RuntimeError):
        _fit_MOGP_MAP(gp, n_tries=1)

    with pytest.raises(RuntimeError):
        _fit_MOGP_MAP(gp, theta0 = np.array([800., 0., 0.]), n_tries=1)

    # bad inputs

    with pytest.raises(AssertionError):
        _fit_MOGP_MAP(x)

    with pytest.raises(AssertionError):
        _fit_MOGP_MAP(gp, n_tries=-1)

    with pytest.raises(AssertionError):
        _fit_MOGP_MAP(gp, theta0=np.ones(1))

    with pytest.raises(AssertionError):
        _fit_MOGP_MAP(gp, theta0=np.zeros((3, 3)))

    with pytest.raises(AssertionError):
        _fit_MOGP_MAP(gp, theta0=np.zeros((2, 1)))

    with pytest.raises(AssertionError):
        _fit_MOGP_MAP(gp, theta0=np.zeros((1, 1, 1)))

    with pytest.raises(AssertionError):
        _fit_MOGP_MAP(gp, theta0=[None, None, None])

    with pytest.raises(AssertionError):
        _fit_MOGP_MAP(gp, theta0=[np.zeros(1), None], processes=1)

    with pytest.raises(AssertionError):
        _fit_MOGP_MAP(gp, processes=-1)