from tempfile import TemporaryFile
import numpy as np
import pytest
from numpy.testing import assert_allclose
from ..GaussianProcess import GaussianProcess, PredictResult
from ..MeanFunction import ConstantMean, LinearMean, MeanFunction
from ..Kernel import SquaredExponential, Matern52
from ..Priors import NormalPrior, GammaPrior, InvGammaPrior
from scipy import linalg

@pytest.fixture
def x():
    return np.array([[1., 2., 3.], [4., 5., 6.]])

@pytest.fixture
def y():
    return np.array([2., 4.])

def test_GaussianProcess_init(x, y):
    "Test function for correct functioning of the init method of GaussianProcess"

    gp = GaussianProcess(x, y)
    assert_allclose(x, gp.inputs)
    assert_allclose(y, gp.targets)
    assert gp.D == 3
    assert gp.n == 2
    assert gp.nugget == None

    gp = GaussianProcess(y, y)
    assert gp.inputs.shape == (2, 1)

    gp = GaussianProcess(x, y, nugget=1.e-12)
    assert_allclose(gp.nugget, 1.e-12)

    gp = GaussianProcess(x, y, mean=ConstantMean(1.), kernel=Matern52(), nugget="fit")

    gp = GaussianProcess(x, y, kernel="SquaredExponential")
    assert isinstance(gp.kernel, SquaredExponential)

    gp = GaussianProcess(x, y, kernel="Matern52")
    assert isinstance(gp.kernel, Matern52)

    gp = GaussianProcess(x, y, mean="a+b*x[0]", use_patsy=False)

    assert str(gp.mean) == "c + c*x[0]"

    gp = GaussianProcess(x, y, mean="c", inputdict={"c": 0})

    assert str(gp.mean) == "c + c*x[0]"

def test_GP_init_failures(x, y):
    "Tests that GaussianProcess fails correctly with bad inputs"

    with pytest.raises(AssertionError):
        gp = GaussianProcess(np.ones((2, 2, 2)), y)

    with pytest.raises(AssertionError):
        gp = GaussianProcess(x, x)

    with pytest.raises(AssertionError):
        gp = GaussianProcess(np.ones((2, 3)), np.ones(3))

    with pytest.raises(ValueError):
        gp = GaussianProcess(x, y, mean=1)

    with pytest.raises(ValueError):
        gp = GaussianProcess(x, y, kernel="blah")

    with pytest.raises(ValueError):
        gp = GaussianProcess(x, y, kernel=1)

    with pytest.raises(ValueError):
        gp = GaussianProcess(x, y, nugget="a")

def test_GaussianProcess_n_params(x, y):
    "test the get_n_params method of GaussianProcess"

    gp = GaussianProcess(x, y)
    assert gp.n_params == x.shape[1] + 2

    gp = GaussianProcess(x, y, mean="x[0]")
    assert gp.n_params == 2 + x.shape[1] + 2

def test_GaussianProcess_nugget(x, y):
    "Tests the get_nugget method of GaussianProcess"

    gp = GaussianProcess(x, y)
    assert gp.nugget is None
    assert gp.nugget_type == "adaptive"

    gp.nugget = "fit"
    assert gp.nugget is None
    assert gp.nugget_type == "fit"

    gp.nugget = 1.
    assert_allclose(gp.nugget, 1.)
    assert gp.nugget_type == "fixed"

    gp.nugget = 0
    assert_allclose(gp.nugget, 0.)
    assert gp.nugget_type == "fixed"

    with pytest.raises(TypeError):
        gp.nugget = [1]

    with pytest.raises(ValueError):
        gp.nugget = "blah"

    with pytest.raises(ValueError):
        gp.nugget = -1.


@pytest.mark.parametrize("mean,nugget,sn", [(None, 0., 1.), (None, "adaptive", 0.),
                                            (MeanFunction("x[0]"), "fit", np.log(1.e-6))])
def test_GaussianProcess_theta(x, y, mean, nugget, sn):
    "test the theta property of GaussianProcess (effectively the same as fit)"

    # zero mean, zero nugget

    gp = GaussianProcess(x, y, mean=mean, nugget=nugget)

    theta = np.ones(gp.n_params)
    theta[-1] = sn

    gp.theta = theta

    switch = gp.mean.get_n_params(x)

    if nugget == "adaptive" or nugget == 0.:
        assert gp.nugget == 0.
        noise = 0.
    else:
        assert_allclose(gp.nugget, np.exp(sn))
        noise = np.exp(sn)*np.eye(x.shape[0])
    Q = gp.kernel.kernel_f(x, x, theta[switch:-1]) + noise
    ym = y - gp.mean.mean_f(x, theta[:switch])

    L_expect = np.linalg.cholesky(Q)
    invQt_expect = np.linalg.solve(Q, ym)
    logpost_expect = 0.5*(np.log(np.linalg.det(Q)) +
                          np.dot(ym, invQt_expect) +
                          gp.n*np.log(2.*np.pi))

    assert_allclose(L_expect, gp.L)
    assert_allclose(invQt_expect, gp.invQt)
    assert_allclose(logpost_expect, gp.current_logpost)

def test_GaussianProcess_priors(x, y):
    "test that priors are set properly"

    gp = GaussianProcess(x, y)

    assert gp.priors == [None, None, None, None, None]

    gp = GaussianProcess(x, y, priors=None)

    assert gp.priors == [None, None, None, None, None]

    priors = []
    gp = GaussianProcess(x, y, priors=priors)

    assert gp.priors == [None, None, None, None, None]

    priors = [None, None, None, None, None]
    gp = GausianProcess(x, y)

    assert gp.priors == priors

    priors = [None, NormalPrior(2., 2.), None, None, NormalPrior(3., 1.)]
    gp = GausianProcess(x, y, priors=priors)

    assert gp.priors[:-1] == priors[:-1]
    assert gp.priors[-1] is None

    priors = [None, NormalPrior(2., 2.), None, None]
    gp = GausianProcess(x, y, priors=priors)

    assert gp.priors[:-1] == priors
    assert gp.priors[-1] is None

    priors = [None, None, None]

    with pytest.raises(AssertionError):
        GausianProcess(x, y, priors=priors)

    with pytest.raises(TypeError):
        GaussianProcess(x, y, priors=1.)

    priors = [1., 2., 3., 4., 5.]

    with pytest.raises(TypeError):
        GaussianProcess(x, y, priors=priors)

@pytest.mark.parametrize("mean,nugget,sn", [(None, 0., 1.), (None, "adaptive", 0.),
                                            (MeanFunction("x[0]"), "fit", np.log(1.e-6))])
def test_GaussianProcess_fit_logposterior(x, y, mean, nugget, sn):
    "test the fit and logposterior methods of GaussianProcess"

    # zero mean, zero nugget

    gp = GaussianProcess(x, y, mean=mean, nugget=nugget)

    theta = np.ones(gp.n_params)
    theta[-1] = sn

    gp.fit(theta)

    switch = gp.mean.get_n_params(x)

    if nugget == "adaptive" or nugget == 0.:
        assert gp.nugget == 0.
        noise = 0.
    else:
        assert_allclose(gp.nugget, np.exp(sn))
        noise = np.exp(sn)*np.eye(x.shape[0])
    Q = gp.kernel.kernel_f(x, x, theta[switch:-1]) + noise
    ym = y - gp.mean.mean_f(x, theta[:switch])

    L_expect = np.linalg.cholesky(Q)
    invQt_expect = np.linalg.solve(Q, ym)
    logpost_expect = 0.5*(np.log(np.linalg.det(Q)) +
                          np.dot(ym, invQt_expect) +
                          gp.n*np.log(2.*np.pi))

    assert_allclose(L_expect, gp.L)
    assert_allclose(invQt_expect, gp.invQt)
    assert_allclose(logpost_expect, gp.current_logpost)
    assert_allclose(logpost_expect, gp.logposterior(theta))

def test_GaussianProcess_logposterior(x, y):
    "test logposterior method of GaussianProcess"

    # logposterior already tested, but check that parameters are re-fit if changed

    gp = GaussianProcess(x, y, nugget = 0.)

    theta = np.ones(gp.n_params)
    gp.fit(theta)

    theta = np.zeros(gp.n_params)

    Q = gp.kernel.kernel_f(x, x, theta[:-1])

    L_expect = np.linalg.cholesky(Q)
    invQt_expect = np.linalg.solve(Q, y)
    logpost_expect = 0.5*(np.log(np.linalg.det(Q)) +
                          np.dot(y, invQt_expect) +
                          gp.n*np.log(2.*np.pi))

    assert_allclose(logpost_expect, gp.logposterior(theta))
    assert_allclose(gp.L, L_expect)
    assert_allclose(invQt_expect, gp.invQt)
    assert_allclose(logpost_expect, gp.current_logpost)

@pytest.fixture
def dx():
    return 1.e-6

@pytest.mark.parametrize("mean,nugget,sn", [(None, 0., 1.), (None, "adaptive", 1.),
                                            ("x[0]", "fit", np.log(1.e-6))])
def test_GaussianProcess_logpost_deriv(x, y, dx, mean, nugget, sn):
    "test logposterior derivatives for GaussianProcess via finite differences"

    gp = GaussianProcess(x, y, mean=mean, nugget=nugget)

    n = gp.n_params
    theta = np.ones(n)
    theta[-1] = sn

    deriv = np.zeros(n)

    for i in range(n):
        dx_array = np.zeros(n)
        dx_array[i] = dx
        deriv[i] = (gp.logposterior(theta) - gp.logposterior(theta - dx_array))/dx

    assert_allclose(deriv, gp.logpost_deriv(theta), atol=1.e-7, rtol=1.e-5)

@pytest.mark.parametrize("mean,nugget,sn", [(None, 0., 1.), ("x[0]", "fit", np.log(1.e-6))])
def test_GaussianProcess_logpost_hessian(x, y, dx, mean, nugget, sn):
    "test the hessian method of GaussianProcess with finite differences"

    # zero mean, no nugget

    gp = GaussianProcess(x, y, nugget=0.)

    n = gp.n_params
    theta = np.ones(n)
    theta[-1] = sn

    hess = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            dx_array = np.zeros(n)
            dx_array[j] = dx
            hess[i, j] = (gp.logpost_deriv(theta)[i] - gp.logpost_deriv(theta - dx_array)[i])/dx

    assert_allclose(hess, gp.logpost_hessian(theta), rtol=1.e-5, atol=1.e-7)

@pytest.mark.parametrize("priors,nugget,sn", [([ NormalPrior(0.9, 0.5), None, NormalPrior(0.5, 2.),
                                              InvGammaPrior(2., 1.), None], 0., 0.),
                                           ([ None, NormalPrior(1.2, 0.2), None,
                                              GammaPrior(2., 1.), InvGammaPrior(2., 1.)], "fit", np.log(1.e-6))])
def test_GaussianProcess_priors(x, y, dx, priors, nugget, sn):
    "test that prior distributions are properly accounted for in posterior"

    gp = GaussianProcess(x, y, priors=priors, nugget=nugget)

    theta = np.ones(gp.n_params)
    theta[-1] = sn

    gp.fit(theta)

    if nugget == 0.:
        noise = 0.
    else:
        assert_allclose(gp.nugget, np.exp(sn))
        noise = np.exp(sn)*np.eye(x.shape[0])
    Q = gp.get_K_matrix() + noise

    L_expect = np.linalg.cholesky(Q)
    invQt_expect = np.linalg.solve(Q, y)
    logpost_expect = 0.5*(np.log(np.linalg.det(Q)) +
                          np.dot(y, invQt_expect) +
                          gp.n*np.log(2.*np.pi))

    for p, t in zip(priors, theta):
        if not p is None:
            logpost_expect -= p.logp(t)

    assert_allclose(L_expect, gp.L)
    assert_allclose(invQt_expect, gp.invQt)
    assert_allclose(logpost_expect, gp.current_logpost)
    assert_allclose(logpost_expect, gp.logposterior(theta))

    n = gp.n_params
    deriv = np.zeros(n)

    for i in range(n):
        dx_array = np.zeros(n)
        dx_array[i] = dx
        deriv[i] = (gp.logposterior(theta) - gp.logposterior(theta - dx_array))/dx

    assert_allclose(deriv, gp.logpost_deriv(theta), atol=1.e-5, rtol=1.e-4)

    hess = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            dx_array = np.zeros(n)
            dx_array[j] = dx
            hess[i, j] = (gp.logpost_deriv(theta)[i] - gp.logpost_deriv(theta - dx_array)[i])/dx

    assert_allclose(hess, gp.logpost_hessian(theta), rtol=1.e-5, atol=1.e-5)

def test_GaussianProcess_predict(x, y, dx):
    "test the predict method of GaussianProcess"

    # zero mean

    gp = GaussianProcess(x, y, nugget=0.)
    theta = np.ones(gp.n_params)

    gp.fit(theta)

    x_test = np.array([[2., 3., 4.]])

    mu, var, deriv = gp.predict(x_test)

    K = gp.kernel.kernel_f(x, x, theta[:-1])
    Ktest = gp.kernel.kernel_f(x_test, x, theta[:-1])

    mu_expect = np.dot(Ktest, gp.invQt)
    var_expect = np.exp(theta[-2]) - np.diag(np.dot(Ktest, np.linalg.solve(K, Ktest.T)))

    D = gp.D

    deriv_expect = np.zeros((1, D))

    for i in range(D):
        dx_array = np.zeros(D)
        dx_array[i] = dx
        deriv_expect[0, i] = (gp.predict(x_test)[0] - gp.predict(x_test - dx_array)[0])/dx

    assert_allclose(mu, mu_expect)
    assert_allclose(var, var_expect)
    assert_allclose(deriv, deriv_expect, atol=1.e-7, rtol=1.e-7)

    # check that reshaping works as expected

    x_test = np.array([2., 3., 4.])

    mu, var, deriv = gp.predict(x_test)

    assert_allclose(mu, mu_expect)
    assert_allclose(var, var_expect)
    assert_allclose(deriv, deriv_expect, atol=1.e-7, rtol=1.e-7)

    # check that with 1D input data can handle 1D prediction data correctly

    gp = GaussianProcess(y, y, nugget=0.)

    gp.fit(np.ones(gp.n_params))

    n_predict = 51
    mu, var, deriv = gp.predict(np.linspace(0., 1., n_predict))

    assert mu.shape == (n_predict,)
    assert var.shape == (n_predict,)
    assert deriv.shape == (n_predict, 1)

    # nonzero mean function

    gp = GaussianProcess(x, y, mean="x[0]", nugget=0.)

    theta = np.ones(gp.n_params)

    gp.fit(theta)

    x_test = np.array([[2., 3., 4.]])

    mu, var, deriv = gp.predict(x_test)

    switch = gp.mean.get_n_params(x)
    m = gp.mean.mean_f(x_test, theta[:switch])
    K = gp.kernel.kernel_f(x, x, theta[switch:-1])
    Ktest = gp.kernel.kernel_f(x_test, x, theta[switch:-1])

    mu_expect = m + np.dot(Ktest, gp.invQt)
    var_expect = np.exp(theta[-2]) - np.diag(np.dot(Ktest, np.linalg.solve(K, Ktest.T)))

    D = gp.D

    deriv_expect = np.zeros((1, D))

    for i in range(D):
        dx_array = np.zeros(D)
        dx_array[i] = dx
        deriv_expect[0, i] = (gp.predict(x_test)[0] - gp.predict(x_test - dx_array)[0])/dx

    assert_allclose(mu, mu_expect)
    assert_allclose(var, var_expect)
    assert_allclose(deriv, deriv_expect, atol=1.e-7, rtol=1.e-7)

    # check unc and deriv flags work

    _, var, deriv = gp.predict(x_test, unc=False, deriv=False)

    assert var is None
    assert deriv is None

    # check that the returned PredictResult works correctly

    pr = gp.predict(x_test)

    assert_allclose(pr.mean, mu_expect)
    assert_allclose(pr.unc, var_expect)
    assert_allclose(pr.deriv, deriv_expect, atol=1.e-7, rtol=1.e-7)

    assert_allclose(pr['mean'], mu_expect)
    assert_allclose(pr['unc'], var_expect)
    assert_allclose(pr['deriv'], deriv_expect, atol=1.e-7, rtol=1.e-7)

    assert_allclose(pr[0], mu_expect)
    assert_allclose(pr[1], var_expect)
    assert_allclose(pr[2], deriv_expect, atol=1.e-7, rtol=1.e-7)

    # check that calling gp is equivalent to predicting

    assert_allclose(gp(x_test), mu_expect)

def test_GaussianProcess_predict_nugget(x, y):
    "test that the nugget works correctly when making predictions"

    nugget = 1.e0

    gp = GaussianProcess(x, y, nugget=nugget)
    theta = np.ones(gp.n_params)

    gp.fit(theta)

    preds = gp.predict(x)

    K = gp.kernel.kernel_f(x, x, theta[:-1])

    var_expect = np.exp(theta[-2]) + nugget - np.diag(np.dot(K, np.linalg.solve(K + np.eye(gp.n)*nugget, K)))

    assert_allclose(preds.unc, var_expect, atol=1.e-7)

    preds = gp.predict(x, include_nugget=False)

    var_expect = np.exp(theta[-2]) - np.diag(np.dot(K, np.linalg.solve(K + np.eye(gp.n)*nugget, K)))

    assert_allclose(preds.unc, var_expect, atol=1.e-7)

def test_GaussianProcess_predict_variance():
    "confirm that caching factorized matrix produces stable variance predictions"

    x = np.linspace(0., 5., 21)
    y = x**2
    x = np.reshape(x, (-1, 1))
    nugget = 1.e-8

    gp = GaussianProcess(x, y, nugget=nugget)

    theta = np.array([-7.352408190715323, 15.041447753599755, 0.])
    gp.fit(theta)

    testing = np.reshape(np.linspace(0., 5., 101), (-1, 1))

    _, var, _ = gp.predict(testing)

    assert_allclose(np.zeros(101), var, atol = 1.e-3)

def test_GaussianProcess_predict_failures(x, y):
    "test situations where predict method of GaussianProcess should fail"

    gp = GaussianProcess(x, y)

    with pytest.raises(ValueError):
        gp.predict(np.array([2., 3., 4.]))

    theta = np.ones(gp.n_params)
    gp.fit(theta)

    with pytest.raises(AssertionError):
        gp.predict(np.ones((2, 2, 2)))

    with pytest.raises(AssertionError):
        gp.predict(np.array([[2., 4.]]))

def test_GaussianProcess_str(x, y):
    "Test function for string method"

    gp = GaussianProcess(x, y)
    assert (str(gp) == "Gaussian Process with {} training examples and {} input variables".format(x.shape[0], x.shape[1]))