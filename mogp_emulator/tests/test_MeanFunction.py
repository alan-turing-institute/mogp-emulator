import numpy as np
import pytest
from numpy.testing import assert_allclose
from ..MeanFunction import MeanFunction, MeanSum, MeanProduct, FixedMean, ConstantMean, LinearMean
from ..MeanFunction import PolynomialMean

@pytest.fixture
def mf():
    return MeanFunction()

@pytest.fixture
def x():
    return np.array([[1., 2., 3.], [4., 5., 6.]])

@pytest.fixture
def params():
    return np.array([1., 2., 4.])

def test_MeanFunction(mf, x, params):
    "test composition of mean functions"

    mf2 = mf + mf

    assert isinstance(mf2, MeanSum)

    mf3 = 5. + mf

    assert isinstance(mf3, MeanSum)
    assert isinstance(mf3.f1, FixedMean)

    mf4 = mf + 3.

    assert isinstance(mf4, MeanSum)
    assert isinstance(mf4.f2, FixedMean)

    mf5 = mf*mf

    assert isinstance(mf5, MeanProduct)

    mf6 = 3.*mf

    assert isinstance(mf6, MeanProduct)
    assert isinstance(mf6.f1, FixedMean)

    mf7 = mf*3.

    assert isinstance(mf7, MeanProduct)
    assert isinstance(mf7.f2, FixedMean)

def test_MeanFunction_failures(mf, x, params):
    "test situations of MeanFunction where an exception should be raised"

    with pytest.raises(NotImplementedError):
        mf.get_n_params(x)

    with pytest.raises(NotImplementedError):
        mf.mean_f(x, params)

    with pytest.raises(NotImplementedError):
        mf.mean_deriv(x, params)

    with pytest.raises(NotImplementedError):
        mf.mean_hessian(x, params)

    with pytest.raises(NotImplementedError):
        mf.mean_inputderiv(x, params)

    with pytest.raises(TypeError):
        "3" + mf

    with pytest.raises(TypeError):
        mf + "3"

    with pytest.raises(TypeError):
        mf*"3"

    with pytest.raises(TypeError):
        "3"*mf

def test_FixedMean(x):
    "test the FixedMean function"

    params = np.zeros(0)

    f = lambda x: np.ones(x.shape[0])
    deriv = lambda x: np.zeros((x.shape[1], x.shape[0]))
    fixed_mean = FixedMean(f, deriv)

    assert fixed_mean.get_n_params(x) == 0

    assert_allclose(fixed_mean.mean_f(x, params), np.ones(x.shape[0]))
    assert_allclose(fixed_mean.mean_f(np.array([1., 2., 3.]), params), np.ones(3))
    assert_allclose(fixed_mean.mean_deriv(x, params), np.zeros((0, x.shape[0])))
    assert_allclose(fixed_mean.mean_hessian(x, params), np.zeros((0, 0, x.shape[0])))
    assert_allclose(fixed_mean.mean_inputderiv(x, params), np.zeros((x.shape[1], x.shape[0])))

def test_ConstantMean(x):
    "test the constant mean function"

    params = np.ones(1)

    constant_mean = ConstantMean()

    assert constant_mean.get_n_params(x) == 1

    assert_allclose(constant_mean.mean_f(x, params), np.broadcast_to(1., x.shape[0]))
    assert_allclose(constant_mean.mean_deriv(x, params), np.ones((1, x.shape[0])))
    assert_allclose(constant_mean.mean_hessian(x, params), np.zeros((1, 1, x.shape[0])))
    assert_allclose(constant_mean.mean_inputderiv(x, params), np.zeros((x.shape[1], x.shape[0])))

def test_LinearMean(x, params):
    "test the linear mean function"

    linear_mean = LinearMean()

    assert linear_mean.get_n_params(x) == 3

    assert_allclose(linear_mean.mean_f(x, params), np.sum(x*params, axis=1))
    assert_allclose(linear_mean.mean_deriv(x, params), np.transpose(x))
    assert_allclose(linear_mean.mean_hessian(x, params), np.zeros((3, 3, x.shape[0])))
    assert_allclose(linear_mean.mean_inputderiv(x, params),
                    np.broadcast_to(np.reshape(params, (-1, 1)), (x.shape[1], x.shape[0])))

def test_MeanSum(x):
    "test the MeanSum function"

    mf = FixedMean(3.) + LinearMean()

    params = np.ones(3)

    assert mf.get_n_params(x) == 3

    assert_allclose(mf.mean_f(x, params), [9., 18.])
    assert_allclose(mf.mean_deriv(x, params), np.transpose(x))
    assert_allclose(mf.mean_hessian(x, params), np.zeros((3, 3, x.shape[0])))
    assert_allclose(mf.mean_inputderiv(x, params), np.ones((x.shape[1], x.shape[0])))

    mf2 = ConstantMean() + LinearMean()

    params = np.ones(4)

    assert mf2.get_n_params(x) == 4

    assert_allclose(mf2.mean_f(x, params), [7., 16.])
    assert_allclose(mf2.mean_deriv(x, params), np.vstack((np.ones((2,)), np.transpose(x))))
    assert_allclose(mf2.mean_hessian(x, params), np.zeros((4, 4, x.shape[0])))
    assert_allclose(mf2.mean_inputderiv(x, params), np.ones((x.shape[1], x.shape[0])))

def test_MeanProduct(x):
    "test the MeanProduct function"

    mf = FixedMean(3.)*LinearMean()

    params = np.ones(3)

    assert mf.get_n_params(x) == 3

    assert_allclose(mf.mean_f(x, params), [18., 45.])
    assert_allclose(mf.mean_deriv(x, params), 3.*np.transpose(x))
    assert_allclose(mf.mean_hessian(x, params), np.zeros((3, 3, x.shape[0])))
    assert_allclose(mf.mean_inputderiv(x, params), 3.*np.ones((x.shape[1], x.shape[0])))

    mf2 = LinearMean()*LinearMean()

    params = np.ones(6)

    assert mf2.get_n_params(x) == 6

    hess_expected = np.zeros((6, 6, 2))
    hess_expected[3:6, 0:3, 0] = np.array([[1., 2., 3.], [2., 4., 6.], [3., 6., 9.]])
    hess_expected[0:3, 3:6, 0] = np.transpose([[1., 2., 3.], [2., 4., 6.], [3., 6., 9.]])
    hess_expected[3:6, 0:3, 1] = np.array([[16., 20., 24.], [20., 25., 30.], [24., 30., 36.]])
    hess_expected[0:3, 3:6, 1] = np.transpose([[16., 20., 24.], [20., 25., 30.], [24., 30., 36.]])

    inputderiv_expected = np.zeros((3, 2))
    inputderiv_expected[:, 0] = 12.
    inputderiv_expected[:, 1] = 30.

    assert_allclose(mf2.mean_f(x, params), [36., 225.])
    assert_allclose(mf2.mean_deriv(x, params), np.vstack((np.transpose(x), np.transpose(x)))*np.broadcast_to([6., 15.], (6, 2)))
    assert_allclose(mf2.mean_hessian(x, params), hess_expected)
    assert_allclose(mf2.mean_inputderiv(x, params), inputderiv_expected)

def test_PolynomialMean(x):
    "test the polynomial mean function"

    poly_mean = PolynomialMean(2)

    with pytest.raises(AssertionError):
        PolynomialMean(-1)

    params = np.arange(1., 8.)

    mean_expected = np.array([(params[0] + params[1]*x[0, 0] + params[2]*x[0, 1] + params[3]*x[0, 2]
                                         + params[4]*x[0, 0]**2 + params[5]*x[0, 1]**2 + params[6]*x[0, 2]**2),
                              (params[0] + params[1]*x[1, 0] + params[2]*x[1, 1] + params[3]*x[1, 2]
                                         + params[4]*x[1, 0]**2 + params[5]*x[1, 1]**2 + params[6]*x[1, 2]**2)])
    deriv_expected = np.array([[1., 1.], x[:, 0], x[:, 1], x[:, 2],
                                         x[:, 0]**2, x[:, 1]**2, x[:, 2]**2])
    hess_expected = np.zeros((7, 7, 2))
    inputderiv_expected = np.array([[params[1] + 2.*params[4]*x[0,0], params[1] + 2.*params[4]*x[1,0]],
                                    [params[2] + 2.*params[5]*x[0,1], params[2] + 2.*params[5]*x[1,1]],
                                    [params[3] + 2.*params[6]*x[0,2], params[3] + 2.*params[6]*x[1,2]]])

    assert_allclose(poly_mean.mean_f(x, params), mean_expected)
    assert_allclose(poly_mean.mean_deriv(x, params), deriv_expected)
    assert_allclose(poly_mean.mean_hessian(x, params), hess_expected)
    assert_allclose(poly_mean.mean_inputderiv(x, params), inputderiv_expected)