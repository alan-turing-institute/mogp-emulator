import numpy as np
import pytest
from numpy.testing import assert_allclose
from ..Kernel import Kernel, SquaredExponential, Matern52

def test_calc_r2():
    "test function for calc_r2 function for kernels"

    k = Kernel()

    x = np.array([[1.], [2.]])
    y = np.array([[2.], [3.]])
    params = np.array([0.])

    assert_allclose(k.calc_r2(x, y, params), np.array([[1., 4.], [0., 1.]]))

    x = np.array([[1., 2.], [2., 3.]])
    y = np.array([[2., 4.], [3., 1.]])
    params = np.array([0., 0.])

    assert_allclose(k.calc_r2(x, y, params),
                    np.array([[5., 5.], [1., 5.]]))

    x = np.array([[1., 2.], [2., 3.]])
    y = np.array([[2., 4.], [3., 1.]])
    params = np.array([np.log(2.), np.log(4.)])

    assert_allclose(k.calc_r2(x, y, params),
                    np.array([[1.*2.+4.*4., 4.*2.+1.*4.],
                              [1.*4.,       1.*2.+4.*4.]]))

    x = np.array([1., 2.])
    y = np.array([2., 3.])
    params = np.array([0.])

    assert_allclose(k.calc_r2(x, y, params), np.array([[1., 4.], [0., 1.]]))

def test_calc_r_failures():
    "test scenarios where calc_r should raise an exception"

    k = Kernel()

    x = np.array([[1.], [2.]])
    y = np.array([[2.], [3.]])
    params = np.array([0., 0.])

    with pytest.raises(AssertionError):
        k.calc_r2(x, y, params)

    params = np.array([[0., 0.], [0., 0.]])

    with pytest.raises(AssertionError):
        k.calc_r2(x, y, params)

    x = np.array([[1.], [2.]])
    y = np.array([[2., 4.], [3., 2.]])
    params = np.array([0.])

    with pytest.raises(AssertionError):
        k.calc_r2(x, y, params)

    x = np.array([[1.], [2.]])
    y = np.array([[[2.], [4.]], [[3.], [2.]]])
    params = np.array([0.])

    with pytest.raises(AssertionError):
        k.calc_r2(x, y, params)

    x = np.array([[2., 4.], [3., 2.]])
    y = np.array([[1.], [2.]])
    params = np.array([0.])

    with pytest.raises(AssertionError):
        k.calc_r2(x, y, params)

    x = np.array([[[2.], [4.]], [[3.], [2.]]])
    y = np.array([[1.], [2.]])
    params = np.array([0.])

    with pytest.raises(AssertionError):
        k.calc_r2(x, y, params)

    with pytest.raises(FloatingPointError):
        k.calc_r2(y, y, np.array([800.]))

def test_calc_dr2dtheta():
    "test calc_drdtheta function"

    k = Kernel()

    dx = 1.e-6

    x = np.array([[1.], [2.]])
    y = np.array([[2.], [3.]])
    params = np.array([0.])

    deriv_fd = np.zeros((1, 2, 2))
    deriv_fd[0] = (k.calc_r2(x, y, params) -
                   k.calc_r2(x, y, params - dx))/dx

    assert_allclose(k.calc_dr2dtheta(x, y, params), deriv_fd, rtol = 1.e-5)

    x = np.array([[1., 2.], [2., 3.]])
    y = np.array([[2., 4.], [3., 1.]])
    params = np.array([0., 0.])

    deriv_fd = np.zeros((2, 2, 2))
    deriv_fd[0] = (k.calc_r2(x, y, params) -
                   k.calc_r2(x, y, params - np.array([dx, 0.])))/dx
    deriv_fd[1] = (k.calc_r2(x, y, params) -
                   k.calc_r2(x, y, params - np.array([0., dx])))/dx

    assert_allclose(k.calc_dr2dtheta(x, y, params), deriv_fd, rtol = 1.e-5)

    deriv = np.zeros((2, 2, 2))
    x = np.array([[1., 2.], [2., 3.]])
    y = np.array([[2., 4.], [3., 1.]])
    params = np.array([np.log(2.), np.log(4.)])

    deriv_fd = np.zeros((2, 2, 2))
    deriv_fd[0] = (k.calc_r2(x, y, params) -
                   k.calc_r2(x, y, params - np.array([dx, 0.])))/dx
    deriv_fd[1] = (k.calc_r2(x, y, params) -
                   k.calc_r2(x, y, params - np.array([0., dx])))/dx

    assert_allclose(k.calc_dr2dtheta(x, y, params), deriv_fd, rtol = 1.e-5)

    x = np.array([1., 2.])
    y = np.array([2., 3.])
    params = np.array([0.])

    deriv_fd = np.zeros((1, 2, 2))
    deriv_fd[0] = (k.calc_r2(x, y, params) -
                   k.calc_r2(x, y, params - dx))/dx

    assert_allclose(k.calc_dr2dtheta(x, y, params), deriv_fd, rtol = 1.e-5)

def test_calc_drdtheta_failures():
    "test situations where calc_drdtheta should fail"

    k = Kernel()

    x = np.array([[1.], [2.]])
    y = np.array([[2.], [3.]])
    params = np.array([0., 0.])

    with pytest.raises(AssertionError):
        k.calc_dr2dtheta(x, y, params)

    params = np.array([[0., 0.], [0., 0.]])

    with pytest.raises(AssertionError):
        k.calc_dr2dtheta(x, y, params)

    x = np.array([[1.], [2.]])
    y = np.array([[2., 4.], [3., 2.]])
    params = np.array([0.])

    with pytest.raises(AssertionError):
        k.calc_dr2dtheta(x, y, params)

    x = np.array([[1.], [2.]])
    y = np.array([[[2.], [4.]], [[3.], [2.]]])
    params = np.array([0.])

    with pytest.raises(AssertionError):
        k.calc_dr2dtheta(x, y, params)

    x = np.array([[2., 4.], [3., 2.]])
    y = np.array([[1.], [2.]])
    params = np.array([0.])

    with pytest.raises(AssertionError):
        k.calc_dr2dtheta(x, y, params)

    x = np.array([[[2.], [4.]], [[3.], [2.]]])
    y = np.array([[1.], [2.]])
    params = np.array([0.])

    with pytest.raises(AssertionError):
        k.calc_dr2dtheta(x, y, params)

def test_calc_d2rdtheta2():
    "test calc_d2rdtheta2 function"

    k = Kernel()

    dx = 1.e-6

    x = np.array([[1.], [2.]])
    y = np.array([[2.], [3.]])
    params = np.array([0.])

    deriv_fd = np.zeros((1, 1, 2, 2))
    deriv_fd[0, 0] = (k.calc_dr2dtheta(x, y, params)[0] -
                      k.calc_dr2dtheta(x, y, params - np.array([dx]))[0])/dx

    assert_allclose(k.calc_d2r2dtheta2(x, y, params), deriv_fd, rtol = 1.e-5)

    x = np.array([[1., 2.], [2., 3.]])
    y = np.array([[2., 4.], [3., 1.]])
    params = np.array([0., 0.])

    deriv_fd = np.zeros((2, 2, 2, 2))
    deriv_fd[0, 0] = (k.calc_dr2dtheta(x, y, params)[0] -
                      k.calc_dr2dtheta(x, y, params - np.array([dx, 0.]))[0])/dx
    deriv_fd[0, 1] = (k.calc_dr2dtheta(x, y, params)[1] -
                      k.calc_dr2dtheta(x, y, params - np.array([dx, 0.]))[1])/dx
    deriv_fd[1, 0] = (k.calc_dr2dtheta(x, y, params)[0] -
                      k.calc_dr2dtheta(x, y, params - np.array([0., dx]))[0])/dx
    deriv_fd[1, 1] = (k.calc_dr2dtheta(x, y, params)[1] -
                      k.calc_dr2dtheta(x, y, params - np.array([0., dx]))[1])/dx

    assert_allclose(k.calc_d2r2dtheta2(x, y, params), deriv_fd, rtol = 1.e-5)

    x = np.array([[1., 2.], [2., 3.]])
    y = np.array([[2., 4.], [3., 1.]])
    params = np.array([np.log(2.), np.log(4.)])

    deriv_fd = np.zeros((2, 2, 2, 2))
    deriv_fd[0, 0] = (k.calc_dr2dtheta(x, y, params)[0] -
                      k.calc_dr2dtheta(x, y, params - np.array([dx, 0.]))[0])/dx
    deriv_fd[0, 1] = (k.calc_dr2dtheta(x, y, params)[1] -
                      k.calc_dr2dtheta(x, y, params - np.array([dx, 0.]))[1])/dx
    deriv_fd[1, 0] = (k.calc_dr2dtheta(x, y, params)[0] -
                      k.calc_dr2dtheta(x, y, params - np.array([0., dx]))[0])/dx
    deriv_fd[1, 1] = (k.calc_dr2dtheta(x, y, params)[1] -
                      k.calc_dr2dtheta(x, y, params - np.array([0., dx]))[1])/dx

    assert_allclose(k.calc_d2r2dtheta2(x, y, params), deriv_fd, rtol = 1.e-5)

    x = np.array([1., 2.])
    y = np.array([2., 3.])
    params = np.array([0.])

    r = np.array([[1., 2.], [1., 1.]])

    deriv_fd = np.zeros((1, 1, 2, 2))
    deriv_fd[0, 0] = (k.calc_dr2dtheta(x, y, params)[0] -
                      k.calc_dr2dtheta(x, y, params - np.array([dx]))[0])/dx

    assert_allclose(k.calc_d2r2dtheta2(x, y, params), deriv_fd, rtol = 1.e-5)

def test_calc_d2rdtheta2_failures():
    "test situations where calc_d2rdtheta2 should fail"

    k = Kernel()

    x = np.array([[1.], [2.]])
    y = np.array([[2.], [3.]])
    params = np.array([0., 0.])

    with pytest.raises(AssertionError):
        k.calc_d2r2dtheta2(x, y, params)

    params = np.array([[0., 0.], [0., 0.]])

    with pytest.raises(AssertionError):
        k.calc_d2r2dtheta2(x, y, params)

    x = np.array([[1.], [2.]])
    y = np.array([[2., 4.], [3., 2.]])
    params = np.array([0.])

    with pytest.raises(AssertionError):
        k.calc_d2r2dtheta2(x, y, params)

    x = np.array([[1.], [2.]])
    y = np.array([[[2.], [4.]], [[3.], [2.]]])
    params = np.array([0.])

    with pytest.raises(AssertionError):
        k.calc_d2r2dtheta2(x, y, params)

    x = np.array([[2., 4.], [3., 2.]])
    y = np.array([[1.], [2.]])
    params = np.array([0.])

    with pytest.raises(AssertionError):
        k.calc_d2r2dtheta2(x, y, params)

    x = np.array([[[2.], [4.]], [[3.], [2.]]])
    y = np.array([[1.], [2.]])
    params = np.array([0.])

    with pytest.raises(AssertionError):
        k.calc_d2r2dtheta2(x, y, params)

def test_squared_exponential_K():
    "test squared exponential K(r) function"

    k = SquaredExponential()

    assert_allclose(k.calc_K(1.), np.exp(-0.5))

    assert_allclose(k.calc_K(np.array([[1., 2.], [3., 4.]])),
                             np.exp(-0.5*np.array([[1., 2.], [3., 4.]])))

    with pytest.raises(AssertionError):
        k.calc_K(-1.)

def test_squared_exponential_dKdr2():
    "test squared exponential dK/dr2 function"

    k = SquaredExponential()

    dx = 1.e-6

    assert_allclose(k.calc_dKdr2(1.), (k.calc_K(1.)-k.calc_K(1.-dx))/dx, rtol = 1.e-5)

    r = np.array([[1., 2.], [3., 4.]])

    assert_allclose(k.calc_dKdr2(r), (k.calc_K(r)-k.calc_K(r-dx))/dx, rtol = 1.e-5)

    with pytest.raises(AssertionError):
        k.calc_dKdr2(-1.)

def test_squared_exponential_d2Kdr22():
    "test squared exponential d2K/dr22 function"

    k = SquaredExponential()

    dx = 1.e-6

    assert_allclose(k.calc_d2Kdr22(1.),
                    (k.calc_dKdr2(1.)-k.calc_dKdr2(1.-dx))/dx, atol = 1.e-5)

    r = np.array([[1., 2.], [3., 4.]])

    assert_allclose(k.calc_d2Kdr22(r),
                    (k.calc_dKdr2(r)-k.calc_dKdr2(r-dx))/dx, rtol = 1.e-5, atol = 1.e-5)

    with pytest.raises(AssertionError):
        k.calc_d2Kdr22(-1.)

def test_squared_exponential():
    "test squared exponential covariance kernel"

    k = SquaredExponential()

    x = np.array([[1.], [2.]])
    y = np.array([[2.], [3.]])
    params = np.array([0.])

    assert_allclose(k.kernel_f(x, y, params),
                    np.exp(-0.5*np.array([[1., 2.], [0., 1.]])**2))

    x = np.array([[1., 2.], [2., 3.]])
    y = np.array([[2., 4.], [3., 1.]])
    params = np.array([0., 0.])

    assert_allclose(k.kernel_f(x, y, params),
                    np.exp(-0.5*np.array([[np.sqrt(5.), np.sqrt(5.)],
                                          [1., np.sqrt(5.)]])**2))

    x = np.array([[1., 2.], [2., 3.]])
    y = np.array([[2., 4.], [3., 1.]])
    params = np.array([np.log(2.), np.log(4.)])

    assert_allclose(k.kernel_f(x, y, params),
                    np.exp(-0.5*np.array([[np.sqrt(1.*2.+4.*4.), np.sqrt(4.*2.+1.*4.)],
                                          [np.sqrt(1.*4.), np.sqrt(1.*2.+4.*4.)]])**2))

    x = np.array([1., 2.])
    y = np.array([2., 3.])
    params = np.array([0.])

    assert_allclose(k.kernel_f(x, y, params),
                    np.exp(-0.5*np.array([[1., 2.], [0., 1.]])**2))

def test_squared_exponential_failures():
    "test scenarios where squared_exponential should raise an exception"

    k = SquaredExponential()

    x = np.array([[1.], [2.]])
    y = np.array([[2.], [3.]])
    params = np.array([0., 0.])

    with pytest.raises(AssertionError):
        k.kernel_f(x, y, params)

    params = np.array([[0., 0.], [0., 0.]])

    with pytest.raises(AssertionError):
        k.kernel_f(x, y, params)

def test_squared_exponential_deriv():
    "test the computation of the gradient of the squared exponential kernel"

    k = SquaredExponential()

    dx = 1.e-6

    x = np.array([[1.], [2.]])
    y = np.array([[2.], [3.]])
    params = np.array([0.])

    deriv_fd = np.zeros((1, 2, 2))
    deriv_fd[0] = (k.kernel_f(x, y, params) -
                   k.kernel_f(x, y, params - np.array([dx])))/dx

    assert_allclose(k.kernel_deriv(x, y, params), deriv_fd, rtol = 1.e-5)

    x = np.array([[1., 2.], [2., 3.]])
    y = np.array([[2., 4.], [3., 1.]])
    params = np.array([0., 0.])

    deriv_fd = np.zeros((2, 2, 2))
    deriv_fd[0] = (k.kernel_f(x, y, params) -
                   k.kernel_f(x, y, params - np.array([dx, 0.])))/dx
    deriv_fd[1] = (k.kernel_f(x, y, params) -
                   k.kernel_f(x, y, params - np.array([0., dx])))/dx

    assert_allclose(k.kernel_deriv(x, y, params), deriv_fd, rtol = 1.e-5)

    x = np.array([[1., 2.], [2., 3.]])
    y = np.array([[2., 4.], [3., 1.]])
    params = np.array([np.log(2.), np.log(4.)])

    deriv_fd = np.zeros((2, 2, 2))
    deriv_fd[0] = (k.kernel_f(x, y, params) -
                   k.kernel_f(x, y, params - np.array([dx, 0.])))/dx
    deriv_fd[1] = (k.kernel_f(x, y, params) -
                   k.kernel_f(x, y, params - np.array([0., dx])))/dx

    assert_allclose(k.kernel_deriv(x, y, params), deriv_fd, rtol = 1.e-5)

    x = np.array([1., 2.])
    y = np.array([2., 3.])
    params = np.array([0.])

    deriv_fd = np.zeros((1, 2, 2))
    deriv_fd[0] = (k.kernel_f(x, y, params) -
                   k.kernel_f(x, y, params - np.array([dx])))/dx

    assert_allclose(k.kernel_deriv(x, y, params), deriv_fd, rtol = 1.e-5)

def test_squared_exponential_deriv_failures():
    "test scenarios where squared_exponential should raise an exception"

    k = SquaredExponential()

    x = np.array([[1.], [2.]])
    y = np.array([[2.], [3.]])
    params = np.array([0., 0.])

    with pytest.raises(AssertionError):
        k.kernel_deriv(x, y, params)

    params = np.array([[0., 0.], [0., 0.]])

    with pytest.raises(AssertionError):
        k.kernel_deriv(x, y, params)

    x = np.array([[1.], [2.]])
    y = np.array([[2., 4.], [3., 2.]])
    params = np.array([0.])

    with pytest.raises(AssertionError):
        k.kernel_deriv(x, y, params)

    x = np.array([[1.], [2.]])
    y = np.array([[[2.], [4.]], [[3.], [2.]]])
    params = np.array([0.])

    with pytest.raises(AssertionError):
        k.kernel_deriv(x, y, params)

    x = np.array([[2., 4.], [3., 2.]])
    y = np.array([[1.], [2.]])
    params = np.array([0.])

    with pytest.raises(AssertionError):
        k.kernel_deriv(x, y, params)

    x = np.array([[[2.], [4.]], [[3.], [2.]]])
    y = np.array([[1.], [2.]])
    params = np.array([0.])

    with pytest.raises(AssertionError):
        k.kernel_deriv(x, y, params)

def test_squared_exponential_hessian():
    "test the function to compute the squared exponential hessian"

    k = SquaredExponential()

    dx = 1.e-6

    x = np.array([[1.], [2.]])
    y = np.array([[2.], [3.]])
    params = np.array([0.])

    hess_fd = np.zeros((1, 1, 2, 2))
    hess_fd[0, 0] = (k.kernel_deriv(x, y, params)[0] -
                     k.kernel_deriv(x, y, params-np.array([dx]))[0])/dx

    assert_allclose(k.kernel_hessian(x, y, params), hess_fd, atol = 1.e-5)

    x = np.array([[1., 2.], [2., 3.]])
    y = np.array([[2., 4.], [3., 1.]])
    params = np.array([0., 0.])

    hess = np.zeros((2, 2, 2, 2))

    hess_fd = np.zeros((2, 2, 2, 2))
    hess_fd[0, 0] = (k.kernel_deriv(x, y, params)[0] -
                     k.kernel_deriv(x, y, params-np.array([dx, 0.]))[0])/dx
    hess_fd[1, 0] = (k.kernel_deriv(x, y, params)[1] -
                     k.kernel_deriv(x, y, params-np.array([dx, 0.]))[1])/dx
    hess_fd[0, 1] = (k.kernel_deriv(x, y, params)[0] -
                     k.kernel_deriv(x, y, params-np.array([0., dx]))[0])/dx
    hess_fd[1, 1] = (k.kernel_deriv(x, y, params)[1] -
                     k.kernel_deriv(x, y, params-np.array([0., dx]))[1])/dx

    assert_allclose(k.kernel_hessian(x, y, params), hess_fd, atol = 1.e-5)

    x = np.array([[1., 2.], [2., 3.]])
    y = np.array([[2., 4.], [3., 1.]])
    params = np.array([np.log(2.), np.log(4.)])

    hess_fd = np.zeros((2, 2, 2, 2))
    hess_fd[0, 0] = (k.kernel_deriv(x, y, params)[0] -
                     k.kernel_deriv(x, y, params-np.array([dx, 0.]))[0])/dx
    hess_fd[1, 0] = (k.kernel_deriv(x, y, params)[1] -
                     k.kernel_deriv(x, y, params-np.array([dx, 0.]))[1])/dx
    hess_fd[0, 1] = (k.kernel_deriv(x, y, params)[0] -
                     k.kernel_deriv(x, y, params-np.array([0., dx]))[0])/dx
    hess_fd[1, 1] = (k.kernel_deriv(x, y, params)[1] -
                     k.kernel_deriv(x, y, params-np.array([0., dx]))[1])/dx

    assert_allclose(k.kernel_hessian(x, y, params), hess_fd, atol = 1.e-5)

    x = np.array([1., 2.])
    y = np.array([2., 3.])
    params = np.array([0.])

    hess_fd = np.zeros((1, 1, 2, 2))
    hess_fd[0, 0] = (k.kernel_deriv(x, y, params)[0] -
                     k.kernel_deriv(x, y, params-np.array([dx]))[0])/dx

    assert_allclose(k.kernel_hessian(x, y, params), hess_fd, atol = 1.e-5)

def test_squared_exponential_hessian_failures():
    "test situaitons where squared_exponential_hessian should fail"

    k = SquaredExponential()

    x = np.array([[1.], [2.]])
    y = np.array([[2.], [3.]])
    params = np.array([0., 0.])

    with pytest.raises(AssertionError):
        k.kernel_hessian(x, y, params)

    params = np.array([[0., 0.], [0., 0.]])

    with pytest.raises(AssertionError):
        k.kernel_hessian(x, y, params)

    x = np.array([[1.], [2.]])
    y = np.array([[2., 4.], [3., 2.]])
    params = np.array([0.])

    with pytest.raises(AssertionError):
        k.kernel_hessian(x, y, params)

    x = np.array([[1.], [2.]])
    y = np.array([[[2.], [4.]], [[3.], [2.]]])
    params = np.array([0.])

    with pytest.raises(AssertionError):
        k.kernel_hessian(x, y, params)

    x = np.array([[2., 4.], [3., 2.]])
    y = np.array([[1.], [2.]])
    params = np.array([0.])

    with pytest.raises(AssertionError):
        k.kernel_hessian(x, y, params)

    x = np.array([[[2.], [4.]], [[3.], [2.]]])
    y = np.array([[1.], [2.]])
    params = np.array([0.])

    with pytest.raises(AssertionError):
        k.kernel_hessian(x, y, params)

def test_matern_5_2_K():
    "test matern 5/2 K(r) function"

    k = Matern52()

    assert_allclose(k.calc_K(1.), (1.+np.sqrt(5.)+5./3.)*np.exp(-np.sqrt(5.)))

    r2 = np.array([[1., 2.], [3., 4.]])

    assert_allclose(k.calc_K(r2), (1.+np.sqrt(5.*r2)+5./3.*r2)*np.exp(-np.sqrt(5.*r2)))

    with pytest.raises(AssertionError):
        k.calc_K(-1.)

def test_matern_5_2_dKdr():
    "test matern 5/2 dK/dr function"

    k = Matern52()

    dx = 1.e-6

    assert_allclose(k.calc_dKdr2(1.), (k.calc_K(1.)-k.calc_K(1.-dx))/dx, rtol = 1.e-5)

    r2 = np.array([[1., 2.], [3., 4.]])

    assert_allclose(k.calc_dKdr2(r2), (k.calc_K(r2)-k.calc_K(r2 - dx))/dx, rtol = 1.e-5)

    with pytest.raises(AssertionError):
        k.calc_dKdr2(-1.)

def test_matern_5_2_d2Kdr22():
    "test squared exponential d2K/dr22 function"

    k = Matern52()

    dx = 1.e-6

    assert_allclose(k.calc_d2Kdr22(1.), (k.calc_dKdr2(1.)-k.calc_dKdr2(1.-dx))/dx, rtol = 1.e-5)

    r2 = np.array([[1., 2.], [3., 4.]])

    assert_allclose(k.calc_d2Kdr22(r2), (k.calc_dKdr2(r2)-k.calc_dKdr2(r2 - dx))/dx, rtol = 1.e-5)

    with pytest.raises(AssertionError):
        k.calc_d2Kdr22(-1.)

def test_matern_5_2():
    "test matern 5/2 covariance kernel"

    k = Matern52()

    x = np.array([[1.], [2.]])
    y = np.array([[2.], [3.]])
    params = np.array([0.])

    D = np.array([[1., 2.], [0., 1.]])

    assert_allclose(k.kernel_f(x, y, params),
                    (1.+np.sqrt(5.)*D+5./3.*D**2)*np.exp(-np.sqrt(5.)*D))

    x = np.array([[1., 2.], [2., 3.]])
    y = np.array([[2., 4.], [3., 1.]])
    params = np.array([0., 0.])

    D = np.array([[np.sqrt(5.), np.sqrt(5.)], [1., np.sqrt(5.)]])

    assert_allclose(k.kernel_f(x, y, params),
                    (1.+np.sqrt(5.)*D+5./3.*D**2)*np.exp(-np.sqrt(5.)*D))

    x = np.array([[1., 2.], [2., 3.]])
    y = np.array([[2., 4.], [3., 1.]])
    params = np.array([np.log(2.), np.log(4.)])

    D = np.array([[np.sqrt(1.*2.+4.*4.), np.sqrt(4.*2.+1.*4.)],
                  [np.sqrt(1.*4.), np.sqrt(1.*2.+4.*4.)]])

    assert_allclose(k.kernel_f(x, y, params),
                    (1.+np.sqrt(5.)*D+5./3.*D**2)*np.exp(-np.sqrt(5.)*D))

    x = np.array([1., 2.])
    y = np.array([2., 3.])
    params = np.array([0.])

    D = np.array([[1., 2.], [0., 1.]])

    assert_allclose(k.kernel_f(x, y, params),
                    (1.+np.sqrt(5.)*D + 5./3.*D**2)*np.exp(-np.sqrt(5.)*D))

def test_matern_5_2_failures():
    "test scenarios where matern_5_2 should raise an exception"

    k = Matern52()

    x = np.array([[1.], [2.]])
    y = np.array([[2.], [3.]])
    params = np.array([0., 0.])

    with pytest.raises(AssertionError):
        k.kernel_f(x, y, params)

    params = np.array([[0., 0.], [0., 0.]])

    with pytest.raises(AssertionError):
        k.kernel_f(x, y, params)

def test_matern_5_2_deriv():
    "test computing the gradient of the matern 5/2 kernel"

    k = Matern52()

    dx = 1.e-6

    x = np.array([[1.], [2.]])
    y = np.array([[2.], [3.]])
    params = np.array([0.])

    deriv_fd = np.zeros((1, 2, 2))
    deriv_fd[0] = (k.kernel_f(x, y, params) -
                   k.kernel_f(x, y, params - np.array([dx])))/dx

    assert_allclose(k.kernel_deriv(x, y, params), deriv_fd, rtol = 1.e-5)

    x = np.array([[1., 2.], [2., 3.]])
    y = np.array([[2., 4.], [3., 1.]])
    params = np.array([0., 0.])

    deriv_fd = np.zeros((2, 2, 2))
    deriv_fd[0] = (k.kernel_f(x, y, params) -
                   k.kernel_f(x, y, params - np.array([dx, 0.])))/dx
    deriv_fd[1] = (k.kernel_f(x, y, params) -
                   k.kernel_f(x, y, params - np.array([0., dx])))/dx

    assert_allclose(k.kernel_deriv(x, y, params), deriv_fd, rtol = 1.e-5)

    x = np.array([[1., 2.], [2., 3.]])
    y = np.array([[2., 4.], [3., 1.]])
    params = np.array([np.log(2.), np.log(4.)])

    deriv_fd = np.zeros((2, 2, 2))
    deriv_fd[0] = (k.kernel_f(x, y, params) -
                   k.kernel_f(x, y, params - np.array([dx, 0.])))/dx
    deriv_fd[1] = (k.kernel_f(x, y, params) -
                   k.kernel_f(x, y, params - np.array([0., dx])))/dx

    assert_allclose(k.kernel_deriv(x, y, params), deriv_fd, rtol = 1.e-5)

    x = np.array([1., 2.])
    y = np.array([2., 3.])
    params = np.array([0.])

    deriv_fd = np.zeros((1, 2, 2))
    deriv_fd[0] = (k.kernel_f(x, y, params) -
                   k.kernel_f(x, y, params - np.array([dx])))/dx

    assert_allclose(k.kernel_deriv(x, y, params), deriv_fd, rtol = 1.e-5)

def test_matern_5_2_deriv_failures():
    "test scenarios where matern_5_2_deriv should raise an exception"

    k = Matern52()

    x = np.array([[1.], [2.]])
    y = np.array([[2.], [3.]])
    params = np.array([0., 0.])

    with pytest.raises(AssertionError):
        k.kernel_deriv(x, y, params)

    params = np.array([[0., 0.], [0., 0.]])

    with pytest.raises(AssertionError):
        k.kernel_deriv(x, y, params)

    x = np.array([[1.], [2.]])
    y = np.array([[2., 4.], [3., 2.]])
    params = np.array([0.])

    with pytest.raises(AssertionError):
        k.kernel_deriv(x, y, params)

    x = np.array([[1.], [2.]])
    y = np.array([[[2.], [4.]], [[3.], [2.]]])
    params = np.array([0.])

    with pytest.raises(AssertionError):
        k.kernel_deriv(x, y, params)

    x = np.array([[2., 4.], [3., 2.]])
    y = np.array([[1.], [2.]])
    params = np.array([0.])

    with pytest.raises(AssertionError):
        k.kernel_deriv(x, y, params)

    x = np.array([[[2.], [4.]], [[3.], [2.]]])
    y = np.array([[1.], [2.]])
    params = np.array([0.])

    with pytest.raises(AssertionError):
        k.kernel_deriv(x, y, params)

def test_matern_5_2_hessian():
    "test the function to compute the squared exponential hessian"

    k = Matern52()

    dx = 1.e-6

    x = np.array([[1.], [2.]])
    y = np.array([[2.], [3.]])
    params = np.array([0.])

    hess_fd = np.zeros((1, 1, 2, 2))
    hess_fd[0, 0] = (k.kernel_deriv(x, y, params)[0] -
                     k.kernel_deriv(x, y, params-np.array([dx]))[0])/dx

    assert_allclose(k.kernel_hessian(x, y, params), hess_fd, rtol = 1.e-5)

    x = np.array([[1., 2.], [2., 3.]])
    y = np.array([[2., 4.], [3., 1.]])
    params = np.array([0., 0.])

    hess_fd = np.zeros((2, 2, 2, 2))
    hess_fd[0, 0] = (k.kernel_deriv(x, y, params)[0] -
                     k.kernel_deriv(x, y, params-np.array([dx, 0.]))[0])/dx
    hess_fd[0, 1] = (k.kernel_deriv(x, y, params)[0] -
                     k.kernel_deriv(x, y, params-np.array([0., dx]))[0])/dx
    hess_fd[1, 0] = (k.kernel_deriv(x, y, params)[1] -
                     k.kernel_deriv(x, y, params-np.array([dx, 0.]))[1])/dx
    hess_fd[1, 1] = (k.kernel_deriv(x, y, params)[1] -
                     k.kernel_deriv(x, y, params-np.array([0., dx]))[1])/dx

    assert_allclose(k.kernel_hessian(x, y, params), hess_fd, rtol = 1.e-5)

    x = np.array([[1., 2.], [2., 3.]])
    y = np.array([[2., 4.], [3., 1.]])
    params = np.array([np.log(2.), np.log(4.)])

    hess_fd = np.zeros((2, 2, 2, 2))
    hess_fd[0, 0] = (k.kernel_deriv(x, y, params)[0] -
                     k.kernel_deriv(x, y, params-np.array([dx, 0.]))[0])/dx
    hess_fd[0, 1] = (k.kernel_deriv(x, y, params)[0] -
                     k.kernel_deriv(x, y, params-np.array([0., dx]))[0])/dx
    hess_fd[1, 0] = (k.kernel_deriv(x, y, params)[1] -
                     k.kernel_deriv(x, y, params-np.array([dx, 0.]))[1])/dx
    hess_fd[1, 1] = (k.kernel_deriv(x, y, params)[1] -
                     k.kernel_deriv(x, y, params-np.array([0., dx]))[1])/dx

    assert_allclose(k.kernel_hessian(x, y, params), hess_fd, rtol = 1.e-5)

    x = np.array([1., 2.])
    y = np.array([2., 3.])
    params = np.array([0.])

    hess_fd = np.zeros((1, 1, 2, 2))
    hess_fd[0, 0] = (k.kernel_deriv(x, y, params)[0] -
                     k.kernel_deriv(x, y, params-np.array([dx]))[0])/dx

    assert_allclose(k.kernel_hessian(x, y, params), hess_fd, rtol = 1.e-5)

def test_matern_5_2_hessian_failures():
    "test situaitons where squared_exponential_hessian should fail"

    k = Matern52()

    x = np.array([[1.], [2.]])
    y = np.array([[2.], [3.]])
    params = np.array([0., 0.])

    with pytest.raises(AssertionError):
        k.kernel_hessian(x, y, params)

    params = np.array([[0., 0.], [0., 0.]])

    with pytest.raises(AssertionError):
        k.kernel_hessian(x, y, params)

    x = np.array([[1.], [2.]])
    y = np.array([[2., 4.], [3., 2.]])
    params = np.array([0.])

    with pytest.raises(AssertionError):
        k.kernel_hessian(x, y, params)

    x = np.array([[1.], [2.]])
    y = np.array([[[2.], [4.]], [[3.], [2.]]])
    params = np.array([0.])

    with pytest.raises(AssertionError):
        k.kernel_hessian(x, y, params)

    x = np.array([[2., 4.], [3., 2.]])
    y = np.array([[1.], [2.]])
    params = np.array([0.])

    with pytest.raises(AssertionError):
        k.kernel_hessian(x, y, params)

    x = np.array([[[2.], [4.]], [[3.], [2.]]])
    y = np.array([[1.], [2.]])
    params = np.array([0.])

    with pytest.raises(AssertionError):
        k.kernel_hessian(x, y, params)

def test_Kernel_str():
    "test string method of generic Kernel class"

    k = Kernel()

    assert str(k) == "Stationary Kernel"

def test_SquaredExponential_str():
    "test string method of SquaredExponential class"

    k = SquaredExponential()

    assert str(k) == "Squared Exponential Kernel"

def test_Matern52_str():
    "test string method of Matern52 class"

    k = Matern52()

    assert str(k) == "Matern 5/2 Kernel"