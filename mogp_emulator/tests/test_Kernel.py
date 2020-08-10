import numpy as np
import pytest
from numpy.testing import assert_allclose
from ..Kernel import Kernel, SquaredExponential, Matern52

def test_calc_r():
    "test function for calc_r function for kernels"

    k = Kernel()

    x = np.array([[1.], [2.]])
    y = np.array([[2.], [3.]])
    params = np.array([0., 0.])

    assert_allclose(k.calc_r(x, y, params), np.array([[1., 2.], [0., 1.]]))

    x = np.array([[1., 2.], [2., 3.]])
    y = np.array([[2., 4.], [3., 1.]])
    params = np.array([0., 0., 0.])

    assert_allclose(k.calc_r(x, y, params),
                    np.array([[np.sqrt(5.), np.sqrt(5.)], [1., np.sqrt(5.)]]))

    x = np.array([[1., 2.], [2., 3.]])
    y = np.array([[2., 4.], [3., 1.]])
    params = np.array([np.log(2.), np.log(4.), 0.])

    assert_allclose(k.calc_r(x, y, params),
                    np.array([[np.sqrt(1.*2.+4.*4.), np.sqrt(4.*2.+1.*4.)],
                              [np.sqrt(1.*4.), np.sqrt(1.*2.+4.*4.)]]))

    x = np.array([1., 2.])
    y = np.array([2., 3.])
    params = np.array([0., 0.])

    assert_allclose(k.calc_r(x, y, params), np.array([[1., 2.], [0., 1.]]))


def test_calc_r_failures():
    "test scenarios where calc_r should raise an exception"

    k = Kernel()

    x = np.array([[1.], [2.]])
    y = np.array([[2.], [3.]])
    params = np.array([0.])

    with pytest.raises(AssertionError):
        k.calc_r(x, y, params)

    params = np.array([[0., 0.], [0., 0.]])

    with pytest.raises(AssertionError):
        k.calc_r(x, y, params)

    x = np.array([[1.], [2.]])
    y = np.array([[2., 4.], [3., 2.]])
    params = np.array([0., 0.])

    with pytest.raises(AssertionError):
        k.calc_r(x, y, params)

    x = np.array([[1.], [2.]])
    y = np.array([[[2.], [4.]], [[3.], [2.]]])
    params = np.array([0., 0.])

    with pytest.raises(AssertionError):
        k.calc_r(x, y, params)

    x = np.array([[2., 4.], [3., 2.]])
    y = np.array([[1.], [2.]])
    params = np.array([0., 0.])

    with pytest.raises(AssertionError):
        k.calc_r(x, y, params)

    x = np.array([[[2.], [4.]], [[3.], [2.]]])
    y = np.array([[1.], [2.]])
    params = np.array([0., 0.])

    with pytest.raises(AssertionError):
        k.calc_r(x, y, params)

    with pytest.raises(FloatingPointError):
        k.calc_r(y, y, np.array([800., 0.]))

def test_calc_drdtheta():
    "test calc_drdtheta function"

    k = Kernel()

    dx = 1.e-6

    x = np.array([[1.], [2.]])
    y = np.array([[2.], [3.]])
    params = np.array([0., 0.])

    r = np.array([[1., 2.], [1., 1.]])

    deriv = np.zeros((1, 2, 2))
    deriv[0] = 0.5*np.array([[1., 4.], [0., 1.]])/r
    deriv_fd = np.zeros((1, 2, 2))
    deriv_fd[0] = (k.calc_r(x, y, params) -
                   k.calc_r(x, y, params - np.array([dx, 0.])))/dx

    assert_allclose(k.calc_drdtheta(x, y, params), deriv)
    assert_allclose(k.calc_drdtheta(x, y, params), deriv_fd, rtol = 1.e-5)

    x = np.array([[1., 2.], [2., 3.]])
    y = np.array([[2., 4.], [3., 1.]])
    params = np.array([0., 0., 0.])

    r = np.array([[np.sqrt(5.), np.sqrt(5.)], [1., np.sqrt(5.)]])

    deriv = np.zeros((2, 2, 2))
    deriv[0] = 0.5*np.array([[1., 4.], [0., 1.]])/r
    deriv[1] = 0.5*np.array([[4., 1.], [1., 4.]])/r
    deriv_fd = np.zeros((2, 2, 2))
    deriv_fd[0] = (k.calc_r(x, y, params) -
                   k.calc_r(x, y, params - np.array([dx, 0., 0.])))/dx
    deriv_fd[1] = (k.calc_r(x, y, params) -
                   k.calc_r(x, y, params - np.array([0., dx, 0.])))/dx

    assert_allclose(k.calc_drdtheta(x, y, params), deriv)
    assert_allclose(k.calc_drdtheta(x, y, params), deriv_fd, rtol = 1.e-5)

    deriv = np.zeros((2, 2, 2))
    x = np.array([[1., 2.], [2., 3.]])
    y = np.array([[2., 4.], [3., 1.]])
    params = np.array([np.log(2.), np.log(4.), 0.])

    r = np.array([[np.sqrt(1.*2.+4.*4.), np.sqrt(4.*2.+1.*4.)],
                  [np.sqrt(1.*4.), np.sqrt(1.*2.+4.*4.)]])

    deriv = np.zeros((2, 2, 2))
    deriv[0] = 0.5*2.*np.array([[1., 4.], [0., 1.]])/r
    deriv[1] = 0.5*4.*np.array([[4., 1.], [1., 4.]])/r
    deriv_fd = np.zeros((2, 2, 2))
    deriv_fd[0] = (k.calc_r(x, y, params) -
                   k.calc_r(x, y, params - np.array([dx, 0., 0.])))/dx
    deriv_fd[1] = (k.calc_r(x, y, params) -
                   k.calc_r(x, y, params - np.array([0., dx, 0.])))/dx

    assert_allclose(k.calc_drdtheta(x, y, params), deriv)
    assert_allclose(k.calc_drdtheta(x, y, params), deriv_fd, rtol = 1.e-5)

    x = np.array([1., 2.])
    y = np.array([2., 3.])
    params = np.array([0., 0.])

    r = np.array([[1., 2.], [1., 1.]])

    deriv = np.zeros((1, 2, 2))
    deriv[0] = 0.5*np.array([[1., 4.], [0., 1.]])/r
    deriv_fd = np.zeros((1, 2, 2))
    deriv_fd[0] = (k.calc_r(x, y, params) -
                   k.calc_r(x, y, params - np.array([dx, 0.])))/dx

    assert_allclose(k.calc_drdtheta(x, y, params), deriv)
    assert_allclose(k.calc_drdtheta(x, y, params), deriv_fd, rtol = 1.e-5)

def test_calc_drdtheta_failures():
    "test situations where calc_drdtheta should fail"

    k = Kernel()

    x = np.array([[1.], [2.]])
    y = np.array([[2.], [3.]])
    params = np.array([0.])

    with pytest.raises(AssertionError):
        k.calc_drdtheta(x, y, params)

    params = np.array([[0., 0.], [0., 0.]])

    with pytest.raises(AssertionError):
        k.calc_drdtheta(x, y, params)

    x = np.array([[1.], [2.]])
    y = np.array([[2., 4.], [3., 2.]])
    params = np.array([0., 0.])

    with pytest.raises(AssertionError):
        k.calc_drdtheta(x, y, params)

    x = np.array([[1.], [2.]])
    y = np.array([[[2.], [4.]], [[3.], [2.]]])
    params = np.array([0., 0.])

    with pytest.raises(AssertionError):
        k.calc_drdtheta(x, y, params)

    x = np.array([[2., 4.], [3., 2.]])
    y = np.array([[1.], [2.]])
    params = np.array([0., 0.])

    with pytest.raises(AssertionError):
        k.calc_drdtheta(x, y, params)

    x = np.array([[[2.], [4.]], [[3.], [2.]]])
    y = np.array([[1.], [2.]])
    params = np.array([0., 0.])

    with pytest.raises(AssertionError):
        k.calc_drdtheta(x, y, params)

def test_calc_d2rdtheta2():
    "test calc_d2rdtheta2 function"

    k = Kernel()

    dx = 1.e-6

    x = np.array([[1.], [2.]])
    y = np.array([[2.], [3.]])
    params = np.array([0., 0.])

    r = np.array([[1., 2.], [1., 1.]])

    deriv = np.zeros((1, 1, 2, 2))
    deriv[0, 0] = (0.5*np.array([[1., 4.], [0., 1.]])/r -
                   0.25*np.array([[1., 4.], [0., 1.]])**2/r**3)
    deriv_fd = np.zeros((1, 1, 2, 2))
    deriv_fd[0, 0] = (k.calc_drdtheta(x, y, params)[0] -
                      k.calc_drdtheta(x, y, params - np.array([dx, 0.]))[0])/dx

    assert_allclose(k.calc_d2rdtheta2(x, y, params), deriv)
    assert_allclose(k.calc_d2rdtheta2(x, y, params), deriv_fd, rtol = 1.e-5)

    x = np.array([[1., 2.], [2., 3.]])
    y = np.array([[2., 4.], [3., 1.]])
    params = np.array([0., 0., 0.])

    r = np.array([[np.sqrt(5.), np.sqrt(5.)], [1., np.sqrt(5.)]])

    deriv = np.zeros((2, 2, 2, 2))
    x12 = np.array([[1., 4.], [0., 1.]])
    x22 = np.array([[4., 1.], [1., 4.]])
    deriv[0, 0] = 0.5*x12/r-0.25*x12*x12/r**3
    deriv[0, 1] = -0.25*x12*x22/r**3
    deriv[1, 0] = -0.25*x22*x12/r**3
    deriv[1, 1] = 0.5*x22/r-0.25*x22*x22/r**3
    deriv_fd = np.zeros((2, 2, 2, 2))
    deriv_fd[0, 0] = (k.calc_drdtheta(x, y, params)[0] -
                      k.calc_drdtheta(x, y, params - np.array([dx, 0., 0.]))[0])/dx
    deriv_fd[0, 1] = (k.calc_drdtheta(x, y, params)[1] -
                      k.calc_drdtheta(x, y, params - np.array([dx, 0., 0.]))[1])/dx
    deriv_fd[1, 0] = (k.calc_drdtheta(x, y, params)[0] -
                      k.calc_drdtheta(x, y, params - np.array([0., dx, 0.]))[0])/dx
    deriv_fd[1, 1] = (k.calc_drdtheta(x, y, params)[1] -
                      k.calc_drdtheta(x, y, params - np.array([0., dx, 0.]))[1])/dx

    assert_allclose(k.calc_d2rdtheta2(x, y, params), deriv)
    assert_allclose(k.calc_d2rdtheta2(x, y, params), deriv_fd, rtol = 1.e-5)

    x = np.array([[1., 2.], [2., 3.]])
    y = np.array([[2., 4.], [3., 1.]])
    params = np.array([np.log(2.), np.log(4.), 0.])

    r = np.array([[np.sqrt(1.*2.+4.*4.), np.sqrt(4.*2.+1.*4.)],
                  [np.sqrt(1.*4.), np.sqrt(1.*2.+4.*4.)]])

    deriv = np.zeros((2, 2, 2, 2))
    x12 = np.array([[1., 4.], [0., 1.]])
    x22 = np.array([[4., 1.], [1., 4.]])
    deriv[0, 0] = 2.*0.5*x12/r-2.*2.*0.25*x12**2/r**3
    deriv[0, 1] = -2.*4.*0.25*x12*x22/r**3
    deriv[1, 0] = -4.*2.*0.25*x22*x12/r**3
    deriv[1, 1] = 4.*0.5*x22/r-4.*4.*0.25*x22*x22/r**3
    deriv_fd = np.zeros((2, 2, 2, 2))
    deriv_fd[0, 0] = (k.calc_drdtheta(x, y, params)[0] -
                      k.calc_drdtheta(x, y, params - np.array([dx, 0., 0.]))[0])/dx
    deriv_fd[0, 1] = (k.calc_drdtheta(x, y, params)[1] -
                      k.calc_drdtheta(x, y, params - np.array([dx, 0., 0.]))[1])/dx
    deriv_fd[1, 0] = (k.calc_drdtheta(x, y, params)[0] -
                      k.calc_drdtheta(x, y, params - np.array([0., dx, 0.]))[0])/dx
    deriv_fd[1, 1] = (k.calc_drdtheta(x, y, params)[1] -
                      k.calc_drdtheta(x, y, params - np.array([0., dx, 0.]))[1])/dx

    assert_allclose(k.calc_d2rdtheta2(x, y, params), deriv)
    assert_allclose(k.calc_d2rdtheta2(x, y, params), deriv_fd, rtol = 1.e-5)

    x = np.array([1., 2.])
    y = np.array([2., 3.])
    params = np.array([0., 0.])

    r = np.array([[1., 2.], [1., 1.]])

    deriv = np.zeros((1, 1, 2, 2))
    deriv[0, 0] = (0.5*np.array([[1., 4.], [0., 1.]])/r -
                   0.25*np.array([[1., 4.], [0., 1.]])**2/r**3)
    deriv_fd = np.zeros((1, 1, 2, 2))
    deriv_fd[0, 0] = (k.calc_drdtheta(x, y, params)[0] -
                      k.calc_drdtheta(x, y, params - np.array([dx, 0.]))[0])/dx

    assert_allclose(k.calc_d2rdtheta2(x, y, params), deriv)
    assert_allclose(k.calc_d2rdtheta2(x, y, params), deriv_fd, rtol = 1.e-5)

def test_calc_d2rdtheta2_failures():
    "test situations where calc_d2rdtheta2 should fail"

    k = Kernel()

    x = np.array([[1.], [2.]])
    y = np.array([[2.], [3.]])
    params = np.array([0.])

    with pytest.raises(AssertionError):
        k.calc_d2rdtheta2(x, y, params)

    params = np.array([[0., 0.], [0., 0.]])

    with pytest.raises(AssertionError):
        k.calc_d2rdtheta2(x, y, params)

    x = np.array([[1.], [2.]])
    y = np.array([[2., 4.], [3., 2.]])
    params = np.array([0., 0.])

    with pytest.raises(AssertionError):
        k.calc_d2rdtheta2(x, y, params)

    x = np.array([[1.], [2.]])
    y = np.array([[[2.], [4.]], [[3.], [2.]]])
    params = np.array([0., 0.])

    with pytest.raises(AssertionError):
        k.calc_d2rdtheta2(x, y, params)

    x = np.array([[2., 4.], [3., 2.]])
    y = np.array([[1.], [2.]])
    params = np.array([0., 0.])

    with pytest.raises(AssertionError):
        k.calc_d2rdtheta2(x, y, params)

    x = np.array([[[2.], [4.]], [[3.], [2.]]])
    y = np.array([[1.], [2.]])
    params = np.array([0., 0.])

    with pytest.raises(AssertionError):
        k.calc_d2rdtheta2(x, y, params)

def test_kernel_calc_drdx():
    "test the calc_drdx method of the kernel class"

    k = Kernel()

    dx = 1.e-6

    x = np.array([[1.], [2.]])
    y = np.array([[2.], [3.]])
    params = np.array([0., 0.])

    r = np.array([[1., 2.], [1., 1.]])

    deriv = np.zeros((1, 2, 2))
    deriv[0] = -np.array([[1., 2.], [0., 1.]])/r
    deriv_fd = np.zeros((1, 2, 2))
    # need to use central differences here as derivative is discontiuous at zero
    deriv_fd[0] = (k.calc_r(x + dx, y, params) - k.calc_r(x - dx, y, params))/dx/2.

    assert_allclose(k.calc_drdx(x, y, params), deriv)
    assert_allclose(k.calc_drdx(x, y, params), deriv_fd, rtol = 1.e-5, atol = 1.e-8)

    x = np.array([[1., 2.], [2., 3.]])
    y = np.array([[2., 4.], [3., 1.]])
    params = np.array([0., 0., 0.])

    r = np.array([[np.sqrt(5.), np.sqrt(5.)], [1., np.sqrt(5.)]])

    deriv = np.zeros((2, 2, 2))
    deriv[0] = -np.array([[1., 2.], [0., 1.]])/r
    deriv[1] = np.array([[-2., 1.], [-1., 2.]])/r
    deriv_fd = np.zeros((2, 2, 2))
    deriv_fd[0] = (k.calc_r(x + np.array([[dx, 0.], [dx, 0.]]), y, params) -
                   k.calc_r(x - np.array([[dx, 0.], [dx, 0.]]), y, params))/dx/2.
    deriv_fd[1] = (k.calc_r(x + np.array([[0., dx], [0., dx]]), y, params) -
                   k.calc_r(x - np.array([[0., dx], [0., dx]]), y, params))/dx/2.

    assert_allclose(k.calc_drdx(x, y, params), deriv)
    assert_allclose(k.calc_drdx(x, y, params), deriv_fd, rtol = 1.e-5, atol = 1.e-7)

    deriv = np.zeros((2, 2, 2))
    x = np.array([[1., 2.], [2., 3.]])
    y = np.array([[2., 4.], [3., 1.]])
    params = np.array([np.log(2.), np.log(4.), 0.])

    r = np.array([[np.sqrt(1.*2.+4.*4.), np.sqrt(4.*2.+1.*4.)],
                  [np.sqrt(1.*4.), np.sqrt(1.*2.+4.*4.)]])

    deriv = np.zeros((2, 2, 2))
    deriv[0] = 2.*np.array([[-1., -2.], [0., -1.]])/r
    deriv[1] = 4.*np.array([[-2., 1.], [-1., 2.]])/r
    deriv_fd = np.zeros((2, 2, 2))
    deriv_fd[0] = (k.calc_r(x + np.array([[dx, 0.], [dx, 0.]]), y, params) -
                   k.calc_r(x - np.array([[dx, 0.], [dx, 0.]]), y, params))/dx/2.
    deriv_fd[1] = (k.calc_r(x + np.array([[0., dx], [0., dx]]), y, params) -
                   k.calc_r(x - np.array([[0., dx], [0., dx]]), y, params))/dx/2.

    assert_allclose(k.calc_drdx(x, y, params), deriv)
    assert_allclose(k.calc_drdx(x, y, params), deriv_fd, rtol = 1.e-5, atol = 1.e-7)

    x = np.array([1., 2.])
    y = np.array([2., 3.])
    params = np.array([0., 0.])

    r = np.array([[1., 2.], [1., 1.]])

    deriv = np.zeros((1, 2, 2))
    deriv[0] = -np.array([[1., 2.], [0., 1.]])/r
    deriv_fd = np.zeros((1, 2, 2))
    # need to use central differences here as derivative is discontiuous at zero
    deriv_fd[0] = (k.calc_r(x + dx, y, params) - k.calc_r(x - dx, y, params))/dx/2.

    assert_allclose(k.calc_drdx(x, y, params), deriv)
    assert_allclose(k.calc_drdx(x, y, params), deriv_fd, rtol = 1.e-5, atol = 1.e-8)

def test_kernel_calc_drdx_failures():
    "test situations where calc_drdx should fail"

    k = Kernel()

    x = np.array([[1.], [2.]])
    y = np.array([[2.], [3.]])
    params = np.array([0.])

    with pytest.raises(AssertionError):
        k.calc_drdx(x, y, params)

    params = np.array([[0., 0.], [0., 0.]])

    with pytest.raises(AssertionError):
        k.calc_drdx(x, y, params)

    x = np.array([[1.], [2.]])
    y = np.array([[2., 4.], [3., 2.]])
    params = np.array([0., 0.])

    with pytest.raises(AssertionError):
        k.calc_drdx(x, y, params)

    x = np.array([[1.], [2.]])
    y = np.array([[[2.], [4.]], [[3.], [2.]]])
    params = np.array([0., 0.])

    with pytest.raises(AssertionError):
        k.calc_drdx(x, y, params)

    x = np.array([[2., 4.], [3., 2.]])
    y = np.array([[1.], [2.]])
    params = np.array([0., 0.])

    with pytest.raises(AssertionError):
        k.calc_drdx(x, y, params)

    x = np.array([[[2.], [4.]], [[3.], [2.]]])
    y = np.array([[1.], [2.]])
    params = np.array([0., 0.])

    with pytest.raises(AssertionError):
        k.calc_drdx(x, y, params)

def test_squared_exponential_K():
    "test squared exponential K(r) function"

    k = SquaredExponential()

    assert_allclose(k.calc_K(1.), np.exp(-0.5))

    assert_allclose(k.calc_K(np.array([[1., 2.], [3., 4.]])),
                             np.exp(-0.5*np.array([[1., 4.], [9., 16.]])))

    with pytest.raises(AssertionError):
        k.calc_K(-1.)

def test_squared_exponential_dKdr():
    "test squared exponential dK/dr function"

    k = SquaredExponential()

    dx = 1.e-6

    assert_allclose(k.calc_dKdr(1.), -np.exp(-0.5))
    assert_allclose(k.calc_dKdr(1.), (k.calc_K(1.)-k.calc_K(1.-dx))/dx, rtol = 1.e-5)

    r = np.array([[1., 2.], [3., 4.]])

    assert_allclose(k.calc_dKdr(r), -r*np.exp(-0.5*r**2))
    assert_allclose(k.calc_dKdr(r), (k.calc_K(r)-k.calc_K(r-dx))/dx, rtol = 1.e-5)

    with pytest.raises(AssertionError):
        k.calc_dKdr(-1.)

def test_squared_exponential_d2Kdr2():
    "test squared exponential d2K/dr2 function"

    k = SquaredExponential()

    dx = 1.e-6

    assert_allclose(k.calc_d2Kdr2(1.), 0.)
    assert_allclose(k.calc_d2Kdr2(1.),
                    (k.calc_dKdr(1.)-k.calc_dKdr(1.-dx))/dx, atol = 1.e-5)

    r = np.array([[1., 2.], [3., 4.]])

    assert_allclose(k.calc_d2Kdr2(r), (r**2 - 1.)*np.exp(-0.5*r**2))
    assert_allclose(k.calc_d2Kdr2(r),
                    (k.calc_dKdr(r)-k.calc_dKdr(r-dx))/dx, rtol = 1.e-5, atol = 1.e-5)

    with pytest.raises(AssertionError):
        k.calc_d2Kdr2(-1.)

def test_squared_exponential():
    "test squared exponential covariance kernel"

    k = SquaredExponential()

    x = np.array([[1.], [2.]])
    y = np.array([[2.], [3.]])
    params = np.array([0., 0.])

    assert_allclose(k.kernel_f(x, y, params),
                    np.exp(-0.5*np.array([[1., 2.], [0., 1.]])**2))

    x = np.array([[1., 2.], [2., 3.]])
    y = np.array([[2., 4.], [3., 1.]])
    params = np.array([0., 0., 0.])

    assert_allclose(k.kernel_f(x, y, params),
                    np.exp(-0.5*np.array([[np.sqrt(5.), np.sqrt(5.)],
                                          [1., np.sqrt(5.)]])**2))

    x = np.array([[1., 2.], [2., 3.]])
    y = np.array([[2., 4.], [3., 1.]])
    params = np.array([np.log(2.), np.log(4.), np.log(2.)])

    assert_allclose(k.kernel_f(x, y, params),
                    2.*np.exp(-0.5*np.array([[np.sqrt(1.*2.+4.*4.), np.sqrt(4.*2.+1.*4.)],
                                             [np.sqrt(1.*4.), np.sqrt(1.*2.+4.*4.)]])**2))

    x = np.array([1., 2.])
    y = np.array([2., 3.])
    params = np.array([0., 0.])

    assert_allclose(k.kernel_f(x, y, params),
                    np.exp(-0.5*np.array([[1., 2.], [0., 1.]])**2))

def test_squared_exponential_failures():
    "test scenarios where squared_exponential should raise an exception"

    k = SquaredExponential()

    x = np.array([[1.], [2.]])
    y = np.array([[2.], [3.]])
    params = np.array([0.])

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
    params = np.array([0., 0.])

    deriv = np.zeros((2, 2, 2))

    deriv[-1] = np.exp(-0.5*np.array([[1., 2.], [0., 1.]])**2)
    deriv[0] = (-0.5*np.array([[1., 4.], [0., 1.]])*
                np.exp(-0.5*np.array([[1., 2.], [0., 1.]])**2))
    deriv_fd = np.zeros((2, 2, 2))
    deriv_fd[0] = (k.kernel_f(x, y, params) -
                   k.kernel_f(x, y, params - np.array([dx, 0.])))/dx
    deriv_fd[1] = (k.kernel_f(x, y, params) -
                   k.kernel_f(x, y, params - np.array([0., dx])))/dx

    assert_allclose(k.kernel_deriv(x, y, params), deriv)
    assert_allclose(k.kernel_deriv(x, y, params), deriv_fd, rtol = 1.e-5)

    x = np.array([[1., 2.], [2., 3.]])
    y = np.array([[2., 4.], [3., 1.]])
    params = np.array([0., 0., 0.])

    deriv = np.zeros((3, 2, 2))

    deriv[-1] = np.exp(-0.5*np.array([[np.sqrt(5.), np.sqrt(5.)],
                                      [1., np.sqrt(5.)]])**2)
    deriv[0] = -0.5*np.array([[1., 4.], [0., 1.]])*deriv[-1]
    deriv[1] = -0.5*np.array([[4., 1.], [1., 4.]])*deriv[-1]
    deriv_fd = np.zeros((3, 2, 2))
    deriv_fd[0] = (k.kernel_f(x, y, params) -
                   k.kernel_f(x, y, params - np.array([dx, 0., 0.])))/dx
    deriv_fd[1] = (k.kernel_f(x, y, params) -
                   k.kernel_f(x, y, params - np.array([0., dx, 0.])))/dx
    deriv_fd[2] = (k.kernel_f(x, y, params) -
                   k.kernel_f(x, y, params - np.array([0., 0., dx])))/dx

    assert_allclose(k.kernel_deriv(x, y, params), deriv)
    assert_allclose(k.kernel_deriv(x, y, params), deriv_fd, rtol = 1.e-5)

    x = np.array([[1., 2.], [2., 3.]])
    y = np.array([[2., 4.], [3., 1.]])
    params = np.array([np.log(2.), np.log(4.), np.log(2.)])

    deriv = np.zeros((3, 2, 2))

    deriv[-1] = 2.*np.exp(-0.5*np.array([[np.sqrt(1.*2.+4.*4.), np.sqrt(4.*2.+1.*4.)],
                                         [np.sqrt(1.*4.), np.sqrt(1.*2.+4.*4.)]])**2)
    deriv[0] = -0.5*np.array([[2., 8.], [0., 2.]])*deriv[-1]
    deriv[1] = -0.5*np.array([[16., 4.], [4., 16.]])*deriv[-1]
    deriv_fd = np.zeros((3, 2, 2))
    deriv_fd[0] = (k.kernel_f(x, y, params) -
                   k.kernel_f(x, y, params - np.array([dx, 0., 0.])))/dx
    deriv_fd[1] = (k.kernel_f(x, y, params) -
                   k.kernel_f(x, y, params - np.array([0., dx, 0.])))/dx
    deriv_fd[2] = (k.kernel_f(x, y, params) -
                   k.kernel_f(x, y, params - np.array([0., 0., dx])))/dx

    assert_allclose(k.kernel_deriv(x, y, params), deriv)
    assert_allclose(k.kernel_deriv(x, y, params), deriv_fd, rtol = 1.e-5)

    x = np.array([1., 2.])
    y = np.array([2., 3.])
    params = np.array([0., 0.])

    deriv = np.zeros((2, 2, 2))

    deriv[-1] = np.exp(-0.5*np.array([[1., 2.], [0., 1.]])**2)
    deriv[0] = (-0.5*np.array([[1., 4.],[0., 1.]])*
                np.exp(-0.5*np.array([[1., 2.], [0., 1.]])**2))
    deriv_fd = np.zeros((2, 2, 2))
    deriv_fd[0] = (k.kernel_f(x, y, params) -
                   k.kernel_f(x, y, params - np.array([dx, 0.])))/dx
    deriv_fd[1] = (k.kernel_f(x, y, params) -
                   k.kernel_f(x, y, params - np.array([0., dx])))/dx

    assert_allclose(k.kernel_deriv(x, y, params), deriv)
    assert_allclose(k.kernel_deriv(x, y, params), deriv_fd, rtol = 1.e-5)

def test_squared_exponential_deriv_failures():
    "test scenarios where squared_exponential should raise an exception"

    k = SquaredExponential()

    x = np.array([[1.], [2.]])
    y = np.array([[2.], [3.]])
    params = np.array([0.])

    with pytest.raises(AssertionError):
        k.kernel_deriv(x, y, params)

    params = np.array([[0., 0.], [0., 0.]])

    with pytest.raises(AssertionError):
        k.kernel_deriv(x, y, params)

    x = np.array([[1.], [2.]])
    y = np.array([[2., 4.], [3., 2.]])
    params = np.array([0., 0.])

    with pytest.raises(AssertionError):
        k.kernel_deriv(x, y, params)

    x = np.array([[1.], [2.]])
    y = np.array([[[2.], [4.]], [[3.], [2.]]])
    params = np.array([0., 0.])

    with pytest.raises(AssertionError):
        k.kernel_deriv(x, y, params)

    x = np.array([[2., 4.], [3., 2.]])
    y = np.array([[1.], [2.]])
    params = np.array([0., 0.])

    with pytest.raises(AssertionError):
        k.kernel_deriv(x, y, params)

    x = np.array([[[2.], [4.]], [[3.], [2.]]])
    y = np.array([[1.], [2.]])
    params = np.array([0., 0.])

    with pytest.raises(AssertionError):
        k.kernel_deriv(x, y, params)

def test_squared_exponential_hessian():
    "test the function to compute the squared exponential hessian"

    k = SquaredExponential()

    dx = 1.e-6

    x = np.array([[1.], [2.]])
    y = np.array([[2.], [3.]])
    params = np.array([0., 0.])

    hess = np.zeros((2, 2, 2, 2))
    r2 = np.array([[1., 4.], [0., 1.]])
    hess[0, 0] = (-0.5*r2+0.25*r2**2)*np.exp(-0.5*r2)
    hess[0, 1] = -0.5*np.exp(-0.5*r2)*r2
    hess[1, 0] = -0.5*np.exp(-0.5*r2)*r2
    hess[1, 1] = np.exp(-0.5*r2)
    hess_fd = np.zeros((2, 2, 2, 2))
    hess_fd[0, 0] = (k.kernel_deriv(x, y, params)[0] -
                     k.kernel_deriv(x, y, params-np.array([dx, 0.]))[0])/dx
    hess_fd[1, 0] = (k.kernel_deriv(x, y, params)[1] -
                     k.kernel_deriv(x, y, params-np.array([dx, 0.]))[1])/dx
    hess_fd[0, 1] = (k.kernel_deriv(x, y, params)[0] -
                     k.kernel_deriv(x, y, params-np.array([0., dx]))[0])/dx
    hess_fd[1, 1] = (k.kernel_deriv(x, y, params)[1] -
                     k.kernel_deriv(x, y, params-np.array([0., dx]))[1])/dx

    assert_allclose(k.kernel_hessian(x, y, params), hess)
    assert_allclose(k.kernel_hessian(x, y, params), hess_fd, atol = 1.e-5)

    x = np.array([[1., 2.], [2., 3.]])
    y = np.array([[2., 4.], [3., 1.]])
    params = np.array([0., 0., 0.])

    hess = np.zeros((3, 3, 2, 2))

    r2 = np.array([[np.sqrt(5.), np.sqrt(5.)], [1., np.sqrt(5.)]])**2
    x12 = np.array([[1., 4.],[0., 1.]])
    x22 = np.array([[4., 1.],[1., 4.]])
    hess[0, 0] = (-0.5*x12+0.25*x12**2)*np.exp(-0.5*r2)
    hess[0, 1] = 0.25*np.exp(-0.5*r2)*x12*x22
    hess[1, 0] = 0.25*np.exp(-0.5*r2)*x12*x22
    hess[1, 1] = (-0.5*x22+0.25*x22**2)*np.exp(-0.5*r2)
    hess[0, 2] = -0.5*np.exp(-0.5*r2)*x12
    hess[2, 0] = -0.5*np.exp(-0.5*r2)*x12
    hess[1, 2] = -0.5*np.exp(-0.5*r2)*x22
    hess[2, 1] = -0.5*np.exp(-0.5*r2)*x22
    hess[2, 2] = np.exp(-0.5*r2)
    hess_fd = np.zeros((3, 3, 2, 2))
    hess_fd[0, 0] = (k.kernel_deriv(x, y, params)[0] -
                     k.kernel_deriv(x, y, params-np.array([dx, 0., 0.]))[0])/dx
    hess_fd[1, 0] = (k.kernel_deriv(x, y, params)[1] -
                     k.kernel_deriv(x, y, params-np.array([dx, 0., 0.]))[1])/dx
    hess_fd[0, 1] = (k.kernel_deriv(x, y, params)[0] -
                     k.kernel_deriv(x, y, params-np.array([0., dx, 0.]))[0])/dx
    hess_fd[1, 1] = (k.kernel_deriv(x, y, params)[1] -
                     k.kernel_deriv(x, y, params-np.array([0., dx, 0.]))[1])/dx
    hess_fd[0, 2] = (k.kernel_deriv(x, y, params)[0] -
                     k.kernel_deriv(x, y, params-np.array([0., 0., dx]))[0])/dx
    hess_fd[2, 0] = (k.kernel_deriv(x, y, params)[2] -
                     k.kernel_deriv(x, y, params-np.array([dx, 0., 0.]))[2])/dx
    hess_fd[2, 1] = (k.kernel_deriv(x, y, params)[2] -
                     k.kernel_deriv(x, y, params-np.array([0., dx, 0.]))[2])/dx
    hess_fd[1, 2] = (k.kernel_deriv(x, y, params)[1] -
                     k.kernel_deriv(x, y, params-np.array([0., 0., dx]))[1])/dx
    hess_fd[2, 2] = (k.kernel_deriv(x, y, params)[2] -
                     k.kernel_deriv(x, y, params-np.array([0., 0., dx]))[2])/dx

    assert_allclose(k.kernel_hessian(x, y, params), hess)
    assert_allclose(k.kernel_hessian(x, y, params), hess_fd, atol = 1.e-5)

    x = np.array([[1., 2.], [2., 3.]])
    y = np.array([[2., 4.], [3., 1.]])
    params = np.array([np.log(2.), np.log(4.), np.log(2.)])

    hess = np.zeros((3, 3, 2, 2))

    r2 = np.array([[np.sqrt(1.*2.+4.*4.), np.sqrt(4.*2.+1.*4.)],
                   [np.sqrt(1.*4.), np.sqrt(1.*2.+4.*4.)]])**2
    x12 = np.array([[1., 4.],[0., 1.]])
    x22 = np.array([[4., 1.],[1., 4.]])
    hess[0, 0] = (-0.5*x12+0.25*x12**2)*2.*2.*2.*np.exp(-0.5*r2)
    hess[0, 1] = 0.25*2.*4.*2.*np.exp(-0.5*r2)*x12*x22
    hess[1, 0] = 0.25*np.exp(-0.5*r2)*x12*x22
    hess[1, 1] = (-0.5*x22+0.25*x22**2)*np.exp(-0.5*r2)
    hess[0, 2] = -0.5*np.exp(-0.5*r2)*x12
    hess[2, 0] = -0.5*np.exp(-0.5*r2)*x12
    hess[1, 2] = -0.5*2.*4.*np.exp(-0.5*r2)*x22
    hess[2, 1] = -0.5*2.*4.*np.exp(-0.5*r2)*x22
    hess[2, 2] = 2.*np.exp(-0.5*r2)
    hess_fd = np.zeros((3, 3, 2, 2))
    hess_fd[0, 0] = (k.kernel_deriv(x, y, params)[0] -
                     k.kernel_deriv(x, y, params-np.array([dx, 0., 0.]))[0])/dx
    hess_fd[1, 0] = (k.kernel_deriv(x, y, params)[1] -
                     k.kernel_deriv(x, y, params-np.array([dx, 0., 0.]))[1])/dx
    hess_fd[0, 1] = (k.kernel_deriv(x, y, params)[0] -
                     k.kernel_deriv(x, y, params-np.array([0., dx, 0.]))[0])/dx
    hess_fd[1, 1] = (k.kernel_deriv(x, y, params)[1] -
                     k.kernel_deriv(x, y, params-np.array([0., dx, 0.]))[1])/dx
    hess_fd[0, 2] = (k.kernel_deriv(x, y, params)[0] -
                     k.kernel_deriv(x, y, params-np.array([0., 0., dx]))[0])/dx
    hess_fd[2, 0] = (k.kernel_deriv(x, y, params)[2] -
                     k.kernel_deriv(x, y, params-np.array([dx, 0., 0.]))[2])/dx
    hess_fd[2, 1] = (k.kernel_deriv(x, y, params)[2] -
                     k.kernel_deriv(x, y, params-np.array([0., dx, 0.]))[2])/dx
    hess_fd[1, 2] = (k.kernel_deriv(x, y, params)[1] -
                     k.kernel_deriv(x, y, params-np.array([0., 0., dx]))[1])/dx
    hess_fd[2, 2] = (k.kernel_deriv(x, y, params)[2] -
                     k.kernel_deriv(x, y, params-np.array([0., 0., dx]))[2])/dx

    assert_allclose(k.kernel_hessian(x, y, params)[1,2], hess[1,2])
    assert_allclose(k.kernel_hessian(x, y, params), hess_fd, atol = 1.e-5)

    x = np.array([1., 2.])
    y = np.array([2., 3.])
    params = np.array([0., 0.])

    hess = np.zeros((2, 2, 2, 2))
    r2 = np.array([[1., 4.], [0., 1.]])
    hess[0, 0] = (-0.5*r2+0.25*r2**2)*np.exp(-0.5*r2)
    hess[0, 1] = -0.5*np.exp(-0.5*r2)*r2
    hess[1, 0] = -0.5*np.exp(-0.5*r2)*r2
    hess[1, 1] = np.exp(-0.5*r2)
    hess_fd = np.zeros((2, 2, 2, 2))
    hess_fd[0, 0] = (k.kernel_deriv(x, y, params)[0] -
                     k.kernel_deriv(x, y, params-np.array([dx, 0.]))[0])/dx
    hess_fd[1, 0] = (k.kernel_deriv(x, y, params)[1] -
                     k.kernel_deriv(x, y, params-np.array([dx, 0.]))[1])/dx
    hess_fd[0, 1] = (k.kernel_deriv(x, y, params)[0] -
                     k.kernel_deriv(x, y, params-np.array([0., dx]))[0])/dx
    hess_fd[1, 1] = (k.kernel_deriv(x, y, params)[1] -
                     k.kernel_deriv(x, y, params-np.array([0., dx]))[1])/dx

    assert_allclose(k.kernel_hessian(x, y, params), hess)
    assert_allclose(k.kernel_hessian(x, y, params), hess_fd, atol = 1.e-5)

def test_squared_exponential_hessian_failures():
    "test situaitons where squared_exponential_hessian should fail"

    k = SquaredExponential()

    x = np.array([[1.], [2.]])
    y = np.array([[2.], [3.]])
    params = np.array([0.])

    with pytest.raises(AssertionError):
        k.kernel_hessian(x, y, params)

    params = np.array([[0., 0.], [0., 0.]])

    with pytest.raises(AssertionError):
        k.kernel_hessian(x, y, params)

    x = np.array([[1.], [2.]])
    y = np.array([[2., 4.], [3., 2.]])
    params = np.array([0., 0.])

    with pytest.raises(AssertionError):
        k.kernel_hessian(x, y, params)

    x = np.array([[1.], [2.]])
    y = np.array([[[2.], [4.]], [[3.], [2.]]])
    params = np.array([0., 0.])

    with pytest.raises(AssertionError):
        k.kernel_hessian(x, y, params)

    x = np.array([[2., 4.], [3., 2.]])
    y = np.array([[1.], [2.]])
    params = np.array([0., 0.])

    with pytest.raises(AssertionError):
        k.kernel_hessian(x, y, params)

    x = np.array([[[2.], [4.]], [[3.], [2.]]])
    y = np.array([[1.], [2.]])
    params = np.array([0., 0.])

    with pytest.raises(AssertionError):
        k.kernel_hessian(x, y, params)

def test_squared_exponential_inputderiv():
    "test the input derivative method of squared exponential"

    k = SquaredExponential()

    dx = 1.e-6

    x = np.array([[1.], [2.]])
    y = np.array([[2.], [3.]])
    params = np.array([0., 0.])

    deriv = np.zeros((1, 2, 2))

    r = np.array([[1., 2.], [0., 1.]])

    deriv[0] = -r*np.exp(-0.5*r**2)*np.array([[-1., -1.], [0., -1.]])
    deriv_fd = np.zeros((1, 2, 2))
    deriv_fd[0] = (k.kernel_f(x + dx, y, params) -
                   k.kernel_f(x - dx, y, params))/dx/2.

    assert_allclose(k.kernel_inputderiv(x, y, params), deriv)
    assert_allclose(k.kernel_inputderiv(x, y, params), deriv_fd, rtol = 1.e-5)

    x = np.array([[1., 2.], [2., 3.]])
    y = np.array([[2., 4.], [3., 1.]])
    params = np.array([0., 0., 0.])

    deriv = np.zeros((2, 2, 2))

    r = np.array([[np.sqrt(5.), np.sqrt(5.)], [1., np.sqrt(5.)]])

    deriv[0] = -np.exp(-0.5*r**2)*np.array([[-1., -2.], [ 0., -1.]])
    deriv[1] = -np.exp(-0.5*r**2)*np.array([[-2.,  1.], [-1.,  2.]])
    deriv_fd = np.zeros((2, 2, 2))
    deriv_fd[0] = (k.kernel_f(x + np.array([[dx, 0.], [dx, 0.]]), y, params) -
                   k.kernel_f(x - np.array([[dx, 0.], [dx, 0.]]), y, params))/dx/2.
    deriv_fd[1] = (k.kernel_f(x + np.array([[0., dx], [0., dx]]), y, params) -
                   k.kernel_f(x - np.array([[0., dx], [0., dx]]), y, params))/dx/2.

    assert_allclose(k.kernel_inputderiv(x, y, params), deriv)
    assert_allclose(k.kernel_inputderiv(x, y, params), deriv_fd, rtol = 1.e-5)

    x = np.array([[1., 2.], [2., 3.]])
    y = np.array([[2., 4.], [3., 1.]])
    params = np.array([np.log(2.), np.log(4.), np.log(2.)])

    deriv = np.zeros((2, 2, 2))

    r = np.array([[np.sqrt(1.*2.+4.*4.), np.sqrt(4.*2.+1.*4.)],
                  [np.sqrt(1.*4.), np.sqrt(1.*2.+4.*4.)]])

    deriv[0] = (-2.*2.*np.exp(-0.5*r**2)*
                np.array([[-1., -2.], [0., -1.]]))
    deriv[1] = (-2.*4.*np.exp(-0.5*r**2)*
                np.array([[-2., 1.], [-1., 2.]]))
    deriv_fd = np.zeros((2, 2, 2))
    deriv_fd[0] = (k.kernel_f(x + np.array([[dx, 0.], [dx, 0.]]), y, params) -
                   k.kernel_f(x - np.array([[dx, 0.], [dx, 0.]]), y, params))/dx/2.
    deriv_fd[1] = (k.kernel_f(x + np.array([[0., dx], [0., dx]]), y, params) -
                   k.kernel_f(x - np.array([[0., dx], [0., dx]]), y, params))/dx/2.

    assert_allclose(k.kernel_inputderiv(x, y, params), deriv)
    assert_allclose(k.kernel_inputderiv(x, y, params), deriv_fd, rtol = 1.e-5)

    x = np.array([1., 2.])
    y = np.array([2., 3.])
    params = np.array([0., 0.])

    deriv = np.zeros((1, 2, 2))

    r = np.array([[1., 2.], [0., 1.]])

    deriv[0] = -r*np.exp(-0.5*r**2)*np.array([[-1., -1.], [0., -1.]])
    deriv_fd = np.zeros((1, 2, 2))
    deriv_fd[0] = (k.kernel_f(x + dx, y, params)-k.kernel_f(x - dx, y, params))/dx/2.

    assert_allclose(k.kernel_inputderiv(x, y, params), deriv)
    assert_allclose(k.kernel_inputderiv(x, y, params), deriv_fd, rtol = 1.e-5)

def test_squared_exponential_inputderiv_failures():
    "test situations where input derivative method should fail"

    k = SquaredExponential()

    x = np.array([[1.], [2.]])
    y = np.array([[2.], [3.]])
    params = np.array([0.])

    with pytest.raises(AssertionError):
       k.kernel_inputderiv(x, y, params)

    params = np.array([[0., 0.], [0., 0.]])

    with pytest.raises(AssertionError):
       k.kernel_inputderiv(x, y, params)

    x = np.array([[1.], [2.]])
    y = np.array([[2., 4.], [3., 2.]])
    params = np.array([0., 0.])

    with pytest.raises(AssertionError):
       k.kernel_inputderiv(x, y, params)

    x = np.array([[1.], [2.]])
    y = np.array([[[2.], [4.]], [[3.], [2.]]])
    params = np.array([0., 0.])

    with pytest.raises(AssertionError):
       k.kernel_inputderiv(x, y, params)

    x = np.array([[2., 4.], [3., 2.]])
    y = np.array([[1.], [2.]])
    params = np.array([0., 0.])

    with pytest.raises(AssertionError):
       k.kernel_inputderiv(x, y, params)

    x = np.array([[[2.], [4.]], [[3.], [2.]]])
    y = np.array([[1.], [2.]])
    params = np.array([0., 0.])

    with pytest.raises(AssertionError):
       k.kernel_inputderiv(x, y, params)

def test_matern_5_2_K():
    "test matern 5/2 K(r) function"

    k = Matern52()

    assert_allclose(k.calc_K(1.), (1.+np.sqrt(5.)+5./3.)*np.exp(-np.sqrt(5.)))

    r = np.array([[1., 2.], [3., 4.]])

    assert_allclose(k.calc_K(r), (1.+np.sqrt(5.)*r+5./3.*r**2)*np.exp(-np.sqrt(5.)*r))

    with pytest.raises(AssertionError):
        k.calc_K(-1.)

def test_matern_5_2_dKdr():
    "test matern 5/2 dK/dr function"

    k = Matern52()

    dx = 1.e-6

    assert_allclose(k.calc_dKdr(1.), -5./3.*(1.+np.sqrt(5.))*np.exp(-np.sqrt(5.)))
    assert_allclose(k.calc_dKdr(1.), (k.calc_K(1.)-k.calc_K(1.-dx))/dx, rtol = 1.e-5)

    r = np.array([[1., 2.], [3., 4.]])

    assert_allclose(k.calc_dKdr(r), -5./3.*r*(1.+np.sqrt(5.)*r)*np.exp(-np.sqrt(5.)*r))
    assert_allclose(k.calc_dKdr(r), (k.calc_K(r)-k.calc_K(r - dx))/dx, rtol = 1.e-5)

    with pytest.raises(AssertionError):
        k.calc_dKdr(-1.)

def test_matern_5_2_d2Kdr2():
    "test squared exponential d2K/dr2 function"

    k = Matern52()

    dx = 1.e-6

    assert_allclose(k.calc_d2Kdr2(1.), 5./3.*(5.-np.sqrt(5.)-1.)*np.exp(-np.sqrt(5.)))
    assert_allclose(k.calc_d2Kdr2(1.), (k.calc_dKdr(1.)-k.calc_dKdr(1.-dx))/dx, rtol = 1.e-5)

    r = np.array([[1., 2.], [3., 4.]])

    assert_allclose(k.calc_d2Kdr2(r), 5./3.*(5.*r**2-np.sqrt(5.)*r-1.)*np.exp(-np.sqrt(5.)*r))
    assert_allclose(k.calc_d2Kdr2(r), (k.calc_dKdr(r)-k.calc_dKdr(r - dx))/dx, rtol = 1.e-5)

    with pytest.raises(AssertionError):
        k.calc_d2Kdr2(-1.)

def test_matern_5_2():
    "test matern 5/2 covariance kernel"

    k = Matern52()

    x = np.array([[1.], [2.]])
    y = np.array([[2.], [3.]])
    params = np.array([0., 0.])

    D = np.array([[1., 2.], [0., 1.]])

    assert_allclose(k.kernel_f(x, y, params),
                    (1.+np.sqrt(5.)*D+5./3.*D**2)*np.exp(-np.sqrt(5.)*D))

    x = np.array([[1., 2.], [2., 3.]])
    y = np.array([[2., 4.], [3., 1.]])
    params = np.array([0., 0., 0.])

    D = np.array([[np.sqrt(5.), np.sqrt(5.)], [1., np.sqrt(5.)]])

    assert_allclose(k.kernel_f(x, y, params),
                    (1.+np.sqrt(5.)*D+5./3.*D**2)*np.exp(-np.sqrt(5.)*D))

    x = np.array([[1., 2.], [2., 3.]])
    y = np.array([[2., 4.], [3., 1.]])
    params = np.array([np.log(2.), np.log(4.), np.log(2.)])

    D = np.array([[np.sqrt(1.*2.+4.*4.), np.sqrt(4.*2.+1.*4.)],
                  [np.sqrt(1.*4.), np.sqrt(1.*2.+4.*4.)]])

    assert_allclose(k.kernel_f(x, y, params),
                    2.*(1.+np.sqrt(5.)*D+5./3.*D**2)*np.exp(-np.sqrt(5.)*D))

    x = np.array([1., 2.])
    y = np.array([2., 3.])
    params = np.array([0., 0.])

    D = np.array([[1., 2.], [0., 1.]])

    assert_allclose(k.kernel_f(x, y, params),
                    (1.+np.sqrt(5.)*D + 5./3.*D**2)*np.exp(-np.sqrt(5.)*D))

def test_matern_5_2_failures():
    "test scenarios where matern_5_2 should raise an exception"

    k = Matern52()

    x = np.array([[1.], [2.]])
    y = np.array([[2.], [3.]])
    params = np.array([0.])

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
    params = np.array([0., 0.])

    deriv = np.zeros((2, 2, 2))

    D = np.array([[1., 2.], [0., 1.]])

    deriv[-1] = (1.+np.sqrt(5.)*D+5./3.*D**2)*np.exp(-np.sqrt(5.)*D)
    deriv[0] = -0.5*D**2*5./3.*(1.+np.sqrt(5.)*D)*np.exp(-np.sqrt(5.)*D)

    deriv_fd = np.zeros((2, 2, 2))
    deriv_fd[0] = (k.kernel_f(x, y, params) -
                   k.kernel_f(x, y, params - np.array([dx, 0.])))/dx
    deriv_fd[1] = (k.kernel_f(x, y, params) -
                   k.kernel_f(x, y, params - np.array([0., dx])))/dx

    assert_allclose(k.kernel_deriv(x, y, params), deriv)
    assert_allclose(k.kernel_deriv(x, y, params), deriv_fd, rtol = 1.e-5)

    x = np.array([[1., 2.], [2., 3.]])
    y = np.array([[2., 4.], [3., 1.]])
    params = np.array([0., 0., 0.])

    D = np.array([[np.sqrt(5.), np.sqrt(5.)], [1., np.sqrt(5.)]])
    D1 = np.array([[1., 2.], [0., 1.]])
    D2 = np.array([[2., 1.], [1., 2.]])

    deriv = np.zeros((3, 2, 2))

    deriv[-1] = (1.+np.sqrt(5.)*D+5./3.*D**2)*np.exp(-np.sqrt(5.)*D)
    deriv[0] = -0.5*D1**2*5./3.*(1.+np.sqrt(5.)*D)*np.exp(-np.sqrt(5.)*D)
    deriv[1] = -0.5*D2**2*5./3.*(1.+np.sqrt(5.)*D)*np.exp(-np.sqrt(5.)*D)

    deriv_fd = np.zeros((3, 2, 2))
    deriv_fd[0] = (k.kernel_f(x, y, params) -
                   k.kernel_f(x, y, params - np.array([dx, 0., 0.])))/dx
    deriv_fd[1] = (k.kernel_f(x, y, params) -
                   k.kernel_f(x, y, params - np.array([0., dx, 0.])))/dx
    deriv_fd[2] = (k.kernel_f(x, y, params) -
                   k.kernel_f(x, y, params - np.array([0., 0., dx])))/dx

    assert_allclose(k.kernel_deriv(x, y, params), deriv)
    assert_allclose(k.kernel_deriv(x, y, params), deriv_fd, rtol = 1.e-5)

    x = np.array([[1., 2.], [2., 3.]])
    y = np.array([[2., 4.], [3., 1.]])
    params = np.array([np.log(2.), np.log(4.), np.log(2.)])

    D = np.array([[np.sqrt(1.*2.+4.*4.), np.sqrt(4.*2.+1.*4.)],
                  [np.sqrt(1.*4.), np.sqrt(1.*2.+4.*4.)]])
    D1 = np.array([[1., 2.], [0., 1.]])
    D2 = np.array([[2., 1.], [1., 2.]])

    deriv = np.zeros((3, 2, 2))

    deriv[-1] = 2.*(1.+np.sqrt(5.)*D+5./3.*D**2)*np.exp(-np.sqrt(5.)*D)
    deriv[0] = -0.5*2.*2.*D1**2*5./3.*(1.+np.sqrt(5.)*D)*np.exp(-np.sqrt(5.)*D)
    deriv[1] = -0.5*2.*4.*D2**2*5./3.*(1.+np.sqrt(5.)*D)*np.exp(-np.sqrt(5.)*D)

    deriv_fd = np.zeros((3, 2, 2))
    deriv_fd[0] = (k.kernel_f(x, y, params) -
                   k.kernel_f(x, y, params - np.array([dx, 0., 0.])))/dx
    deriv_fd[1] = (k.kernel_f(x, y, params) -
                   k.kernel_f(x, y, params - np.array([0., dx, 0.])))/dx
    deriv_fd[2] = (k.kernel_f(x, y, params) -
                   k.kernel_f(x, y, params - np.array([0., 0., dx])))/dx

    assert_allclose(k.kernel_deriv(x, y, params), deriv)
    assert_allclose(k.kernel_deriv(x, y, params), deriv_fd, rtol = 1.e-5)

    x = np.array([1., 2.])
    y = np.array([2., 3.])
    params = np.array([0., 0.])

    deriv = np.zeros((2, 2, 2))

    D = np.array([[1., 2.], [0., 1.]])

    deriv[-1] = (1.+np.sqrt(5.)*D+5./3.*D**2)*np.exp(-np.sqrt(5.)*D)
    deriv[0] = -0.5*D**2*5./3.*(1.+np.sqrt(5.)*D)*np.exp(-np.sqrt(5.)*D)

    deriv_fd = np.zeros((2, 2, 2))
    deriv_fd[0] = (k.kernel_f(x, y, params) -
                   k.kernel_f(x, y, params - np.array([dx, 0.])))/dx
    deriv_fd[1] = (k.kernel_f(x, y, params) -
                   k.kernel_f(x, y, params - np.array([0, dx])))/dx

    assert_allclose(k.kernel_deriv(x, y, params), deriv)
    assert_allclose(k.kernel_deriv(x, y, params), deriv_fd, rtol = 1.e-5)

def test_matern_5_2_deriv_failures():
    "test scenarios where matern_5_2_deriv should raise an exception"

    k = Matern52()

    x = np.array([[1.], [2.]])
    y = np.array([[2.], [3.]])
    params = np.array([0.])

    with pytest.raises(AssertionError):
        k.kernel_deriv(x, y, params)

    params = np.array([[0., 0.], [0., 0.]])

    with pytest.raises(AssertionError):
        k.kernel_deriv(x, y, params)

    x = np.array([[1.], [2.]])
    y = np.array([[2., 4.], [3., 2.]])
    params = np.array([0., 0.])

    with pytest.raises(AssertionError):
        k.kernel_deriv(x, y, params)

    x = np.array([[1.], [2.]])
    y = np.array([[[2.], [4.]], [[3.], [2.]]])
    params = np.array([0., 0.])

    with pytest.raises(AssertionError):
        k.kernel_deriv(x, y, params)

    x = np.array([[2., 4.], [3., 2.]])
    y = np.array([[1.], [2.]])
    params = np.array([0., 0.])

    with pytest.raises(AssertionError):
        k.kernel_deriv(x, y, params)

    x = np.array([[[2.], [4.]], [[3.], [2.]]])
    y = np.array([[1.], [2.]])
    params = np.array([0., 0.])

    with pytest.raises(AssertionError):
        k.kernel_deriv(x, y, params)

def test_matern_5_2_hessian():
    "test the function to compute the squared exponential hessian"

    k = Matern52()

    dx = 1.e-6

    x = np.array([[1.], [2.]])
    y = np.array([[2.], [3.]])
    params = np.array([0., 0.])

    D = np.array([[1., 2.], [0., 1.]])

    hess = np.zeros((2, 2, 2, 2))
    hess[0, 0] = 5./3.*np.exp(-np.sqrt(5.)*D)*(5./4.*D**4-(1.+np.sqrt(5.)*D)*D**2/2.)
    hess[0, 1] = -0.5*D**2*5./3.*(1.+np.sqrt(5.)*D)*np.exp(-np.sqrt(5.)*D)
    hess[1, 0] = -0.5*D**2*5./3.*(1.+np.sqrt(5.)*D)*np.exp(-np.sqrt(5.)*D)
    hess[1, 1] = (1.+np.sqrt(5.)*D+5./3.*D**2)*np.exp(-np.sqrt(5.)*D)

    hess_fd = np.zeros((2, 2, 2, 2))
    hess_fd[0, 0] = (k.kernel_deriv(x, y, params)[0] -
                     k.kernel_deriv(x, y, params-np.array([dx, 0.]))[0])/dx
    hess_fd[0, 1] = (k.kernel_deriv(x, y, params)[0] -
                     k.kernel_deriv(x, y, params-np.array([0., dx]))[0])/dx
    hess_fd[1, 0] = (k.kernel_deriv(x, y, params)[1] -
                     k.kernel_deriv(x, y, params-np.array([dx, 0.]))[1])/dx
    hess_fd[1, 1] = (k.kernel_deriv(x, y, params)[1] -
                     k.kernel_deriv(x, y, params-np.array([0., dx]))[1])/dx

    assert_allclose(k.kernel_hessian(x, y, params), hess)
    assert_allclose(k.kernel_hessian(x, y, params), hess_fd, rtol = 1.e-5)

    x = np.array([[1., 2.], [2., 3.]])
    y = np.array([[2., 4.], [3., 1.]])
    params = np.array([0., 0., 0.])

    D = np.array([[np.sqrt(5.), np.sqrt(5.)], [1., np.sqrt(5.)]])
    D1 = np.array([[1., 2.], [0., 1.]])
    D2 = np.array([[2., 1.], [1., 2.]])

    hess = np.zeros((3, 3, 2, 2))

    hess[0, 0] = 5./3.*np.exp(-np.sqrt(5.)*D)*(5./4.*D1**4-(1.+np.sqrt(5.)*D)*D1**2/2.)
    hess[0, 1] = 5./3.*np.exp(-np.sqrt(5.)*D)*(5./4.*D1**2*D2**2)
    hess[1, 0] = 5./3.*np.exp(-np.sqrt(5.)*D)*(5./4.*D1**2*D2**2)
    hess[1, 1] = 5./3.*np.exp(-np.sqrt(5.)*D)*(5./4.*D2**4-(1.+np.sqrt(5.)*D)*D2**2/2.)
    hess[0, 2] = -0.5*D1**2*5./3.*(1.+np.sqrt(5.)*D)*np.exp(-np.sqrt(5.)*D)
    hess[2, 0] = -0.5*D1**2*5./3.*(1.+np.sqrt(5.)*D)*np.exp(-np.sqrt(5.)*D)
    hess[1, 2] = -0.5*D2**2*5./3.*(1.+np.sqrt(5.)*D)*np.exp(-np.sqrt(5.)*D)
    hess[2, 1] = -0.5*D2**2*5./3.*(1.+np.sqrt(5.)*D)*np.exp(-np.sqrt(5.)*D)
    hess[2, 2] = (1.+np.sqrt(5.)*D+5./3.*D**2)*np.exp(-np.sqrt(5.)*D)

    hess_fd = np.zeros((3, 3, 2, 2))
    hess_fd[0, 0] = (k.kernel_deriv(x, y, params)[0] -
                     k.kernel_deriv(x, y, params-np.array([dx, 0., 0.]))[0])/dx
    hess_fd[0, 1] = (k.kernel_deriv(x, y, params)[0] -
                     k.kernel_deriv(x, y, params-np.array([0., dx, 0.]))[0])/dx
    hess_fd[1, 0] = (k.kernel_deriv(x, y, params)[1] -
                     k.kernel_deriv(x, y, params-np.array([dx, 0., 0.]))[1])/dx
    hess_fd[1, 1] = (k.kernel_deriv(x, y, params)[1] -
                     k.kernel_deriv(x, y, params-np.array([0., dx, 0.]))[1])/dx
    hess_fd[0, 2] = (k.kernel_deriv(x, y, params)[0] -
                     k.kernel_deriv(x, y, params-np.array([0., 0., dx]))[0])/dx
    hess_fd[2, 0] = (k.kernel_deriv(x, y, params)[2] -
                     k.kernel_deriv(x, y, params-np.array([dx, 0., 0.]))[2])/dx
    hess_fd[1, 2] = (k.kernel_deriv(x, y, params)[1] -
                     k.kernel_deriv(x, y, params-np.array([0., 0., dx]))[1])/dx
    hess_fd[2, 1] = (k.kernel_deriv(x, y, params)[2] -
                     k.kernel_deriv(x, y, params-np.array([0., dx, 0.]))[2])/dx
    hess_fd[2, 2] = (k.kernel_deriv(x, y, params)[2] -
                     k.kernel_deriv(x, y, params-np.array([0., 0., dx]))[2])/dx

    assert_allclose(k.kernel_hessian(x, y, params), hess)
    assert_allclose(k.kernel_hessian(x, y, params), hess_fd, rtol = 1.e-5)

    x = np.array([[1., 2.], [2., 3.]])
    y = np.array([[2., 4.], [3., 1.]])
    params = np.array([np.log(2.), np.log(4.), np.log(2.)])

    D = np.array([[np.sqrt(1.*2.+4.*4.), np.sqrt(4.*2.+1.*4.)],
                  [np.sqrt(1.*4.), np.sqrt(1.*2.+4.*4.)]])
    D1 = np.array([[1., 2.], [0., 1.]])
    D2 = np.array([[2., 1.], [1., 2.]])

    hess = np.zeros((3, 3, 2, 2))

    hess[0, 0] = (5./3.*2.*np.exp(-np.sqrt(5.)*D)*
                  (5./4.*2.*2.*D1**4 - (1.+np.sqrt(5.)*D)*2.*D1**2/2.))
    hess[0, 1] = 5./3.*2.*np.exp(-np.sqrt(5.)*D)*(5./4.*2.*4.*D1**2*D2**2)
    hess[1, 0] = 5./3.*2.*np.exp(-np.sqrt(5.)*D)*(5./4.*2.*4.*D1**2*D2**2)
    hess[1, 1] = (5./3.*2.*np.exp(-np.sqrt(5.)*D)*(5./4.*4.*4.*D2**4 -
                  (1. + np.sqrt(5.)*D)*4.*D2**2/2.))
    hess[0, 2] = -0.5*2.*2.*D1**2*5./3.*(1. + np.sqrt(5.)*D)*np.exp(-np.sqrt(5.)*D)
    hess[2, 0] = -0.5*2.*2.*D1**2*5./3.*(1. + np.sqrt(5.)*D)*np.exp(-np.sqrt(5.)*D)
    hess[1, 2] = -0.5*2.*4.*D2**2*5./3.*(1. + np.sqrt(5.)*D)*np.exp(-np.sqrt(5.)*D)
    hess[2, 1] = -0.5*2.*4.*D2**2*5./3.*(1. + np.sqrt(5.)*D)*np.exp(-np.sqrt(5.)*D)
    hess[2, 2] = 2.*(1. + np.sqrt(5.)*D+5./3.*D**2)*np.exp(-np.sqrt(5.)*D)

    hess_fd = np.zeros((3, 3, 2, 2))
    hess_fd[0, 0] = (k.kernel_deriv(x, y, params)[0] -
                     k.kernel_deriv(x, y, params-np.array([dx, 0., 0.]))[0])/dx
    hess_fd[0, 1] = (k.kernel_deriv(x, y, params)[0] -
                     k.kernel_deriv(x, y, params-np.array([0., dx, 0.]))[0])/dx
    hess_fd[1, 0] = (k.kernel_deriv(x, y, params)[1] -
                     k.kernel_deriv(x, y, params-np.array([dx, 0., 0.]))[1])/dx
    hess_fd[1, 1] = (k.kernel_deriv(x, y, params)[1] -
                     k.kernel_deriv(x, y, params-np.array([0., dx, 0.]))[1])/dx
    hess_fd[0, 2] = (k.kernel_deriv(x, y, params)[0] -
                     k.kernel_deriv(x, y, params-np.array([0., 0., dx]))[0])/dx
    hess_fd[2, 0] = (k.kernel_deriv(x, y, params)[2] -
                     k.kernel_deriv(x, y, params-np.array([dx, 0., 0.]))[2])/dx
    hess_fd[1, 2] = (k.kernel_deriv(x, y, params)[1] -
                     k.kernel_deriv(x, y, params-np.array([0., 0., dx]))[1])/dx
    hess_fd[2, 1] = (k.kernel_deriv(x, y, params)[2] -
                     k.kernel_deriv(x, y, params-np.array([0., dx, 0.]))[2])/dx
    hess_fd[2, 2] = (k.kernel_deriv(x, y, params)[2] -
                     k.kernel_deriv(x, y, params-np.array([0., 0., dx]))[2])/dx

    assert_allclose(k.kernel_hessian(x, y, params), hess)
    assert_allclose(k.kernel_hessian(x, y, params), hess_fd, rtol = 1.e-5)

    x = np.array([1., 2.])
    y = np.array([2., 3.])
    params = np.array([0., 0.])

    D = np.array([[1., 2.], [0., 1.]])

    hess = np.zeros((2, 2, 2, 2))
    hess[0, 0] = 5./3.*np.exp(-np.sqrt(5.)*D)*(5./4.*D**4-(1.+np.sqrt(5.)*D)*D**2/2.)
    hess[0, 1] = -0.5*D**2*5./3.*(1.+np.sqrt(5.)*D)*np.exp(-np.sqrt(5.)*D)
    hess[1, 0] = -0.5*D**2*5./3.*(1.+np.sqrt(5.)*D)*np.exp(-np.sqrt(5.)*D)
    hess[1, 1] = (1.+np.sqrt(5.)*D+5./3.*D**2)*np.exp(-np.sqrt(5.)*D)

    hess_fd = np.zeros((2, 2, 2, 2))
    hess_fd[0, 0] = (k.kernel_deriv(x, y, params)[0] -
                     k.kernel_deriv(x, y, params-np.array([dx, 0.]))[0])/dx
    hess_fd[0, 1] = (k.kernel_deriv(x, y, params)[0] -
                     k.kernel_deriv(x, y, params-np.array([0., dx]))[0])/dx
    hess_fd[1, 0] = (k.kernel_deriv(x, y, params)[1] -
                     k.kernel_deriv(x, y, params-np.array([dx, 0.]))[1])/dx
    hess_fd[1, 1] = (k.kernel_deriv(x, y, params)[1] -
                     k.kernel_deriv(x, y, params-np.array([0., dx]))[1])/dx

    assert_allclose(k.kernel_hessian(x, y, params), hess)
    assert_allclose(k.kernel_hessian(x, y, params), hess_fd, rtol = 1.e-5)

def test_matern_5_2_hessian_failures():
    "test situaitons where squared_exponential_hessian should fail"

    k = Matern52()

    x = np.array([[1.], [2.]])
    y = np.array([[2.], [3.]])
    params = np.array([0.])

    with pytest.raises(AssertionError):
        k.kernel_hessian(x, y, params)

    params = np.array([[0., 0.], [0., 0.]])

    with pytest.raises(AssertionError):
        k.kernel_hessian(x, y, params)

    x = np.array([[1.], [2.]])
    y = np.array([[2., 4.], [3., 2.]])
    params = np.array([0., 0.])

    with pytest.raises(AssertionError):
        k.kernel_hessian(x, y, params)

    x = np.array([[1.], [2.]])
    y = np.array([[[2.], [4.]], [[3.], [2.]]])
    params = np.array([0., 0.])

    with pytest.raises(AssertionError):
        k.kernel_hessian(x, y, params)

    x = np.array([[2., 4.], [3., 2.]])
    y = np.array([[1.], [2.]])
    params = np.array([0., 0.])

    with pytest.raises(AssertionError):
        k.kernel_hessian(x, y, params)

    x = np.array([[[2.], [4.]], [[3.], [2.]]])
    y = np.array([[1.], [2.]])
    params = np.array([0., 0.])

    with pytest.raises(AssertionError):
        k.kernel_hessian(x, y, params)

def test_matern_5_2_inputderiv():
    "test input derivative method of Matern 5/2 kernel"

    k = Matern52()

    dx = 1.e-6

    x = np.array([[1.], [2.]])
    y = np.array([[2.], [3.]])
    params = np.array([0., 0.])

    deriv = np.zeros((1, 2, 2))

    r = np.array([[1., 2.], [0., 1.]])

    deriv[0] = (-5./3.*r*(1.+np.sqrt(5.)*r)*np.exp(-np.sqrt(5.)*r)
                *np.array([[-1., -1.], [0., -1.]]))
    deriv_fd = np.zeros((1, 2, 2))
    deriv_fd[0] = (k.kernel_f(x + dx, y, params) -
                   k.kernel_f(x - dx, y, params))/dx/2.

    assert_allclose(k.kernel_inputderiv(x, y, params), deriv)
    assert_allclose(k.kernel_inputderiv(x, y, params), deriv_fd,
                    rtol = 1.e-5, atol = 1.e-8)

    x = np.array([[1., 2.], [2., 3.]])
    y = np.array([[2., 4.], [3., 1.]])
    params = np.array([0., 0., 0.])

    deriv = np.zeros((2, 2, 2))

    r = np.array([[np.sqrt(5.), np.sqrt(5.)], [1., np.sqrt(5.)]])

    deriv[0] = (-5./3.*(1.+np.sqrt(5.)*r)*np.exp(-np.sqrt(5.)*r)*
                np.array([[-1., -2.], [0., -1.]]))
    deriv[1] = (-5./3.*(1.+np.sqrt(5.)*r)*np.exp(-np.sqrt(5.)*r)*
                np.array([[-2., 1.], [-1., 2.]]))
    deriv_fd = np.zeros((2, 2, 2))
    deriv_fd[0] = (k.kernel_f(x + np.array([[dx, 0.], [dx, 0.]]), y, params) -
                   k.kernel_f(x - np.array([[dx, 0.], [dx, 0.]]), y, params))/dx/2.
    deriv_fd[1] = (k.kernel_f(x + np.array([[0., dx], [0., dx]]), y, params) -
                   k.kernel_f(x - np.array([[0., dx], [0., dx]]), y, params))/dx/2.

    assert_allclose(k.kernel_inputderiv(x, y, params), deriv)
    assert_allclose(k.kernel_inputderiv(x, y, params), deriv_fd, rtol = 1.e-5)

    x = np.array([[1., 2.], [2., 3.]])
    y = np.array([[2., 4.], [3., 1.]])
    params = np.array([np.log(2.), np.log(4.), np.log(2.)])

    deriv = np.zeros((2, 2, 2))

    r = np.array([[np.sqrt(1.*2.+4.*4.), np.sqrt(4.*2.+1.*4.)],
                  [np.sqrt(1.*4.), np.sqrt(1.*2.+4.*4.)]])

    deriv[0] = (-np.exp(2.)*np.exp(2.)*5./3.*(1.+np.sqrt(5.)*r)*
                np.exp(-np.sqrt(5.)*r)*np.array([[-1., -2.], [0., -1.]]))
    deriv[1] = (-np.exp(2.)*np.exp(4.)*5./3.*(1.+np.sqrt(5.)*r)*
                np.exp(-np.sqrt(5.)*r)*np.exp(-0.5*r**2)*np.array([[-2., 1.], [-1., 2.]]))
    deriv_fd = np.zeros((2, 2, 2))
    deriv_fd[0] = (k.kernel_f(x + np.array([[dx, 0.], [dx, 0.]]), y, params) -
                   k.kernel_f(x - np.array([[dx, 0.], [dx, 0.]]), y, params))/dx/2.
    deriv_fd[1] = (k.kernel_f(x + np.array([[0., dx], [0., dx]]), y, params) -
                   k.kernel_f(x - np.array([[0., dx], [0., dx]]), y, params))/dx/2.

    #assert_allclose(k.kernel_inputderiv(x, y, params), deriv)
    assert_allclose(k.kernel_inputderiv(x, y, params), deriv_fd, rtol = 1.e-5)

    x = np.array([1., 2.])
    y = np.array([2., 3.])
    params = np.array([0., 0.])

    deriv = np.zeros((1, 2, 2))

    r = np.array([[1., 2.], [0., 1.]])

    deriv[0] = (-5./3.*r*(1.+np.sqrt(5.)*r)*
                np.exp(-np.sqrt(5.)*r)*np.array([[-1., -1.], [0., -1.]]))
    deriv_fd = np.zeros((1, 2, 2))
    deriv_fd[0] = (k.kernel_f(x + dx, y, params) -
                   k.kernel_f(x - dx, y, params))/dx/2.

    assert_allclose(k.kernel_inputderiv(x, y, params), deriv)
    assert_allclose(k.kernel_inputderiv(x, y, params), deriv_fd,
                    rtol = 1.e-5, atol = 1.e-8)

def test_matern_5_2_inputderiv_failures():
    "test situations where input derivative should fail"

    k = Matern52()

    x = np.array([[1.], [2.]])
    y = np.array([[2.], [3.]])
    params = np.array([0.])

    with pytest.raises(AssertionError):
       k.kernel_inputderiv(x, y, params)

    params = np.array([[0., 0.], [0., 0.]])

    with pytest.raises(AssertionError):
       k.kernel_inputderiv(x, y, params)

    x = np.array([[1.], [2.]])
    y = np.array([[2., 4.], [3., 2.]])
    params = np.array([0., 0.])

    with pytest.raises(AssertionError):
       k.kernel_inputderiv(x, y, params)

    x = np.array([[1.], [2.]])
    y = np.array([[[2.], [4.]], [[3.], [2.]]])
    params = np.array([0., 0.])

    with pytest.raises(AssertionError):
       k.kernel_inputderiv(x, y, params)

    x = np.array([[2., 4.], [3., 2.]])
    y = np.array([[1.], [2.]])
    params = np.array([0., 0.])

    with pytest.raises(AssertionError):
       k.kernel_inputderiv(x, y, params)

    x = np.array([[[2.], [4.]], [[3.], [2.]]])
    y = np.array([[1.], [2.]])
    params = np.array([0., 0.])

    with pytest.raises(AssertionError):
       k.kernel_inputderiv(x, y, params)

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