import numpy as np
import pytest
from numpy.testing import assert_allclose
from ..Kernel import calc_r, squared_exponential, matern_5_2

def test_calc_r():
    "test function for calc_r function for kernels"
    
    x = np.array([[1.], [2.]])
    y = np.array([[2.], [3.]])
    params = np.array([0., 0.])
    
    assert_allclose(calc_r(x, y, params), np.array([[1., 2.], [0., 1.]]))
    
    x = np.array([[1., 2.], [2., 3.]])
    y = np.array([[2., 4.], [3., 1.]])
    params = np.array([0., 0., 0.])
    
    assert_allclose(calc_r(x, y, params), np.array([[np.sqrt(5.), np.sqrt(5.)], [1., np.sqrt(5.)]]))
    
    x = np.array([[1., 2.], [2., 3.]])
    y = np.array([[2., 4.], [3., 1.]])
    params = np.array([np.log(2.), np.log(4.), 0.])
    
    assert_allclose(calc_r(x, y, params),
                    np.array([[np.sqrt(1.*2.+4.*4.), np.sqrt(4.*2.+1.*4.)], [np.sqrt(1.*4.), np.sqrt(1.*2.+4.*4.)]]))
                    
    x = np.array([1., 2.])
    y = np.array([2., 3.])
    params = np.array([0., 0.])
    
    assert_allclose(calc_r(x, y, params), np.array([[1., 2.], [0., 1.]]))
    
    
def test_calc_r_failures():
    "test scenarios where calc_r should raise an exception"
    
    x = np.array([[1.], [2.]])
    y = np.array([[2.], [3.]])
    params = np.array([0.])
    
    with pytest.raises(AssertionError):
        calc_r(x, y, params)
    
    params = np.array([[0., 0.], [0., 0.]])
    
    with pytest.raises(AssertionError):
        calc_r(x, y, params)
        
    x = np.array([[1.], [2.]])
    y = np.array([[2., 4.], [3., 2.]])
    params = np.array([0., 0.])
    
    with pytest.raises(AssertionError):
        calc_r(x, y, params)
        
    x = np.array([[1.], [2.]])
    y = np.array([[[2.], [4.]], [[3.], [2.]]])
    params = np.array([0., 0.])
    
    with pytest.raises(AssertionError):
        calc_r(x, y, params)
        
    x = np.array([[2., 4.], [3., 2.]])
    y = np.array([[1.], [2.]])
    params = np.array([0., 0.])
    
    with pytest.raises(AssertionError):
        calc_r(x, y, params)
        
    x = np.array([[[2.], [4.]], [[3.], [2.]]])
    y = np.array([[1.], [2.]])
    params = np.array([0., 0.])
    
    with pytest.raises(AssertionError):
        calc_r(x, y, params)
        
def test_squared_exponential():
    "test squared exponential covariance kernel"
    
    x = np.array([[1.], [2.]])
    y = np.array([[2.], [3.]])
    params = np.array([0., 0.])
    
    assert_allclose(squared_exponential(x, y, params), np.exp(-0.5*np.array([[1., 2.], [0., 1.]])**2))
    
    x = np.array([[1., 2.], [2., 3.]])
    y = np.array([[2., 4.], [3., 1.]])
    params = np.array([0., 0., 0.])
    
    assert_allclose(squared_exponential(x, y, params), np.exp(-0.5*np.array([[np.sqrt(5.), np.sqrt(5.)], [1., np.sqrt(5.)]])**2))
    
    x = np.array([[1., 2.], [2., 3.]])
    y = np.array([[2., 4.], [3., 1.]])
    params = np.array([np.log(2.), np.log(4.), np.log(2.)])
    
    assert_allclose(squared_exponential(x, y, params),
                    2.*np.exp(-0.5*np.array([[np.sqrt(1.*2.+4.*4.), np.sqrt(4.*2.+1.*4.)], [np.sqrt(1.*4.), np.sqrt(1.*2.+4.*4.)]])**2))
                    
    x = np.array([1., 2.])
    y = np.array([2., 3.])
    params = np.array([0., 0.])
    
    assert_allclose(squared_exponential(x, y, params), np.exp(-0.5*np.array([[1., 2.], [0., 1.]])**2))

def test_squared_exponential_failures():
    "test scenarios where squared_exponential should raise an exception"
    
    x = np.array([[1.], [2.]])
    y = np.array([[2.], [3.]])
    params = np.array([0.])
    
    with pytest.raises(AssertionError):
        squared_exponential(x, y, params)
    
    params = np.array([[0., 0.], [0., 0.]])
    
    with pytest.raises(AssertionError):
        squared_exponential(x, y, params)
    
def test_matern_5_2():
    "test matern 5/2 covariance kernel"
    
    x = np.array([[1.], [2.]])
    y = np.array([[2.], [3.]])
    params = np.array([0., 0.])
    
    D = np.array([[1., 2.], [0., 1.]])
    
    assert_allclose(matern_5_2(x, y, params), (1.+np.sqrt(5.)*D+5./3.*D**2)*np.exp(-np.sqrt(5.)*D))
    
    x = np.array([[1., 2.], [2., 3.]])
    y = np.array([[2., 4.], [3., 1.]])
    params = np.array([0., 0., 0.])
    
    D = np.array([[np.sqrt(5.), np.sqrt(5.)], [1., np.sqrt(5.)]])
    
    assert_allclose(matern_5_2(x, y, params), (1.+np.sqrt(5.)*D+5./3.*D**2)*np.exp(-np.sqrt(5.)*D))
    
    x = np.array([[1., 2.], [2., 3.]])
    y = np.array([[2., 4.], [3., 1.]])
    params = np.array([np.log(2.), np.log(4.), np.log(2.)])
    
    D = np.array([[np.sqrt(1.*2.+4.*4.), np.sqrt(4.*2.+1.*4.)], [np.sqrt(1.*4.), np.sqrt(1.*2.+4.*4.)]])
    
    assert_allclose(matern_5_2(x, y, params),
                    2.*(1.+np.sqrt(5.)*D+5./3.*D**2)*np.exp(-np.sqrt(5.)*D))
                    
    x = np.array([1., 2.])
    y = np.array([2., 3.])
    params = np.array([0., 0.])
    
    D = np.array([[1., 2.], [0., 1.]])
    
    assert_allclose(matern_5_2(x, y, params), (1.+np.sqrt(5.)*D + 5./3.*D**2)*np.exp(-np.sqrt(5.)*D))
    
def test_matern_5_2_failures():
    "test scenarios where matern_5_2 should raise an exception"
    
    x = np.array([[1.], [2.]])
    y = np.array([[2.], [3.]])
    params = np.array([0.])
    
    with pytest.raises(AssertionError):
        matern_5_2(x, y, params)
    
    params = np.array([[0., 0.], [0., 0.]])
    
    with pytest.raises(AssertionError):
        matern_5_2(x, y, params)