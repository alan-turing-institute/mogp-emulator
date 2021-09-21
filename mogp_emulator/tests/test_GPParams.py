import pytest
import numpy as np
from numpy.testing import assert_allclose
from ..GPParams import GPParams

def test_GPParams_init():
    "Test init method of GPParams"
    
    gpp = GPParams()
    
    assert gpp.n_mean == 0
    assert gpp.n_corr == 1
    assert gpp.n_cov == 1
    assert gpp.n_nugget == 1
    assert gpp.mean_data.shape == (0,)
    assert gpp.data is None

    gpp = GPParams(n_mean=2, n_corr=3, nugget=False, mean_data=np.ones(2), data=np.ones(4))
    
    assert gpp.n_mean == 2
    assert gpp.n_corr == 3
    assert gpp.n_cov == 1
    assert gpp.n_nugget == 0
    assert_allclose(gpp.mean_data, np.ones(2))
    assert_allclose(gpp.data, np.ones(4))

def test_GPParams_init_failures():
    "Test failures of GPParams init method"
    
    with pytest.raises(AssertionError):
        GPParams(n_mean=-1)

    with pytest.raises(AssertionError):
        GPParams(n_corr=-1)
        
    with pytest.raises(AssertionError):
        GPParams(data=np.ones(1))

def test_GPParams_n_params():
    "Test the n_params property of GPParams"
    
    gpp = GPParams()
    
    assert gpp.n_params == 3
    
    gpp = GPParams(n_mean=2, n_corr=3, nugget=False)
    
    assert gpp.n_params == 4

def test_GPParams_mean():
    "Test the mean functionality of GPParams"
    
    gpp = GPParams()
    
    assert gpp.n_mean == 0
    assert len(gpp.mean) == 0
    
    with pytest.raises(AssertionError):
        gpp.mean = [1.]
    
    gpp = GPParams(n_mean=2, data=np.ones(3))
    
    assert gpp.n_mean == 2
    assert gpp.mean_data is None
    
    gpp.mean = np.array([2., 3.])
    
    assert_allclose(gpp.mean, np.array([2., 3.]))
    assert_allclose(gpp.mean_data, np.array([2., 3.]))

    with pytest.raises(AssertionError):
        gpp.mean = []
        
    gpp = GPParams(n_mean=1)
    
    assert gpp.mean is None
    
    gpp.mean = 1.
    
    assert_allclose(gpp.mean, np.ones(1))

def test_GPParams_corr():
    "Test the correlation functionality of GPParams"
    
    gpp = GPParams()
    
    assert gpp.n_corr == 1
    assert gpp.corr is None
    assert gpp.corr_raw is None
    
    with pytest.raises(ValueError):
        gpp.corr_raw = 2.
    
    gpp.data = np.zeros(gpp.n_params)
    assert_allclose(gpp.corr, np.ones(1))
    assert_allclose(gpp.corr_raw, np.zeros(1))

    gpp.corr = 2.
    
    assert_allclose(gpp.corr, np.array([2.]))
    assert_allclose(gpp.corr_raw, -2.*np.log(2.))

    gpp.corr = np.array([3.])
    
    assert_allclose(gpp.corr, np.array([3.]))
    assert_allclose(gpp.corr_raw, -2.*np.log(3.))
    
    gpp.corr_raw = 0.
    
    assert_allclose(gpp.corr, np.ones(1))
    assert_allclose(gpp.corr_raw, np.zeros(1))
    
    gpp.corr_raw = np.array([3.])
    
    assert_allclose(gpp.corr, np.exp(-0.5*np.array([3.])))
    assert_allclose(gpp.corr_raw, 3.)

    with pytest.raises(AssertionError):
        gpp.corr = np.array([2., 3.])
    
    with pytest.raises(AssertionError):
        gpp.corr_raw = np.array([2., 3.])
        
    with pytest.raises(AssertionError):
        gpp.corr = -1.
        
    gpp = GPParams(n_corr=3, data=np.array([2., 3., 2., 0., 0.]))
    
    assert gpp.n_corr == 3
    assert_allclose(gpp.corr, np.exp(-0.5*np.array([2., 3., 2.])))
    assert_allclose(gpp.corr_raw, np.array([2., 3., 2.]))

    gpp.corr = np.ones(3)
    
    assert_allclose(gpp.corr, np.ones(3))
    assert_allclose(gpp.corr_raw, np.zeros(3))
    
    gpp.corr_raw = np.ones(3)
    
    assert_allclose(gpp.corr, np.exp(-0.5*np.ones(3)))
    assert_allclose(gpp.corr_raw, np.ones(3))

    with pytest.raises(AssertionError):
        gpp.corr = np.array([2., 3.])

    with pytest.raises(AssertionError):
        gpp.corr_raw = np.array([2., 3.])
        
    with pytest.raises(AssertionError):
        gpp.corr = np.array([-1., 3., 3.])

def test_GPParams_cov():
    "Test the covariance functionality of GPParams"
    
    gpp = GPParams()
    
    assert gpp.n_cov == 1
    assert gpp.cov is None
    assert gpp.cov_raw is None
    
    with pytest.raises(ValueError):
        gpp.cov_raw = 2.
    
    gpp.data = np.zeros(gpp.n_params)
    assert_allclose(gpp.cov, np.ones(1))
    assert_allclose(gpp.cov_raw, np.zeros(1))

    gpp.cov = 2.
    
    assert_allclose(gpp.cov, np.array([2.]))
    assert_allclose(gpp.cov_raw, np.log(2.))

    gpp.cov = np.array([3.])
    
    assert_allclose(gpp.cov, np.array([3.]))
    assert_allclose(gpp.cov_raw, np.log(3.))
    
    gpp.cov_raw = 0.
    
    assert_allclose(gpp.cov, np.ones(1))
    assert_allclose(gpp.cov_raw, np.zeros(1))
    
    gpp.cov_raw = np.array([3.])
    
    assert_allclose(gpp.cov, np.exp(np.array([3.])))
    assert_allclose(gpp.cov_raw, 3.)

    with pytest.raises(AssertionError):
        gpp.cov = np.array([2., 3.])
    
    with pytest.raises(AssertionError):
        gpp.cov_raw = np.array([2., 3.])
        
    with pytest.raises(AssertionError):
        gpp.cov = -1.
        
    gpp = GPParams(data=np.array([0., 2., 0.]))
    
    assert gpp.n_cov == 1
    assert_allclose(gpp.cov, np.exp(np.array([2.])))
    assert_allclose(gpp.cov_raw, np.array([2.]))

    gpp.cov = np.ones(1)
    
    assert_allclose(gpp.cov, np.ones(1))
    assert_allclose(gpp.cov_raw, np.zeros(1))
    
    gpp.cov = 1.
    
    assert_allclose(gpp.cov, np.ones(1))
    assert_allclose(gpp.cov_raw, np.zeros(1))
    
    gpp.cov_raw = np.ones(1)
    
    assert_allclose(gpp.cov, np.exp(np.ones(1)))
    assert_allclose(gpp.cov_raw, np.ones(1))

    with pytest.raises(AssertionError):
        gpp.cov = np.array([2., 3.])

    with pytest.raises(AssertionError):
        gpp.cov_raw = np.array([2., 3.])
        
    with pytest.raises(AssertionError):
        gpp.cov = np.array([-1.])

def test_GPParams_nugget():
    "Test the covariance functionality of GPParams"
    
    gpp = GPParams()
    
    assert gpp.n_nugget == 1
    assert gpp.nugget is None
    assert gpp.nugget_raw is None
    
    with pytest.raises(ValueError):
        gpp.nugget_raw = 2.
    
    gpp.data = np.zeros(gpp.n_params)
    assert_allclose(gpp.nugget, np.ones(1))
    assert_allclose(gpp.nugget_raw, np.zeros(1))

    gpp.nugget = 2.
    
    assert_allclose(gpp.nugget, np.array([2.]))
    assert_allclose(gpp.nugget_raw, np.log(2.))

    gpp.nugget = np.array([3.])
    
    assert_allclose(gpp.nugget, np.array([3.]))
    assert_allclose(gpp.nugget_raw, np.log(3.))
    
    gpp.nugget_raw = 0.
    
    assert_allclose(gpp.nugget, np.ones(1))
    assert_allclose(gpp.nugget_raw, np.zeros(1))
    
    gpp.nugget_raw = np.array([3.])
    
    assert_allclose(gpp.nugget, np.exp(np.array([3.])))
    assert_allclose(gpp.nugget_raw, 3.)

    with pytest.raises(AssertionError):
        gpp.nugget = np.array([2., 3.])
    
    with pytest.raises(AssertionError):
        gpp.nugget_raw = np.array([2., 3.])
        
    with pytest.raises(AssertionError):
        gpp.nugget = -1.
        
    gpp = GPParams(data=np.array([0., 0., 2.]))
    
    assert gpp.n_nugget == 1
    assert_allclose(gpp.nugget, np.exp(np.array([2.])))
    assert_allclose(gpp.nugget_raw, np.array([2.]))

    gpp = GPParams(nugget=False)
    
    assert gpp.n_nugget == 0
    assert gpp.nugget is None
    
    gpp.data = np.zeros(gpp.n_params)
    assert len(gpp.nugget) == 0
    assert len(gpp.nugget_raw) == 0

def test_GPParams_data():
    "Test the data getter and setter methods"
    
    gpp = GPParams()
    
    assert gpp.get_data() is None
    
    gpp.data = np.zeros(gpp.n_params)
    
    assert_allclose(gpp.get_data(), np.zeros(3))
    
    gpp.set_data(np.ones(3))
    
    assert_allclose(gpp.data, np.ones(3))
    
    with pytest.raises(AssertionError):
        gpp.set_data(np.ones(4))
        
    gpp = GPParams(n_mean=2, n_corr=3, data = np.zeros(5))
    
    assert_allclose(gpp.get_data(), np.zeros(5))
    
    gpp.set_data(np.ones(5))
    
    assert_allclose(gpp.data, np.ones(5))
    
    gpp.set_data(None)
    
    assert gpp.data is None

def test_GPParams_same_shape():
    
    gpp = GPParams()
    
    assert gpp.same_shape(GPParams())
    assert not gpp.same_shape(GPParams(n_mean=1))
    assert not gpp.same_shape(GPParams(n_corr=2))
    assert not gpp.same_shape(GPParams(nugget=False))
    assert gpp.same_shape(np.zeros(3))
    assert not gpp.same_shape(np.zeros(4))

    gpp = GPParams(n_mean=2, n_corr=3, nugget=False)
    
    assert gpp.same_shape(GPParams(n_mean=2, n_corr=3, nugget=False))
    assert not gpp.same_shape(GPParams(n_mean=1, n_corr=3, nugget=False))
    assert not gpp.same_shape(GPParams(n_mean=2, n_corr=2, nugget=False))
    assert not gpp.same_shape(GPParams(n_mean=2, n_corr=3, nugget=True))
    assert gpp.same_shape(np.zeros(4))
    assert not gpp.same_shape(np.zeros(7))
    
    with pytest.raises(ValueError):
        gpp.same_shape([])

def test_GPParams_str():
    "Test the str method of GPParams doesn't raise an error"
    
    str(GPParams())
