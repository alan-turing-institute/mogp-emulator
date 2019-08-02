import numpy as np
import pytest
from numpy.testing import assert_allclose
from ..MCMC import MH_proposal, MCMC_step, autothin_samples

def test_MH_proposal():
    "test the Metropolis-Hastings proposal distribution"
    
    np.random.seed(438)
    
    current_param = np.zeros(2)
    cov = np.eye(2)
    
    assert_allclose(MH_proposal(current_param, cov), np.array([-0.392811129690724 ,  0.4478392215027822]))
    
    with pytest.raises(AssertionError):
        MH_proposal(np.zeros((2,2)), cov)
        
    with pytest.raises(AssertionError):
        MH_proposal(np.zeros(2), np.ones((2, 2, 2)))
        
    with pytest.raises(AssertionError):
        MH_proposal(np.zeros(2), np.ones((2, 3)))
        
    with pytest.raises(AssertionError):
        MH_proposal(np.zeros(3), np.ones((2, 2)))
        
    with pytest.raises(AssertionError):
        MH_proposal(np.zeros(2), -1.*np.eye(2))

def test_MCMC_step():
    "test the MCMC step routine"
    
    def loglikelihood(x):
        return 1.
        
    current_param = np.zeros(2)
    cov = np.eye(2)
    
    next_point_expected = np.array([-0.392811129690724 ,  0.4478392215027822])
    
    np.random.seed(438)
    
    next_point, accept = MCMC_step(loglikelihood, current_param, cov)
    
    assert_allclose(next_point, next_point_expected)
    assert accept
    
    def loglikelihood(x):
        return 100.*np.sum(x)

    np.random.seed(438)
    
    next_point, accept = MCMC_step(loglikelihood, current_param, cov, 1.)
    
    assert_allclose(next_point, next_point_expected)
    assert accept
    
    np.random.seed(438)
    
    next_point, accept = MCMC_step(loglikelihood, current_param, cov, -1.)
    
    assert_allclose(next_point, next_point_expected)
    assert not accept
    
    with pytest.raises(AssertionError):
        MCMC_step(loglikelihood, current_param, cov, 0.)
        
    def loglikelihood(x, y):
        return -1.
    
    with pytest.raises(AssertionError):
        MCMC_step(loglikelihood, current_param, cov)
        
    with pytest.raises(AssertionError):
        MCMC_step(2., current_param, cov)
        

def test_autothin_samples():
    "test the autothinning routine"
    
    a = np.zeros(100)
    a[45:55] = 1.
    
    assert autothin_samples(a) == 7
    
    a = np.zeros((100, 2))
    a[45:55, 0] = 1.
    a[40:60, 1] = 1.
    
    assert autothin_samples(a) == 11
    
    a = np.zeros((2, 2, 2))
    
    with pytest.raises(AssertionError):
        autothin_samples(a)
    

