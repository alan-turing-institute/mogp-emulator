import numpy as np
from numpy.testing import assert_allclose
import scipy.stats
import pytest
from inspect import signature
from ..ExperimentalDesign import ExperimentalDesign, MonteCarloDesign

def test_ExperimentalDesign_init():
    "test the init method of ExperimentalDesign"
    
    ed = ExperimentalDesign(3)
    assert ed.n_parameters == 3
    assert len(ed.distributions) == 3
    for dist in ed.distributions:
        assert callable(dist)
        assert len(signature(dist).parameters) == 1
    
    ed = ExperimentalDesign(3.)
    assert ed.n_parameters == 3
    assert len(ed.distributions) == 3
    for dist in ed.distributions:
        assert callable(dist)
        assert len(signature(dist).parameters) == 1
    
    ed = ExperimentalDesign(3, (0., 1.))
    assert ed.n_parameters == 3
    assert len(ed.distributions) == 3
    for dist in ed.distributions:
        assert callable(dist)
        assert len(signature(dist).parameters) == 1
    
    ed = ExperimentalDesign(3, scipy.stats.uniform(loc = 0., scale = 1.).ppf)
    assert ed.n_parameters == 3
    assert len(ed.distributions) == 3
    for dist in ed.distributions:
        assert callable(dist)
        assert len(signature(dist).parameters) == 1

    ed = ExperimentalDesign(2, [(0., 5.), (0., 10.)])
    assert ed.n_parameters == 2
    assert len(ed.distributions) == 2
    for dist in ed.distributions:
        assert callable(dist)
        assert len(signature(dist).parameters) == 1
    
    ed = ExperimentalDesign([(0., 1.), (0., 5.), (0., 10.)])
    assert ed.n_parameters == 3
    assert len(ed.distributions) == 3
    for dist in ed.distributions:
        assert callable(dist)
        assert len(signature(dist).parameters) == 1
    
    ed = ExperimentalDesign([scipy.stats.uniform(loc = 0., scale = 1.).ppf,
                             scipy.stats.uniform(loc = 0., scale = 5.).ppf,
                             scipy.stats.uniform(loc = 0., scale = 10.).ppf])
    assert ed.n_parameters == 3
    assert len(ed.distributions) == 3
    for dist in ed.distributions:
        assert callable(dist)
        assert len(signature(dist).parameters) == 1
    
    ed = ExperimentalDesign([(0., 1.), scipy.stats.uniform(loc = 0., scale = 5.).ppf, (0., 10.)])
    assert ed.n_parameters == 3
    assert len(ed.distributions) == 3
    for dist in ed.distributions:
        assert callable(dist)
        assert len(signature(dist).parameters) == 1
    
def test_ExperimentalDesign_init_failures():
    "test situations where init for ExperimentalDesign should fail"
    
    with pytest.raises(ValueError):
        ed = ExperimentalDesign(-1)

    with pytest.raises(ValueError):
        ed = ExperimentalDesign(4, (0., 2., 2.))

    with pytest.raises(TypeError):
        ed = ExperimentalDesign(3, (0., 2., 2.))
    
    with pytest.raises(ValueError):
        ed = ExperimentalDesign(3, scipy.stats.uniform.ppf)
        
    with pytest.raises(ValueError):
        ed = ExperimentalDesign(2, [(0., 1.), (0., 5.), (0., 10.)])
        
    with pytest.raises(ValueError):
        ed = ExperimentalDesign([(1., 0.), (0., 5.), (0., 10.)])
        
    with pytest.raises(ValueError):
        ed = ExperimentalDesign(2, [(0., 1., 2), (0., 5.)])
        
    with pytest.raises(ValueError):
        ed = ExperimentalDesign([scipy.stats.uniform.ppf, (0., 5.)])
        
    
def test_ExperimentalDesign_get_n_parameters():
    "test get_n_samples method of ExperimentalDesign"
    
    ed = ExperimentalDesign(3)
    assert ed.get_n_parameters() == 3
    
def test_ExperimentalDesign_get_method():
    "test get_method method of ExperimentalDesign"
    
    ed = ExperimentalDesign(3)
    with pytest.raises(NotImplementedError):
        ed.get_method()
        
def test_ExperimentalDesign_sample():
    "test draw_sample method of ExperimentalDesign"
    
    ed = ExperimentalDesign(3)
    with pytest.raises(NotImplementedError):
        ed.sample(3)
    
def test_ExperimentalDesign_str():
    "test string method of ExperimentalDesign"
    
    ed = ExperimentalDesign(3)
    assert str(ed) == "Experimental Design with 3 parameters"

def test_MonteCarloDesign_init():
    "test that init method correctly sets method for Monte Carlo"
    
    ed = MonteCarloDesign(3)
    assert ed.get_method() == "Monte Carlo"
    assert str(ed) == "Monte Carlo Experimental Design with 3 parameters"
    
def test_MonteCarloDesign_sample():
    "test method to draw samples from a Monte Carlo design"
    
    ed = MonteCarloDesign(3)
    np.random.seed(73859)
    sample = ed.sample(4)
    sample_expected = np.array([[0.4587231420553423, 0.8712779046673758, 0.9592574641762792],
                                [0.6863958521075596, 0.1642774978932529, 0.5102796498909351],
                                [0.0945231904820333, 0.7578914914795801, 0.4338919363034113],
                                [0.8612216997446789, 0.0194411385272029, 0.0765283763808666]])
    assert_allclose(sample, sample_expected)
    
    ed = MonteCarloDesign([scipy.stats.uniform(loc = 0., scale = 5.).ppf, scipy.stats.uniform(loc = 0., scale = 2.).ppf])
    sample = ed.sample(3)
    sample_expected = np.array([[1.4976951596996646, 1.9820310379261286],
                                [4.841151376332333 , 0.9447917510829495],
                                [3.8191224939802453, 0.6532873534667354]])
    assert_allclose(sample, sample_expected)
    
def test_MonteCarloDesign_sample_failures():
    "test situation where Monte Carlo Design should raise an exception"
    
    with pytest.raises(AssertionError):
        ed = MonteCarloDesign(3)
        sample = ed.sample(-2)
        
    f = lambda x: np.nan
    
    with pytest.raises(AssertionError):
        ed = MonteCarloDesign(2, f)
        sample = ed.sample(2)