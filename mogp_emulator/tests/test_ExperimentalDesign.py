import numpy
from numpy.testing import assert_allclose
import scipy.stats
import pytest
from inspect import signature
from ..ExperimentalDesign import ExperimentalDesign

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
    
def test_ExperimentalDesign_str():
    "test string method of ExperimentalDesign"
    
    ed = ExperimentalDesign(3)
    assert str(ed) == "Experimental Design with 3 parameters"