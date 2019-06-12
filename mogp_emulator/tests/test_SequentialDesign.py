import numpy as np
from numpy.testing import assert_allclose
import pytest
from inspect import signature
import types
from ..ExperimentalDesign import LatinHypercubeDesign
from ..SequentialDesign import SequentialDesign

def test_SequentialDesign_init():
    "test the init method of ExperimentalDesign"
    
    ed = LatinHypercubeDesign(3)
    
    def f(x):
        return np.array([1.])
        
    sd = SequentialDesign(ed, f)
    assert type(sd.base_design).__name__ == 'LatinHypercubeDesign'
    assert callable(sd.f)
    assert len(signature(sd.f).parameters) == 1
    assert sd.n_samples == None
    assert sd.n_init == 10
    assert sd.n_cand == 50
    assert_allclose(sd.nugget, 1.)
    assert sd.current_iteration == 0ß
    assert not sd.initialized
    assert sd.initial_design == None
    assert sd.inputs == None
    assert sd.targets == None
    assert sd.candidates == None
    
    sd = SequentialDesign(ed, f, n_samples = 100, n_init = 20, n_cand = 100, nugget = 0.1)
    assert type(sd.base_design).__name__ == 'LatinHypercubeDesign'
    assert callable(sd.f)
    assert len(signature(sd.f).parameters) == 1
    assert sd.n_samples == 100
    assert sd.n_init == 20
    assert sd.n_cand == 100
    assert_allclose(sd.nugget, 0.1)
    assert sd.current_iteration == 0
    assert not sd.initialized
    assert sd.initial_design == None
    assert sd.inputs == None
    assert sd.targets == None
    assert sd.candidates == None
    
def test_SequentialDesign_init_failures():
    "test cases where init should fail"
    
    ed = LatinHypercubeDesign(3)
    
    def f(x):
        return np.array([1.])
    
    with pytest.raises(TypeError):
        sd = SequentialDesign(3., f)
        
    with pytest.raises(TypeError):
        sd = SequentialDesign(ed, 3.)
    
    def f2(x, y):
        return np.array([1.])
    
    with pytest.raises(ValueError):
        sd = SequentialDesign(ed, f2)
        
    with pytest.raises(ValueError):
        sd = SequentialDesign(ed, f, n_samples = -1)
        
    with pytest.raises(ValueError):
        sd = SequentialDesign(ed, f, n_init = -1)
    
    with pytest.raises(ValueError):
        sd = SequentialDesign(ed, f, n_samples = 10, n_init = 20)
        
    with pytest.raises(ValueError):
        sd = SequentialDesign(ed, f, n_cand = -1)
        
    with pytest.raises(ValueError):
        sd = SequentialDesign(ed, f, nugget = -1.)