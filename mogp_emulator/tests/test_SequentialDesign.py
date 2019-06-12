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
    assert sd.n_targets == 1
    assert sd.n_samples == None
    assert sd.n_init == 10
    assert sd.n_cand == 50
    assert_allclose(sd.nugget, 1.)
    assert sd.current_iteration == 0
    assert not sd.initialized
    assert sd.inputs == None
    assert sd.targets == None
    assert sd.candidates == None
    
    sd = SequentialDesign(ed, f, n_targets = 5, n_samples = 100, n_init = 20, n_cand = 100, nugget = 0.1)
    assert type(sd.base_design).__name__ == 'LatinHypercubeDesign'
    assert callable(sd.f)
    assert len(signature(sd.f).parameters) == 1
    assert sd.n_targets == 5
    assert sd.n_samples == 100
    assert sd.n_init == 20
    assert sd.n_cand == 100
    assert_allclose(sd.nugget, 0.1)
    assert sd.current_iteration == 0
    assert not sd.initialized
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
        sd = SequentialDesign(ed, f, n_targets = -1)
           
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
        
def test_SequentialDesign_get_n_targets():
    "test the get_n_targets method"
    
    ed = LatinHypercubeDesign(3)
    
    def f(x):
        return np.array([1.])
        
    sd = SequentialDesign(ed, f, n_targets = 3)
    assert sd.get_n_targets() == 3

def test_SequentialDesign_get_n_parameters():
    "test the get_n_parameters method"
    
    ed = LatinHypercubeDesign(3)
    
    def f(x):
        return np.array([1.])
        
    sd = SequentialDesign(ed, f)
    assert sd.get_n_parameters() == 3
    
def test_SequentialDesign_get_n_init():
    "test the get_n_init method"
    
    ed = LatinHypercubeDesign(3)
    
    def f(x):
        return np.array([1.])
        
    sd = SequentialDesign(ed, f, n_init = 20)
    assert sd.get_n_init() == 20
    
def test_SequentialDesign_get_n_samples():
    "test the get_n_cand method"
    
    ed = LatinHypercubeDesign(3)
    
    def f(x):
        return np.array([1.])
        
    sd = SequentialDesign(ed, f, n_samples = 20)
    assert sd.get_n_samples() == 20
    
def test_SequentialDesign_get_n_cand():
    "test the get_n_cand method"
    
    ed = LatinHypercubeDesign(3)
    
    def f(x):
        return np.array([1.])
        
    sd = SequentialDesign(ed, f, n_cand = 20)
    assert sd.get_n_cand() == 20
    
def test_SequentialDesign_get_nugget():
    "test the get_nugget method"
    
    ed = LatinHypercubeDesign(3)
    
    def f(x):
        return np.array([1.])
        
    sd = SequentialDesign(ed, f, nugget = 0.1)
    assert_allclose(sd.get_nugget(), 0.1)
    
def test_SequentialDesign_get_current_iteration():
    "test the get_current_iteration method"
    
    ed = LatinHypercubeDesign(3)
    
    def f(x):
        return np.array([1.])
        
    sd = SequentialDesign(ed, f)
    assert sd.get_current_iteration() == 0
    
    sd.current_iteration = 20
    assert sd.get_current_iteration() == 20
    
def test_SequentialDesign_get_inputs():
    "test the get_inputs method"
    
    ed = LatinHypercubeDesign(3)
    
    def f(x):
        return np.array([1.])
        
    sd = SequentialDesign(ed, f)
    assert sd.get_inputs() == None
    
    sd.inputs = np.zeros((3, 4))
    assert_allclose(sd.get_inputs(), np.zeros((3, 4)))
    
def test_SequentialDesign_get_targets():
    "test the get_targets method"
    
    ed = LatinHypercubeDesign(3)
    
    def f(x):
        return np.array([1.])
        
    sd = SequentialDesign(ed, f)
    assert sd.get_targets() == None
    
    sd.targets = np.zeros((3, 4))
    assert_allclose(sd.get_targets(), np.zeros((3, 4)))
    
def test_SequentialDesign_get_candidates():
    "test the get_candidates method"
    
    ed = LatinHypercubeDesign(3)
    
    def f(x):
        return np.array([1.])
        
    sd = SequentialDesign(ed, f)
    assert sd.get_candidates() == None
    
    sd.candidates = np.zeros((3, 4))
    assert_allclose(sd.get_candidates(), np.zeros((3, 4)))
    
def test_SequentialDesign_get_base_design():
    "test the get_base_design method"
    
    ed = LatinHypercubeDesign(3)
    
    def f(x):
        return np.array([1.])
        
    sd = SequentialDesign(ed, f)
    assert sd.get_base_design() == "LatinHypercubeDesign"
    
def test_SequentialDesign_generate_initial_design():
    "test the generate_initial_design method"
    
    np.random.seed(74632)
    
    ed = LatinHypercubeDesign(3)
    
    def f(x):
        return np.array([1.])
        
    sd = SequentialDesign(ed, f, n_init = 4)
    initial_design_expected = np.array([[0.9660431763890672, 0.2080126306969736, 0.2576380063570568],
                                        [0.0684779421445063, 0.9308367720360009, 0.1428493015158686],
                                        [0.6345029195085983, 0.6651343562344474, 0.8827198350687029],
                                        [0.4531112960399023, 0.3977273628763245, 0.5867585643640021]])
    initial_design_actual = sd.generate_initial_design()
    assert_allclose(initial_design_actual, initial_design_expected)
    assert_allclose(sd.inputs, initial_design_expected)
    assert sd.current_iteration == 4
    
def test_SequentialDesign_set_initial_targets():
    "test the set_initial_targets method"
    
    np.random.seed(74632)
    
    ed = LatinHypercubeDesign(3)
    
    def f(x):
        return np.sum(x)
        
    sd = SequentialDesign(ed, f, n_init = 4)
    initial_design_expected = np.array([[0.9660431763890672, 0.2080126306969736, 0.2576380063570568],
                                        [0.0684779421445063, 0.9308367720360009, 0.1428493015158686],
                                        [0.6345029195085983, 0.6651343562344474, 0.8827198350687029],
                                        [0.4531112960399023, 0.3977273628763245, 0.5867585643640021]])
    targets_expected = np.array([np.sum(i) for i in initial_design_expected])
    sd.generate_initial_design()
    sd.set_initial_targets(targets_expected)
    assert_allclose(sd.targets, np.reshape(targets_expected, (1,4)))
    assert sd.initialized
    
    sd = SequentialDesign(ed, f, n_init = 4)
    targets_expected = np.array([[np.sum(i) for i in initial_design_expected]])
    sd.generate_initial_design()
    sd.set_initial_targets(targets_expected)
    assert_allclose(sd.targets, targets_expected)
    assert sd.initialized
    
    sd = SequentialDesign(ed, f, n_init = 4)
    with pytest.raises(ValueError):
        sd.set_initial_targets(targets_expected)
        
    sd = SequentialDesign(ed, f, n_init = 4)
    sd.generate_initial_design()
    with pytest.raises(AssertionError):
        sd.set_initial_targets(np.zeros((2, 4)))
        
    sd = SequentialDesign(ed, f, n_init = 4)
    sd.inputs = np.zeros((5,2))
    with pytest.raises(AssertionError):
        sd.set_initial_targets(targets_expected)
        
def test_SequentialDesign_run_init_design():
    "test method to run initial design"
    
    np.random.seed(74632)
    
    ed = LatinHypercubeDesign(3)
    
    def f(x):
        return np.sum(x)
        
    sd = SequentialDesign(ed, f, n_init = 4)
    initial_design_expected = np.array([[0.9660431763890672, 0.2080126306969736, 0.2576380063570568],
                                        [0.0684779421445063, 0.9308367720360009, 0.1428493015158686],
                                        [0.6345029195085983, 0.6651343562344474, 0.8827198350687029],
                                        [0.4531112960399023, 0.3977273628763245, 0.5867585643640021]])
    targets_expected = np.array([[np.sum(i) for i in initial_design_expected]])
    sd.run_init_design()
    assert_allclose(sd.inputs, initial_design_expected)
    assert_allclose(sd.targets, targets_expected)
    assert sd.initialized
    assert sd.current_iteration == 4
    
def test_SequentialDesign_generate_candidates():
    "test the _generate_candidates method"
    
    np.random.seed(74632)
    
    ed = LatinHypercubeDesign(3)
    
    def f(x):
        return np.sum(x)
        
    sd = SequentialDesign(ed, f, n_cand = 4)
    candidates_expected = np.array([[0.9660431763890672, 0.2080126306969736, 0.2576380063570568],
                                    [0.0684779421445063, 0.9308367720360009, 0.1428493015158686],
                                    [0.6345029195085983, 0.6651343562344474, 0.8827198350687029],
                                    [0.4531112960399023, 0.3977273628763245, 0.5867585643640021]])
    sd._generate_candidates()
    assert_allclose(sd.candidates, candidates_expected)
    
def test_SequentialDesign_eval_metric():
    "test the _eval_metric method"
    
    ed = LatinHypercubeDesign(3)
    
    def f(x):
        return np.sum(x)
        
    sd = SequentialDesign(ed, f)
    
    with pytest.raises(NotImplementedError):
        sd._eval_metric()
        
def test_SequentialDesign_get_next_point():
    "test the get_next_point method"
    
    np.random.seed(74632)
    
    ed = LatinHypercubeDesign(3)
    
    def f(x):
        return np.sum(x)
        
    def tmp_eval_metric(self):
        return 0
        
    sd = SequentialDesign(ed, f, n_init = 4, n_cand = 4)
    
    sd._eval_metric = types.MethodType(tmp_eval_metric, sd)
    sd.run_init_design()
    next_point = sd.get_next_point()
    
    next_point_expected = np.array([3.9602716910300234e-01, 4.3469440375712098e-02, 9.3294684823072194e-01])
    inputs_expected = np.array([[0.9660431763890672, 0.2080126306969736, 0.2576380063570568],
                                [0.0684779421445063, 0.9308367720360009, 0.1428493015158686],
                                [0.6345029195085983, 0.6651343562344474, 0.8827198350687029],
                                [0.4531112960399023, 0.3977273628763245, 0.5867585643640021],
                                [3.9602716910300234e-01, 4.3469440375712098e-02, 9.3294684823072194e-01]])
    candidates_expected = np.array([[3.9602716910300234e-01, 4.3469440375712098e-02, 9.3294684823072194e-01],
                                    [6.7353171430888004e-01, 3.2967090024809237e-01, 4.7561098969781079e-01],
                                    [9.9391552790043791e-01, 5.0015522522360367e-01, 2.4850731127782361e-01],
                                    [6.6098271480205528e-04, 9.7616245971529891e-01, 7.4815518971467765e-01]])
    assert_allclose(next_point, next_point_expected)
    assert_allclose(sd.inputs, inputs_expected)
    assert_allclose(sd.candidates, candidates_expected)
    
def test_SequentialDesign_set_next_target():
    "test the set_next_target method"
    
    np.random.seed(74632)
    
    ed = LatinHypercubeDesign(3)
    
    def f(x):
        return np.sum(x)
        
    def tmp_eval_metric(self):
        return 0
        
    sd = SequentialDesign(ed, f, n_init = 4, n_cand = 4)
    
    sd._eval_metric = types.MethodType(tmp_eval_metric, sd)
    sd.run_init_design()
    next_point = sd.get_next_point()
    new_target = np.reshape(np.array(np.sum(next_point)), (1,))
    
    inputs_expected = np.array([[0.9660431763890672, 0.2080126306969736, 0.2576380063570568],
                                [0.0684779421445063, 0.9308367720360009, 0.1428493015158686],
                                [0.6345029195085983, 0.6651343562344474, 0.8827198350687029],
                                [0.4531112960399023, 0.3977273628763245, 0.5867585643640021],
                                [3.9602716910300234e-01, 4.3469440375712098e-02, 9.3294684823072194e-01]])
    targets_expected = np.array([[np.sum(i) for i in inputs_expected]])
    
    sd.set_next_target(new_target)
    assert_allclose(sd.targets, targets_expected)
    assert sd.current_iteration == 5
    
        
    sd = SequentialDesign(ed, f, n_init = 4, n_cand = 4)
    
    sd._eval_metric = types.MethodType(tmp_eval_metric, sd)
    sd.run_init_design()
    next_point = sd.get_next_point()
    
    with pytest.raises(AssertionError):
        sd.set_next_target(np.zeros((2,)))
        
        
    sd = SequentialDesign(ed, f, n_init = 4, n_cand = 4)
    
    sd.generate_initial_design()
    sd._eval_metric = types.MethodType(tmp_eval_metric, sd)
    next_point = sd.get_next_point()
    next_target = np.sum(next_point)
    
    with pytest.raises(ValueError):
        sd.set_next_target(next_target)
        
    sd = SequentialDesign(ed, f, n_init = 4, n_cand = 4)
    
    with pytest.raises(ValueError):
        sd.set_next_target(next_target)
        
    sd = SequentialDesign(ed, f, n_init = 4, n_cand = 4)
    sd._eval_metric = types.MethodType(tmp_eval_metric, sd)
    sd.run_init_design()
    
    with pytest.raises(AssertionError):
        sd.set_next_target(next_target)
        
    sd = SequentialDesign(ed, f, n_init = 4, n_cand = 4)
    sd._eval_metric = types.MethodType(tmp_eval_metric, sd)
    sd.run_init_design()
    sd.get_next_point()
    sd.targets = np.zeros((5,3))
    
    with pytest.raises(AssertionError):
        sd.set_next_target(next_target)