"""
Simple demos and sanity checks for the HistoryMatching class.

Provided methods:
    get_y_simulated_1D:
        A toy model that acts as the simulator output for constructing GPEs for 
        1D data.
    
    get_y_simulated_2D:
        A toy model that acts as the simulator output for constructing GPEs for 
        2D data.

    demo_1D:
        Follows the example of 
        http://www.int.washington.edu/talks/WorkShops/int_16_2a/People/Vernon_I/Vernon2.pdf 
        in setting up a simple test model, constructing a gaussian process to 
        emulate it, and ruling out expectations of the GPE based on an 
        historical observation.

    demo_2D:
        As demo_1D, however the toy model is expanded to take two inputs rather
        then 1 extending it to a second dimension. Exists primarily to confirm 
        that the HistoryMatching class is functional with higher-dimensional
        parameter spaces. 

"""

from mogp_emulator.HistoryMatching import HistoryMatching
from mogp_emulator.GaussianProcess import GaussianProcess
import numpy as np
from numpy.testing import assert_allclose
import pytest

def get_y_simulated_1D(x):
    n_points = len(x)
    f = np.zeros(n_points)
    for i in range(n_points):
        f[i] = np.sin(2.*np.pi*x[i] / 50.) 
    return f

def test_sanity_checks():
    "test basic functioning of HistoryMatching"
    
    # Create a gaussian process
    x_training = np.array([
        [0.],
        [10.],
        [20.],
        [30.],
        [43.],
        [50.]
    ])

    y_training = get_y_simulated_1D(x_training)

    gp = GaussianProcess(x_training, y_training)
    np.random.seed(47)
    gp.learn_hyperparameters()

    # Define observation and implausibility threshold
    obs = [-0.8, 0.0004]

    # Coords to predict
    n_rand = 2000
    x_predict_min = -3
    x_predict_max = 53
    x_predict = np.random.rand(n_rand)
    x_predict = np.sort(x_predict, axis=0)
    x_predict *= (x_predict_max - x_predict_min)
    x_predict += x_predict_min
    x_predict = x_predict[:,None]

    coords = x_predict

    expectations = gp.predict(coords)

    # Create History Matching Instance
    print("---TEST INPUTS---")
    print("No Args")
    hm = HistoryMatching()
    hm.status()

    print("Obs Only a - list")
    hm = HistoryMatching(obs=obs)
    hm.status()

    print("Obs only b - single-element list")
    hm = HistoryMatching(obs=[3])
    hm.status()

    print("Obs only c - single-value")
    hm = HistoryMatching(obs=3)
    hm.status()

    print("gp Only")
    hm = HistoryMatching(gp=gp)
    hm.status()

    print("Coords only a - 2d ndarray")
    hm = HistoryMatching(coords=coords)
    hm.status()

    print("Coords only b - 1d ndarray")
    hm = HistoryMatching(coords=np.random.rand(n_rand))
    hm.status()

    print("Coords only c - list")
    hm = HistoryMatching(coords=[a for a in range(n_rand)])
    hm.status()

    print("Expectation only")
    hm = HistoryMatching(expectations=expectations)
    hm.status()

    print("Threshold Only")
    hm = HistoryMatching(threshold=3)
    hm.status()


    print("---TEST ASSIGNMENT---")
    print("Assign gp")
    hm = HistoryMatching(obs)
    hm.status()
    hm.set_gp(gp)
    hm.status()

    print("Assign Obs")
    hm = HistoryMatching(gp)
    hm.status()
    hm.set_obs(obs)
    hm.status()

    print("Assign Coords")
    hm = HistoryMatching()
    hm.status()
    hm.set_coords(coords)
    hm.status()

    print("Assign Expectations")
    hm = HistoryMatching()
    hm.status()
    hm.set_expectations(expectations)
    hm.status()

    print("Assign Threshold")
    hm = HistoryMatching()
    hm.status()
    hm.set_threshold(3.)
    hm.status()


    print("---TEST IMPLAUSABILIY---")
    print("implausibility test a - no vars")
    hm = HistoryMatching(obs=obs, gp=gp, coords=coords)
    I = hm.get_implausibility()

    print("implausibility test b - single value")
    hm = HistoryMatching(obs=obs, gp=gp, coords=coords)
    I = hm.get_implausibility(7.)

    print("implausibility test c - multiple values")
    hm = HistoryMatching(obs=obs, gp=gp, coords=coords)
    I = hm.get_implausibility(7., 8, 109.5)

    print("implausibility test d - single list")
    hm = HistoryMatching(obs=obs, gp=gp, coords=coords)
    var = [a for a in range(2000)]
    I = hm.get_implausibility([a for a in range(2000)])

    print("implausibility test d - multiple lists")
    hm = HistoryMatching(obs=obs, gp=gp, coords=coords)
    I = hm.get_implausibility(var, [v+2 for v in var], [v*7 for v in var])

    print("implausibility test e - single 1D ndarray ")
    hm = HistoryMatching(obs=obs, gp=gp, coords=coords)
    var = np.asarray(var)
    I = hm.get_implausibility(var)

    print("implausibility test f - single 2D ndarray ")
    hm = HistoryMatching(obs=obs, gp=gp, coords=coords)
    var = np.reshape(np.asarray(var), (-1,1))
    I = hm.get_implausibility(var)

def test_HistoryMatching_init():
    "test the init method of HistoryMatching"
    
    hm = HistoryMatching()
    
    assert hm.gp == None
    assert hm.obs == None
    assert hm.coords == None
    assert hm.expectations == None
    assert hm.ndim == None
    assert hm.ncoords == None
    assert_allclose(hm.threshold, 3.)
    assert hm.I == None
    assert hm.NROY == None
    assert hm.RO == None
    
    gp = GaussianProcess(np.reshape(np.linspace(0., 1.), (-1, 1)), np.linspace(0., 1.))
    coords = np.array([[0.2], [0.4]])
    hm = HistoryMatching(gp=gp, obs=1., coords=coords, expectations=None, threshold=5.)
    
    assert hm.gp == gp
    assert_allclose(hm.obs, [1., 0.])
    assert_allclose(hm.coords, coords)
    assert hm.expectations == None
    assert hm.ndim == 1
    assert hm.ncoords == len(coords)
    assert_allclose(hm.threshold, 5.)
    assert hm.I == None
    assert hm.NROY == None
    assert hm.RO == None
    
    expectations = (np.array([1.]), np.array([0.2]), np.array([[0.1]]))
    hm = HistoryMatching(gp=None, obs=[1., 0.1], coords=None, expectations=expectations, threshold=5.)
    
    assert hm.gp == None
    assert_allclose(hm.obs, [1., 0.1])
    assert hm.coords == None
    for a, b in zip(hm.expectations, expectations):
        assert_allclose(a, b)
    assert hm.ndim == None
    assert hm.ncoords == len(expectations[0])
    assert_allclose(hm.threshold, 5.)
    assert hm.I == None
    assert hm.NROY == None
    assert hm.RO == None
    
def test_HistoryMatching_init_failures():
    "check situations where the init method of HistoryMatching fails"
    
    with pytest.raises(ValueError):
        hm = HistoryMatching(obs = [1., 2., 3.])
        
    with pytest.raises(TypeError):
        hm = HistoryMatching(obs = "abc")
        
    with pytest.raises(TypeError):
        hm = HistoryMatching(expectations = (np.array([0.1]), 1., 1.))
        
    with pytest.raises(AssertionError):
        hm = HistoryMatching(threshold = -1.)

def test_HistoryMatching_get_implausibility():
    "test the get_implausibility method of HistoryMatching"
    
    expectations = (np.array([2., 10.]), np.array([0., 0.]), np.array([[1., 2.]]))
    hm = HistoryMatching(obs = [1., 1.], expectations = expectations)
    I = hm.get_implausibility()
    
    assert_allclose(I, [1., 9.])
    assert_allclose(hm.I, [1., 9.])
    
    I = hm.get_implausibility(1.)
    
    assert_allclose(I, [1./np.sqrt(2.), 9./np.sqrt(2.)])
    assert_allclose(hm.I, [1./np.sqrt(2.), 9./np.sqrt(2.)])
    
    I = hm.get_implausibility(1., 2.)
    
    assert_allclose(I, [0.5, 4.5])
    assert_allclose(hm.I, [0.5, 4.5])
    
    gp = GaussianProcess(np.reshape(np.linspace(0., 1.), (-1, 1)), np.linspace(0., 1.))
    np.random.seed(57483)
    gp.learn_hyperparameters()
    coords = np.array([[0.1], [1.]])
    obs = [1., 0.01]
    mean, unc, _ = gp.predict(coords)
    I_exp = np.abs(mean - obs[0])/np.sqrt(unc + obs[1])
    
    hm = HistoryMatching(gp = gp, obs = obs, coords = coords)
    I = hm.get_implausibility()
    
    assert_allclose(I, I_exp)
    assert_allclose(hm.I, I_exp)
    
    with pytest.raises(AssertionError):
        hm.get_implausibility(-1.)
    
    hm = HistoryMatching(gp = gp, obs = obs, coords = coords, expectations = expectations)
    
    with pytest.raises(Exception):
        hm.get_implausibility()
    
def test_HistoryMatching_get_NROY():
    "test the get_NROY method of HistoryMatching"
    
    hm = HistoryMatching(obs = [1., 1.], expectations = (np.array([2., 10.]), np.array([0., 0.]), np.array([[1., 2.]])))
    I = hm.get_implausibility()
    
    NROY = hm.get_NROY()
    
    assert NROY == [0]
    
    hm = HistoryMatching(obs = [1., 0.], expectations = (np.array([2., 10.]), np.array([0., 0.]), np.array([[1., 2.]])))
    
    NROY = hm.get_NROY(1.)
    
    assert NROY == [0]
    
def test_HistoryMatching_get_RO():
    "test the get_RO method of HistoryMatching"
    
    hm = HistoryMatching(obs = [1., 1.], expectations = (np.array([2., 10.]), np.array([0., 0.]), np.array([[1., 2.]])))
    I = hm.get_implausibility()
    
    RO = hm.get_RO()
    
    hm = HistoryMatching(obs = [1., 0.], expectations = (np.array([2., 10.]), np.array([0., 0.]), np.array([[1., 2.]])))
    
    RO = hm.get_RO(1.)
    
    assert RO == [1]

def test_HistoryMatching_set_gp():
    'test the set_gp method of HistoryMatching'
    
    gp = GaussianProcess(np.reshape(np.linspace(0., 1.), (-1, 1)), np.linspace(0., 1.))
    
    hm = HistoryMatching()
    hm.set_gp(gp)
    
    assert hm.gp == gp
    
    with pytest.raises(TypeError):
        hm.set_gp(gp = 1.)
    
def test_HistoryMatching_set_obs():
    'test the set_obs method of HistoryMatching'
    
    obs = [1., 0.1]
    
    hm = HistoryMatching()
    hm.set_obs(obs)
    
    assert_allclose(hm.obs, obs)
    
    obs = 1.
    hm.set_obs(obs)
    
    assert_allclose(hm.obs, [obs, 0.])
    
    with pytest.raises(ValueError):
        hm.set_obs([1., 2., 3.])
        
    with pytest.raises(AssertionError):
        hm.set_obs([1., -1.])
        
def test_HistoryMatching_set_coords():
    "test the set_coords method of HistoryMatching"
    
    coords = np.array([[1.]])
    
    hm = HistoryMatching()
    hm.set_coords(coords)
    
    assert_allclose(hm.coords, coords)
    
    coords = np.array([1.])
    
    hm.set_coords(coords)
    
    assert_allclose(hm.coords, np.reshape(coords, (1, 1)))
    
    hm.set_coords(None)
    
    assert hm.coords == None
    
def test_HistoryMatching_set_expectations():
    "test the set_expectations method of HistoryMatching"
    
    expectations = (np.array([0.]), np.array([0.1]), np.array([[2.]]))
    
    hm = HistoryMatching()
    hm.set_expectations(expectations)
    
    for a, b in zip(hm.expectations, expectations):
        assert_allclose(a, b)
        
    hm.set_expectations(None)
    assert hm.expectations == None
    
    expectations = (np.array([0.]), np.array([-0.1]), np.array([[2.]]))
    
    with pytest.raises(AssertionError):
        hm.set_expectations(expectations)
        
    
def test_HistoryMatching_set_threshold():
    "test the set_threshold method of HistoryMatching"
    
    hm = HistoryMatching()
    
    hm.set_threshold(1.)
    
    assert_allclose(hm.threshold, 1.)
    
    with pytest.raises(AssertionError):
        hm.set_threshold(-1.)
        
def test_HistoryMatching_update():
    "test the update method of HistoryMatching"
    
    hm = HistoryMatching()
    
    hm.coords = np.array([[1.]])
    
    hm.update()
    
    assert hm.ncoords == 1
    assert hm.ndim == 1
    
    hm = HistoryMatching()
    
    hm.expectations = (np.array([0.]), np.array([0.1]), np.array([[2.]]))
    hm.update()
    
    assert hm.ncoords == 1
    
def test_HistoryMatching_str():
    "test the str method of HistoryMatching"
    
    hm = HistoryMatching()
    
    str_exp = ("History Matching tools created with:\n" + 
          "Gaussian Process: None\n" +
          "Observations: None\n" +
          "Coords: None\n" +
          "Expectations: None\n" +
          "No. of Input Dimensions: None\n" +
          "No. of Descrete Expectation Values: None\n" +
          "I_threshold: {}\n" + 
          "I: None\n" +
          "NROY: None\n" + 
          "RO: None\n").format(3.)
          
    assert str(hm) == str_exp