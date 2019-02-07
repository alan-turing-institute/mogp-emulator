import numpy as np
import pytest
from numpy.testing import assert_allclose
from .. import MultiOutputGP

def test_MultiOutputGP_init():
    "Test function for correct functioning of the init method of MultiOutputGP"
    x = np.reshape(np.array([1., 2., 3., 4., 5., 6.]), (2, 1, 3))
    y = np.reshape(np.array([2., 4.]), (2, 1))
    gp = MultiOutputGP(x, y)
    for emulator, xvals, yvals in zip(gp.emulators, x, y):
        assert_allclose(emulator.inputs, xvals)
        assert_allclose(emulator.targets, yvals)
    assert gp.n_emulators == 2
    assert gp.n == 1
    assert gp.D == 3
    
    x = np.reshape(np.array([1., 2., 3.]), (1, 3))
    y = np.reshape(np.array([2.]), (1, ))
    gp = MultiOutputGP(x, y)
    assert_allclose(gp.emulators[0].inputs, x)
    assert_allclose(gp.emulators[0].targets, y)
    assert gp.n_emulators == 1
    assert gp.n == 1
    assert gp.D == 3

def test_MultiOutputGP_init_failures():
    "Tests of MultiOutputGP init method that should fail"
    x = np.reshape(np.array([1., 2., 3., 4., 5., 6.]), (2, 1, 3))
    y = np.reshape(np.array([2.]), (1, ))
    with pytest.raises(ValueError):
        gp = MultiOutputGP(x, y)
        
    x = np.reshape(np.array([1., 2., 3., 4., 5., 6.]), (2, 3))
    y = np.reshape(np.array([2., 3., 4., 5.]), (2, 2))
    with pytest.raises(ValueError):
        gp = MultiOutputGP(x, y)
        
    x = np.reshape(np.array([1., 2., 3., 4., 5., 6.]), (2, 1, 3))
    y = np.reshape(np.array([2., 3.]), (1, 2))
    with pytest.raises(ValueError):
        gp = MultiOutputGP(x, y)
        
    x = np.reshape(np.array([1., 2., 3., 4., 5., 6.]), (2, 3))
    y = np.reshape(np.array([2., 3., 4., 5.]), (4,))
    with pytest.raises(ValueError):
        gp = MultiOutputGP(x, y)

def test_MultiOutputGP_get_n_emulators():
    "Test function for the get_n_emulators method"
    x = np.reshape(np.array([1., 2., 3.]), (1, 1, 3))
    y = np.reshape(np.array([2.]), (1, 1))
    gp = MultiOutputGP(x, y)
    assert gp.get_n_emulators() == 1
    
    x = np.reshape(np.array([1., 2., 3., 4., 5., 6.]), (2, 1, 3))
    y = np.reshape(np.array([2., 4.]), (2, 1))
    gp = MultiOutputGP(x, y)
    assert gp.get_n_emulators() == 2
    
def test_MultiOutputGP_get_n():
    "Test function for the get_n method"
    x = np.reshape(np.array([1., 2., 3.]), (1, 1, 3))
    y = np.reshape(np.array([2.]), (1, 1))
    gp = MultiOutputGP(x, y)
    assert gp.get_n() == 1
    
def test_MultiOutputGP_get_D():
    "Test function for the get_D method"
    x = np.reshape(np.array([1., 2., 3.]), (1, 1, 3))
    y = np.reshape(np.array([2.]), (1, 1))
    gp = MultiOutputGP(x, y)
    assert gp.get_D() == 3

def test_MultiOutputGP_learn_hyperparameters():
    "Test function for the learn_hyperparameters method"
    x = np.reshape(np.array([1., 2., 3., 4., 5., 6., 7., 8., 9.]), (1, 3, 3))
    y = np.reshape(np.array([2., 4., 6.]), (1, 3))
    gp = MultiOutputGP(x, y)
    np.random.seed(12345)
    l = gp.learn_hyperparameters()
    loglike, theta = [list(t) for t in zip(*l)]
    loglike_expected = [5.0351448514868675]
    theta_expected = [np.array([-15.25894117, -98.28917731, -56.75514771,  13.35044907, -21.60315425])]
    for emulator, loglike_val, theta_val, loglike_exp, theta_exp in zip(gp.emulators, loglike, theta, loglike_expected, theta_expected):
        assert_allclose(loglike_val, loglike_exp)
        assert_allclose(theta_val, theta_exp)
        assert_allclose(emulator.current_loglikelihood, loglike_exp)
        assert_allclose(emulator.current_theta, theta_exp)
        
def test_MultiOutputGP_learn_hyperparameters_failures():
    "Test function for the learn_hyperparameters method with bad inputs"
    x = np.reshape(np.array([1., 2., 3., 4., 5., 6., 7., 8., 9.]), (1, 3, 3))
    y = np.reshape(np.array([2., 4., 6.]), (1, 3))
    gp = MultiOutputGP(x, y)
    with pytest.raises(AssertionError):
        l = gp.learn_hyperparameters(n_tries = -1)
    with pytest.raises(AssertionError):
        l = gp.learn_hyperparameters(processes = -1)
    with pytest.raises(AssertionError):
        l = gp.learn_hyperparameters(x0 = [1., 2.])

def test_MultiOutputGP_str():
    "Test function for string method"
    x = np.reshape(np.array([1., 2., 3.]), (1, 1, 3))
    y = np.reshape(np.array([2.]), (1, 1))
    gp = MultiOutputGP(x, y)
    assert (str(gp) == "Multi-Output Gaussian Process with:\n1 emulators\n1 training examples\n3 input variables")