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
    
def test_MultiOutputGP_get_n():
    "Test function for the get_n method"
    x = np.reshape(np.array([1., 2., 3.]), (1, 1, 3))
    y = np.reshape(np.array([2.]), (1, 1))
    gp = MultiOutputGP(x, y)
    assert gp.get_D() == 3
    
def test_MultiOutputGP_str():
    "Test function for string method"
    x = np.reshape(np.array([1., 2., 3.]), (1, 1, 3))
    y = np.reshape(np.array([2.]), (1, 1))
    gp = MultiOutputGP(x, y)
    assert (str(gp) == "Multi-Output Gaussian Process with:\n1 emulators\n1 training examples\n3 input variables")