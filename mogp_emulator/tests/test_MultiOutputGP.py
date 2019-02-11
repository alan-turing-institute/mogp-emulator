from tempfile import TemporaryFile
import numpy as np
import pytest
from numpy.testing import assert_allclose
from .. import MultiOutputGP

def test_MultiOutputGP_init():
    "Test function for correct functioning of the init method of MultiOutputGP"
    x = np.reshape(np.array([1., 2., 3.]), (1, 3))
    y = np.reshape(np.array([2., 4.]), (2, 1))
    gp = MultiOutputGP(x, y)
    for emulator, yvals in zip(gp.emulators, y):
        assert_allclose(emulator.inputs, x)
        assert_allclose(emulator.targets, yvals)
    assert gp.n_emulators == 2
    assert gp.n == 1
    assert gp.D == 3
    
    x = np.reshape(np.array([1., 2., 3., 4., 5., 6.]), (2, 3))
    y = np.array([2., 4.])
    gp = MultiOutputGP(x, y)
    assert_allclose(gp.emulators[0].inputs, x)
    assert_allclose(gp.emulators[0].targets, y)
    assert gp.n_emulators == 1
    assert gp.n == 2
    assert gp.D == 3
    
    with TemporaryFile() as tmp:
        np.savez(tmp, inputs=np.array([[1., 2., 3.], [4., 5., 6]]),
                                      targets = np.array([[2., 4.]]))
        tmp.seek(0)
        gp = MultiOutputGP(tmp)
        
    x = np.reshape(np.array([1., 2., 3., 4., 5., 6.]), (2, 3))
    y = np.array([2., 4.])
    assert_allclose(gp.emulators[0].inputs, x)
    assert_allclose(gp.emulators[0].targets, y)
    assert gp.n_emulators == 1
    assert gp.n == 2
    assert gp.D == 3
    with pytest.raises(AttributeError):
        gp.emulators[0].current_theta

    with TemporaryFile() as tmp:
        np.savez(tmp, inputs=np.array([[1., 2., 3.], [4., 5., 6]]),
                                      targets = np.array([[2., 4.]]),
                                      theta = np.array([[1., 2., 3., 4., 5.]]))
        tmp.seek(0)
        gp = MultiOutputGP(tmp)
        
    x = np.reshape(np.array([1., 2., 3., 4., 5., 6.]), (2, 3))
    y = np.array([2., 4.])
    theta = np.array([1., 2., 3., 4., 5.])
    assert_allclose(gp.emulators[0].inputs, x)
    assert_allclose(gp.emulators[0].targets, y)
    assert_allclose(gp.emulators[0].current_theta, theta)
    assert gp.n_emulators == 1
    assert gp.n == 2
    assert gp.D == 3

def test_MultiOutputGP_init_failures():
    "Tests of MultiOutputGP init method that should fail"
    x = np.reshape(np.array([1., 2., 3.]), (1, 3))
    y = np.reshape(np.array([2., 3., 4.]), (1, 3))
    with pytest.raises(ValueError):
        gp = MultiOutputGP(x, y)
        
    x = np.reshape(np.array([1., 2., 3., 4., 5., 6.]), (6,))
    y = np.reshape(np.array([2., 3., 4., 5.]), (2, 2))
    with pytest.raises(ValueError):
        gp = MultiOutputGP(x, y)
        
    x = np.reshape(np.array([1., 2., 3., 4., 5., 6.]), (2, 3))
    y = np.reshape(np.array([2., 3.]), (2, 1))
    with pytest.raises(ValueError):
        gp = MultiOutputGP(x, y)
        
    x = np.reshape(np.array([1., 2., 3., 4., 5., 6.]), (2, 3))
    y = np.reshape(np.array([2., 3., 4., 5.]), (4,))
    with pytest.raises(ValueError):
        gp = MultiOutputGP(x, y)

def test_MultiOutputGP_save_emulators():
    "Test function for the save_emulators method"
    x = np.reshape(np.array([1., 2., 3., 4., 5., 6.]), (2, 3))
    y = np.array([[2., 4.]])
    gp = MultiOutputGP(x, y)
    
    with TemporaryFile() as tmp:
        gp.save_emulators(tmp)
        tmp.seek(0)
        emulator_file = np.load(tmp)
        assert_allclose(emulator_file['inputs'], x)
        assert_allclose(emulator_file['targets'], y)
        with pytest.raises(KeyError):
            emulator_file['theta']
        
    x = np.reshape(np.array([1., 2., 3., 4., 5., 6., 7., 8., 9.]), (3, 3))
    y = np.reshape(np.array([2., 4., 6.]), (1, 3))
    gp = MultiOutputGP(x, y)
    theta = np.array([np.array([-15.258941170727503, -98.2891773079696  , -56.75514771203786 ,
        13.350449073864349, -21.60315424663098 ])])
    gp._set_params(theta)
    
    with TemporaryFile() as tmp:
        gp.save_emulators(tmp)
        tmp.seek(0)
        emulator_file = np.load(tmp)
        assert_allclose(emulator_file['inputs'], x)
        assert_allclose(emulator_file['targets'], y)
        assert_allclose(emulator_file['theta'], theta)

def test_MultiOutputGP_get_n_emulators():
    "Test function for the get_n_emulators method"
    x = np.reshape(np.array([1., 2., 3.]), (1, 3))
    y = np.reshape(np.array([2.]), (1, 1))
    gp = MultiOutputGP(x, y)
    assert gp.get_n_emulators() == 1
    
    x = np.reshape(np.array([1., 2., 3., 4., 5., 6.]), (2, 3))
    y = np.reshape(np.array([2., 4., 6., 8.]), (2, 2))
    gp = MultiOutputGP(x, y)
    assert gp.get_n_emulators() == 2
    
def test_MultiOutputGP_get_n():
    "Test function for the get_n method"
    x = np.reshape(np.array([1., 2., 3.]), (1, 3))
    y = np.reshape(np.array([2.]), (1, 1))
    gp = MultiOutputGP(x, y)
    assert gp.get_n() == 1
    
def test_MultiOutputGP_get_D():
    "Test function for the get_D method"
    x = np.reshape(np.array([1., 2., 3.]), (1, 3))
    y = np.reshape(np.array([2.]), (1, 1))
    gp = MultiOutputGP(x, y)
    assert gp.get_D() == 3

def test_MultiOutputGP_set_params():
    "Test function for the _set_params method"
    x = np.reshape(np.array([1., 2., 3., 4., 5., 6., 7., 8., 9.]), (3, 3))
    y = np.reshape(np.array([2., 4., 6.]), (1, 3))
    gp = MultiOutputGP(x, y)
    theta = np.array([np.array([-15.258941170727503, -98.2891773079696  , -56.75514771203786 ,
        13.350449073864349, -21.60315424663098 ])])
    theta_expected = [np.array([-15.258941170727503, -98.2891773079696  , -56.75514771203786 ,
        13.350449073864349, -21.60315424663098 ])]
    loglike_expected = [5.0351448514868675]
    gp._set_params(theta)
    for emulator, loglike_exp, theta_exp in zip(gp.emulators, loglike_expected, theta_expected):
        assert_allclose(emulator.current_loglikelihood, loglike_exp)
        assert_allclose(emulator.current_theta, theta_exp)
        
def test_MultiOutputGP_set_params_failures():
    "Test function for the _set_params method with bad inputs"
    x = np.reshape(np.array([1., 2., 3., 4., 5., 6., 7., 8., 9.]), (3, 3))
    y = np.reshape(np.array([2., 4., 6.]), (1, 3))
    gp = MultiOutputGP(x, y)
    theta = np.array([np.array([-98.28917731, -56.75514771,  13.35044907, -21.60315425])])
    with pytest.raises(AssertionError):
        gp._set_params(theta)
        
    x = np.reshape(np.array([1., 2., 3., 4., 5., 6., 7., 8., 9.]), (3, 3))
    y = np.reshape(np.array([2., 4., 6.]), (1, 3))
    gp = MultiOutputGP(x, y)
    theta = np.array([[-15.25894117, -98.28917731, -56.75514771,  13.35044907, -21.60315425],
                      [-15.25894117, -98.28917731, -56.75514771,  13.35044907, -21.60315425]])
    with pytest.raises(AssertionError):
        gp._set_params(theta)

def test_MultiOutputGP_learn_hyperparameters():
    "Test function for the learn_hyperparameters method"
    x = np.reshape(np.array([1., 2., 3., 4., 5., 6., 7., 8., 9.]), (3, 3))
    y = np.reshape(np.array([2., 4., 6.]), (1, 3))
    gp = MultiOutputGP(x, y)
    np.random.seed(12345)
    l = gp.learn_hyperparameters(processes = 1)
    loglike, theta = [list(t) for t in zip(*l)]
    loglike_expected = [5.0351448514868675]
    theta_expected = [np.array([-15.258941170727503, -98.2891773079696  , -56.75514771203786 ,
        13.350449073864349, -21.60315424663098 ])]
    for emulator, loglike_val, theta_val, loglike_exp, theta_exp in zip(gp.emulators, loglike, theta, loglike_expected, theta_expected):
        assert_allclose(loglike_val, loglike_exp)
        assert_allclose(theta_val, theta_exp)
        assert_allclose(emulator.current_loglikelihood, loglike_exp)
        assert_allclose(emulator.current_theta, theta_exp)
        
def test_MultiOutputGP_learn_hyperparameters_failures():
    "Test function for the learn_hyperparameters method with bad inputs"
    x = np.reshape(np.array([1., 2., 3., 4., 5., 6., 7., 8., 9.]), (3, 3))
    y = np.reshape(np.array([2., 4., 6.]), (1, 3))
    gp = MultiOutputGP(x, y)
    with pytest.raises(AssertionError):
        l = gp.learn_hyperparameters(n_tries = -1)
    with pytest.raises(AssertionError):
        l = gp.learn_hyperparameters(processes = -1)
    with pytest.raises(AssertionError):
        l = gp.learn_hyperparameters(x0 = [1., 2.])

def test_MultiOutputGP_predict():
    "Test function for the predict method"
    x = np.reshape(np.array([1., 2., 3., 4., 5., 6., 7., 8., 9.]), (3, 3))
    y = np.reshape(np.array([2., 4., 6.]), (1, 3)) 
    x_star = np.reshape(np.array([2., 3., 4.]), (1, 3))
    predict_expected = np.array([[2.66699913]])
    var_expected = np.array([[-38.33591033]])
    deriv_expected = np.array([[[6.66666432e-01, 5.81212843e-37, 6.34358943e-19]]])
    gp = MultiOutputGP(x, y)
    np.random.seed(12345)
    l = gp.learn_hyperparameters(processes = 1)
    predict_actual, var_actual, deriv_actual = gp.predict(x_star)
    assert_allclose(predict_actual, predict_expected)
    assert_allclose(var_actual, var_expected)
    assert_allclose(deriv_actual, deriv_expected)
    
    predict_actual, var_actual, deriv_actual = gp.predict(x_star, do_deriv = False, do_unc = False)
    
    assert_allclose(predict_actual, predict_expected)
    assert var_actual is None
    assert deriv_actual is None

    x = np.reshape(np.array([1., 2., 3., 4., 5., 6., 7., 8., 9.]), (3, 3))
    y = np.reshape(np.array([2., 4., 6.]), (1, 3)) 
    x_star = np.reshape(np.array([2., 3., 4.]), (3,))
    predict_expected = np.array([[2.66699913]])
    var_expected = np.array([[-38.33591033]])
    deriv_expected = np.array([[[6.66666432e-01, 5.81212843e-37, 6.34358943e-19]]])
    gp = MultiOutputGP(x, y)
    np.random.seed(12345)
    l = gp.learn_hyperparameters(processes = 1)
    predict_actual, var_actual, deriv_actual = gp.predict(x_star)
    assert_allclose(predict_actual, predict_expected)
    assert_allclose(var_actual, var_expected)
    assert_allclose(deriv_actual, deriv_expected)
    
    x = np.reshape(np.array([1., 2., 3., 4., 5., 6., 7., 8., 9.]), (3, 3))
    y = np.reshape(np.array([2., 4., 6., 1., 3., 5.]), (2, 3)) 
    x_star = np.reshape(np.array([2., 3., 4., 5., 6., 7.]), (2, 3))
    predict_expected = np.array([[2.66699913, 4.66700054],[1.83012099, 1.98165765]])
    var_expected = np.array([[-32.0978161, -14.4292788], [2.38779955, 2.318997]])
    deriv_expected = np.array([[[6.66666432e-01, 5.81212843e-37, 6.34358943e-19],
        [6.66667138e-01, 5.81213459e-37, 6.34359616e-19]],
        [[5.00141830e-02, 6.76677190e-08, 1.78880996e-02],
        [2.32872071e-02, 3.15069065e-08, 8.32891504e-03]]])
    gp = MultiOutputGP(x, y)
    np.random.seed(12345)
    l = gp.learn_hyperparameters(processes = 1)
    predict_actual, var_actual, deriv_actual = gp.predict(x_star)
    assert_allclose(predict_actual, predict_expected)
    assert_allclose(var_actual, var_expected)
    assert_allclose(deriv_actual, deriv_expected)

def test_MultiOutputGP_predict_failures():
    "Test function for the predict method with bad inputs"
    x = np.reshape(np.array([1., 2., 3., 4., 5., 6., 7., 8., 9.]), (3, 3))
    y = np.reshape(np.array([2., 4., 6.]), (1, 3)) 
    x_star = np.reshape(np.array([2., 3.]), (2,))
    gp = MultiOutputGP(x, y)
    np.random.seed(12345)
    l = gp.learn_hyperparameters()
    with pytest.raises(AssertionError):
        predict, unc, deriv = gp.predict(x_star)
    
    x_star = np.reshape(np.array([2., 3., 4., 6., 7., 8.]), (3, 2))
    gp = MultiOutputGP(x, y)
    np.random.seed(12345)
    l = gp.learn_hyperparameters()
    with pytest.raises(AssertionError):
        predict, unc, deriv = gp.predict(x_star)
        
    x_star = np.reshape(np.array([2., 3., 4.]), (1, 3))
    gp = MultiOutputGP(x, y)
    np.random.seed(12345)
    l = gp.learn_hyperparameters()
    with pytest.raises(AssertionError):
        predict, unc, deriv = gp.predict(x_star, processes = -1)

def test_MultiOutputGP_str():
    "Test function for string method"
    x = np.reshape(np.array([1., 2., 3.]), (1, 3))
    y = np.reshape(np.array([2.]), (1, 1))
    gp = MultiOutputGP(x, y)
    assert (str(gp) == "Multi-Output Gaussian Process with:\n1 emulators\n1 training examples\n3 input variables")