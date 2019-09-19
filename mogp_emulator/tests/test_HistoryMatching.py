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

def get_y_simulated_1D(x):
    n_points = len(x)
    f = np.zeros(n_points)
    for i in range(n_points):
        f[i] = np.sin(2.*np.pi*x[i] / 50.) 
    return f

def sanity_checks():
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
    print("Implausability test a - no vars")
    hm = HistoryMatching(obs=obs, gp=gp, coords=coords)
    I = hm.get_implausability()

    print("Implausability test b - single value")
    hm = HistoryMatching(obs=obs, gp=gp, coords=coords)
    I = hm.get_implausability(7.)

    print("Implausability test c - multiple values")
    hm = HistoryMatching(obs=obs, gp=gp, coords=coords)
    I = hm.get_implausability(7., 8, 109.5)

    print("Implausability test d - single list")
    hm = HistoryMatching(obs=obs, gp=gp, coords=coords)
    var = [a for a in range(2000)]
    I = hm.get_implausability([a for a in range(2000)])

    print("Implausability test d - multiple lists")
    hm = HistoryMatching(obs=obs, gp=gp, coords=coords)
    I = hm.get_implausability(var, [v+2 for v in var], [v*7 for v in var])

    print("Implausability test e - single 1D ndarray ")
    hm = HistoryMatching(obs=obs, gp=gp, coords=coords)
    var = np.asarray(var)
    I = hm.get_implausability(var)

    print("Implausability test f - single 2D ndarray ")
    hm = HistoryMatching(obs=obs, gp=gp, coords=coords)
    var = np.reshape(np.asarray(var), (-1,1))
    I = hm.get_implausability(var)
