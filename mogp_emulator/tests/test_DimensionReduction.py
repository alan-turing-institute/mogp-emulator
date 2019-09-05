import numpy as np
import pytest
from .. import gKDR
from .. import GaussianProcess
from scipy import linalg

def fn(x):
    return 10*(x[0] + x[1]) + (x[1] - x[0])

def test_DimensionReduction_basic():
    Y = np.array([[1],[2.1],[3.2]])
    X = np.array([[1,2,3],[4,5.1,6],[7.1,8,9.1]])
    dr = gKDR(X,Y,2,2,2,1E-5)

def test_DimensionReduction_GP():
    X = np.random.rand(100,2)
    Y = np.apply_along_axis(fn, 1, X)
    dr = gKDR(X,Y,1)

    gp = GaussianProcess(X, Y)
    gp.learn_hyperparameters()

    gp_red = GaussianProcess(dr(X), Y)
    gp_red.learn_hyperparameters()
    
    Xnew = np.random.rand(50,2)

    Yexpect = np.apply_along_axis(fn, 1, Xnew)
    Ynew = gp.predict(Xnew)[0]
    Ynew_red = gp_red.predict(dr(Xnew))[0]

    print(np.mean(np.abs(Ynew - Yexpect)))
    print(np.mean(np.abs(Ynew_red - Yexpect)))
