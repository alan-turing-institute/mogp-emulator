import numpy as np
import pytest
from scipy import linalg
from .. import gKDR
from .. import GaussianProcess

def test_DimensionReduction_basic():
    """Basic check that we can create gKDR with the expected arguments"""
    Y = np.array([[1],[2.1],[3.2]])
    X = np.array([[1,2,3],[4,5.1,6],[7.1,8,9.1]])
    dr = gKDR(X,Y,K=2,SGX=2,SGY=2,EPS=1E-5)


def fn(x):
    """A linear function for testing the dimension reduction"""
    return 10*(x[0] + x[1]) + (x[1] - x[0])


def test_DimensionReduction_GP():
    """Test that a GP based on reduced inputs behaves well."""
    
    # Make some test points on a grid.  Any deterministic set of
    # points would work well for this test.
    
    X = np.mgrid[0:10,0:10].T.reshape(-1,2)/10.0
    Y = np.apply_along_axis(fn, 1, X)
    dr = gKDR(X,Y,1)

    gp = GaussianProcess(X, Y)
    np.random.seed(10)
    gp.learn_hyperparameters()

    gp_red = GaussianProcess(dr(X), Y)
    gp_red.learn_hyperparameters()
    
    ## some points offset w.r.t the initial grid
    Xnew = (np.mgrid[0:10,0:10].T.reshape(-1,2) + 0.5)/10.0

    Yexpect = np.apply_along_axis(fn, 1, Xnew)
    Ynew = gp.predict(Xnew)[0] # value prediction 
    Ynew_red = gp_red.predict(dr(Xnew))[0] # value prediction

    # check that the fit was reasonable in both cases
    assert(np.max(np.abs(Ynew - Yexpect)) <= 0.02)
    assert(np.max(np.abs(Ynew_red - Yexpect)) <= 0.02)
