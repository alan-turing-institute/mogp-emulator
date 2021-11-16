import numpy as np
from mogp_emulator.linalg.cholesky import ChoInv, fixed_cholesky
from mogp_emulator.Priors import MeanPriors

def calc_Ainv(Kinv, dm, B):
    "Compute inverse of A matrix"
    
    assert isinstance(Kinv, ChoInv)
    assert isinstance(B, MeanPriors)
    
    A = np.dot(dm.T, Kinv.solve(dm)) + B.inv_cov()
    
    L = fixed_cholesky(A)
    
    return ChoInv(L)
    
def calc_A_deriv(Kinv, dm, dKdtheta):
    "Compute derivative of Ainv"

    assert isinstance(Kinv, ChoInv)
    assert dKdtheta.ndim == 3
    assert dKdtheta.shape[1] == dKdtheta.shape[2]
    assert dKdtheta.shape[1] == Kinv.L.shape[0]
    
    return -np.transpose(np.dot(dm.T, np.transpose(Kinv.solve(np.transpose(np.dot(dKdtheta, Kinv.solve(dm)), (1, 0, 2))), (1, 0, 2))), (1, 0, 2))

def calc_mean_params(Ainv, Kinv_t, dm, B):
    "Compute analytical mean solution"
    
    assert isinstance(Ainv, ChoInv)
    assert isinstance(B, MeanPriors)
    
    return Ainv.solve(np.dot(dm.T, Kinv_t) + B.inv_cov_b())
    
def calc_R(Kinv_Ktest, dm, dmtest):
    "Compute R matrix"
    
    return dmtest.T - np.dot(dm.T, Kinv_Ktest)
    
def logdet_deriv(Kinv, dKdtheta):
    "Compute the derivative of the logdeterminant of a matrix"
    
    assert isinstance(Kinv, ChoInv)
    assert dKdtheta.ndim == 3
    assert dKdtheta.shape[1] == dKdtheta.shape[2]
    assert dKdtheta.shape[1] == Kinv.L.shape[0]
    
    return np.trace(Kinv.solve(np.transpose(dKdtheta, (1, 2, 0))))