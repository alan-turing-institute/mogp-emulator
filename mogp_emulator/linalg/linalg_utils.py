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

def calc_mean_params(Ainv, Kinv_t, dm, B):
    "Compute analytical mean solution"
    
    assert isinstance(Ainv, ChoInv)
    assert isinstance(B, MeanPriors)
    
    return Ainv.solve(np.dot(dm.T, Kinv_t) + B.inv_cov_b())
    
def calc_R(Kinv_Ktest, dm, dmtest):
    "Compute R matrix"
    
    return dmtest.T - np.dot(dm.T, Kinv_Ktest)