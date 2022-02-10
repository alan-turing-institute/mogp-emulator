import numpy as np
from mogp_emulator.linalg import cholesky_factor
from scipy.linalg import solve_triangular

def mahalanobis(predictions, mean, cov, n):
    cov_inv, _ = cholesky_factor(cov, 0., "fixed")
    M = np.dot(predictions - mean, cov_inv.solve(predictions-mean))
    n2 = len(predictions)
    var = 2*n2*(n2 + n - 2)/(n - 4)
    return (M - n2)/np.sqrt(var)

def standard_errors(predictions, mean, cov):
    if cov.ndim == 2:
        std = np.sqrt(np.diag(cov))
    else:
        std = np.sqrt(cov)
    return (predictions - mean)/std

def pivoted_errors(predictions, mean, cov):
    cov_inv, _ = cholesky_factor(cov, 0., "pivot")
    return solve_triangular(cov_inv.L, (predictions - mean)[cov_inv.P], lower=True)