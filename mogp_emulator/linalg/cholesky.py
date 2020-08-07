import numpy as np
from scipy import linalg
from scipy.linalg import lapack

def check_cholesky_inputs(A):
    """
    Check inputs to cholesky routines

    This function is used by both specialized Cholesky routines to check inputs. It verifies
    that the input is a 2D square matrix and that all diagonal elements are positive (a
    necessary but not sufficient condition for positive definiteness). If these checks pass,
    it returns the input as a numpy array.

    :param A: The matrix to be inverted as an array of shape ``(n,n)``. Must be a symmetric positive
              definite matrix.
    :type A: ndarray or similar
    :returns: The matrix to be inverted as an array of shape ``(n,n)``. Must be a symmetric positive
              definite matrix.
    :rtype: ndarray
    """

    A = np.array(A)
    assert A.ndim == 2, "A must have shape (n,n)"
    assert A.shape[0] == A.shape[1], "A must have shape (n,n)"
    np.testing.assert_allclose(A.T, A)

    diagA = np.diag(A)
    if np.any(diagA <= 0.):
        raise linalg.LinAlgError("not pd: non-positive diagonal elements")

    return A

def jit_cholesky(A, maxtries = 5):
    """
    Performs Jittered Cholesky Decomposition

    Performs a Jittered Cholesky decomposition, adding noise to the diagonal of the matrix as needed
    in order to ensure that the matrix can be inverted. Adapted from code in GPy.

    On occasion, the matrix that needs to be inverted in fitting a GP is nearly singular. This arises
    when the training samples are very close to one another, and can be averted by adding a noise term
    to the diagonal of the matrix. This routine performs an exact Cholesky decomposition if it can
    be done, and if it cannot it successively adds noise to the diagonal (starting with 1.e-6 times
    the mean of the diagonal and incrementing by a factor of 10 each time) until the matrix can be
    decomposed or the algorithm reaches ``maxtries`` attempts. The routine returns the lower
    triangular matrix and the amount of noise necessary to stabilize the decomposition.

    :param A: The matrix to be inverted as an array of shape ``(n,n)``. Must be a symmetric positive
              definite matrix.
    :type A: ndarray
    :param maxtries: (optional) Maximum allowable number of attempts to stabilize the Cholesky
                     Decomposition. Must be a positive integer (default = 5)
    :type maxtries: int
    :returns: Lower-triangular factored matrix (shape ``(n,n)`` and the noise that was added to
              the diagonal to achieve that result.
    :rtype: tuple containing an ndarray and a float
    """

    A = check_cholesky_inputs(A)
    assert int(maxtries) > 0, "maxtries must be a positive integer"

    A = np.ascontiguousarray(A)
    L, info = lapack.dpotrf(A, lower = 1)
    if info == 0:
        return L, 0.
    else:
        diagA = np.diag(A)
        jitter = diagA.mean() * 1e-6
        num_tries = 1
        while num_tries <= maxtries and np.isfinite(jitter):
            try:
                L = linalg.cholesky(A + np.eye(A.shape[0]) * jitter, lower=True)
                return L, jitter
            except:
                jitter *= 10
            finally:
                num_tries += 1
        raise linalg.LinAlgError("not positive definite, even with jitter.")

    return L, jitter
