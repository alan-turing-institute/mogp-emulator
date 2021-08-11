import numpy as np
from scipy import linalg
from scipy.linalg import lapack, cho_solve

def _check_cholesky_inputs(A):
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

    A = _check_cholesky_inputs(A)
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

def pivot_cholesky(A):
    """
    Pivoted cholesky decomposition routine

    Performs a pivoted Cholesky decomposition on a square, potentially
    singular (i.e. can have collinear rows) covariance matrix. Rows
    and columns are interchanged such that the diagonal entries of the
    factorized matrix decrease from the first entry to the last entry.
    Any collinear rows are skipped, and the diagonal entries for those
    rows are instead replaced by a decreasing entry. This allows the
    factorized matrix to still be used to compute the log determinant
    in the same way, which is required to compute the marginal log
    likelihood/posterior in the GP fitting.

    Returns the factorized matrix, and an ordered array of integer
    indices indicating the pivoting order. Raises an error if the
    decomposition fails.

    :param A: The matrix to be inverted as an array of shape ``(n,n)``.
              Must be a symmetric matrix (though can be singular).
    :type A: ndarray
    :returns: Lower-triangular factored matrix (shape ``(n,n)`` an an array of
              integers indicating the pivoting order needed to produce the
              factorization.
    :rtype: tuple containing an ndarray of shape `(n,n)` of floats and a
            ndarray of shape `(n,)` of integers.
    """

    A = _check_cholesky_inputs(A)

    A = np.ascontiguousarray(A)
    L, P, rank, info = lapack.dpstrf(A, lower = 1)
    L = np.tril(L)

    if info < 0:
        raise linalg.LinAlgError("Illegal value in covariance matrix")

    n = A.shape[0]

    idx = np.arange(rank, n)
    divs = np.cumprod(np.arange(rank+1, n+1, dtype=np.float64))
    L[idx, idx] = L[rank-1, rank-1]/divs

    return L, P-1

def _pivot_transpose(P):
    """
    Invert a pivot matrix by taking its transpose

    Inverts the pivot matrix by taking its transpose. Since the pivot
    matrix is represented by an array of integers(to make swapping the
    rows more simple by using numpy indexing), this is done by
    creating the equivalent array of integers representing the
    transpose.  Input must be a list of non-negative integers, where
    each integer up to the length of the array appears exactly
    once. Otherwise this function will raise an exception.

    :param P: Input pivot matrix, represented as an array of non-negative
              integers.
    :type P: ndarray containing integers
    :returns: Transpose of the pivot matrix, represented as an array of
              non-negative integers.
    :rtype: ndarray containing integers.

    """

    assert np.array(P).ndim == 1, "pivot matrix must be a list of integers"

    try:
        return np.array([np.where(P == idx)[0][0] for idx in range(len(P))], dtype=np.int32)
    except IndexError:
        raise ValueError("Bad values for pivot matrix input to pivot_transpose")


def pivot_cho_solve(L, P, b):
    """Solve a Linear System factorized using Pivoted Cholesky Decomposition

    Solve a system :math:`{Ax = b}` where the matrix has been factorized using the
    `pivot_cholesky` function. Can also solve a system which has been
    factorized using the regular Cholesky decomposition routine if
    `P` is the set of integers from 0 to the length of the linear system.
    The routine rearranges the order of the RHS based on the pivoting
    order that was used, and then rearranges back to the original
    ordering of the RHS when returning the array.

    :param L: Factorized :math:`{A}` square matrix using
              pivoting. Assumes lower triangular factorization is
              used.
    :type L: ndarray
    :param P: Pivot matrix, expressed as a list or 1D array of
              integers that is the same length as `L`. Must contain
              only nonzero integers as entries, with each number up to
              the length of the array appearing exactly once.
    :type P: list or ndarray
    :param b: Right hand side to be solved. Can be any array that
              satisfies the rules of the scipy `cho_solve` routine.
    :type b: ndarray
    :returns: Solution to the appropriate linear system as a ndarray.
    :rtype: ndarray
    """

    assert len(P) == L.shape[0], "Length of pivot matrix must match linear system"

    try:
        return cho_solve((L, True), b[P])[_pivot_transpose(P)]
    except (IndexError, ValueError):
        raise ValueError("Bad values for pivot matrix in pivot_cho_solve")
