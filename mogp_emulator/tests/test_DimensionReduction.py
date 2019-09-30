import numpy as np
from .. import gKDR
from ..DimensionReduction import median_dist, gram_matrix_sqexp, gram_matrix
from .. import GaussianProcess


def test_DimensionReduction_basic():
    """Basic check that we can create gKDR with the expected arguments"""
    Y = np.array([[1], [2.1], [3.2]])
    X = np.array([[1, 2, 3], [4, 5.1, 6], [7.1, 8, 9.1]])
    gKDR(X, Y, K=2, SGX=2, SGY=2, EPS=1E-5)


def fn(x):
    """A linear function for testing the dimension reduction"""
    return 10*(x[0] + x[1]) + (x[1] - x[0])


def test_DimensionReduction_GP():
    """Test that a GP based on reduced inputs behaves well."""

    # Make some test points on a grid.  Any deterministic set of
    # points would work well for this test.

    X = np.mgrid[0:10, 0:10].T.reshape(-1, 2)/10.0
    Y = np.apply_along_axis(fn, 1, X)
    dr = gKDR(X, Y, 1)

    gp = GaussianProcess(X, Y)
    np.random.seed(10)
    gp.learn_hyperparameters()

    gp_red = GaussianProcess(dr(X), Y)
    gp_red.learn_hyperparameters()

    # some points offset w.r.t the initial grid
    Xnew = (np.mgrid[0:10, 0:10].T.reshape(-1, 2) + 0.5)/10.0

    Yexpect = np.apply_along_axis(fn, 1, Xnew)
    Ynew = gp.predict(Xnew)[0]  # value prediction
    Ynew_red = gp_red.predict(dr(Xnew))[0]  # value prediction

    # check that the fit was reasonable in both cases
    assert(np.max(np.abs(Ynew - Yexpect)) <= 0.02)
    assert(np.max(np.abs(Ynew_red - Yexpect)) <= 0.02)


def test_DimensionReduction_B():
    """Test that a dimension reduction gives the same result as a
    pre-computed result from the Fukumizu matlab code (see
    [Fukumizu1]_)."""

    X = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    Y = np.array([0.1, 1.0, 3.0, 3.6])
    dr = gKDR(X, Y, 2, SGX=1.0, SGY=2.0)

    B_expected = np.array([[-0.2653073259794961, -0.9641638982982144],
                           [-0.9641638982982144,  0.2653073259794961]])

    for i in range(B_expected.shape[1]):
        r = dr.B[:, i]/B_expected[:, i]
        assert(np.allclose(r, 1.0) or np.allclose(r, -1.0))


def test_DimensionReduction_median_dist():
    X1 = np.array([[0.0], [1.0], [2.0]])
    assert(np.allclose(median_dist(X1), 1))

    X2 = np.array([[0.0], [1.0], [2.0], [3.0]])
    assert(np.allclose(median_dist(X2), 1.5))


def test_DimensionReduction_gram_matrix():
    X = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])

    def k_dot(x0, x1):
        return np.dot(x0, x1)

    def k_sqexp(x0, x1):
        d = x0 - x1
        return np.exp(-0.5 * np.dot(d, d))

    G_dot = gram_matrix(X, k_dot)
    assert(np.allclose(G_dot, np.array([[0.0, 0.0, 0.0, 0.0],
                                        [0.0, 1.0, 0.0, 1.0],
                                        [0.0, 0.0, 1.0, 1.0],
                                        [0.0, 1.0, 1.0, 2.0]])))

    G_sqexp1 = gram_matrix_sqexp(X, 1.0)
    G_sqexp2 = gram_matrix(X, k_sqexp)
    G_sqexp_expected = np.exp(np.array([[+0.0, -0.5, -0.5, -1.0],
                                        [-0.5,  0.0, -1.0, -0.5],
                                        [-0.5, -1.0,  0.0, -0.5],
                                        [-1.0, -0.5, -0.5,  0.0]]))

    assert(np.allclose(G_sqexp1, G_sqexp_expected))
    assert(np.allclose(G_sqexp2, G_sqexp_expected))
