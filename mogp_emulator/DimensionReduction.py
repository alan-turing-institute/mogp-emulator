"""This module provides classes and utilities for performing dimension
reduction.  Currently there is a single class :class:`mogp_emulator.gKDR` which
implements the method of Fukumizu and Leng [FL13]_.

Example: ::

  >>> from mogp_emulator import gKDR
  >>> import numpy as np
  >>> X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
  >>> Y = np.array([0.0, 1.0, 5.0, 6.0])
  >>> xnew = np.array([0.5, 0.5])
  >>> dr = gKDR(X, Y, 1)
  >>> dr(xnew)
  array([0.60092477])

In this example, the reduction was performed from a two- to a one-dimensional
input space.  The value returned by ``dr(xnew)`` is the input coordinate `xnew`
transformed to the reduced space.

The following example illustrates how to perform Gaussian process regression on
the reduced input space:

::

  >>> import numpy as np
  >>> from mogp_emulator import gKDR, GaussianProcess

  ### generate some training data (from the function f)

  >>> def f(x):
  ...     return np.sqrt(x[0] + np.sin(0.1 * x[1]))

  >>> X = np.mgrid[0:10,0:10].T.reshape(-1,2)/10.0
  >>> print(X)
  [[0.  0. ]
   [0.1 0. ]
   [0.2 0. ]
   [0.3 0. ]
   [0.4 0. ]
   [0.5 0. ]
   [0.6 0. ]
   [0.7 0. ]
   [0.8 0. ]
   [0.9 0. ]
   [0.  0.1]
   [0.1 0.1]
   ...
   [0.8 0.9]
   [0.9 0.9]]

  >>> Y = np.apply_along_axis(f, 1, X)

  ### reduced input space
  >>> dr = gKDR(X,Y,1)

  ### train a Gaussian Process with reduced inputs
  >>> gp = GaussianProcess(dr(X), Y)
  >>> gp.learn_hyperparameters()

  ### make a prediction (given an input in the reduced space)
  >>> Xnew = np.array([0.12, 0.37])
  >>> gp.predict(dr(Xnew))
  (array([0.398083]), ...)

  >>> f(Xnew)
  0.396221

"""

import sys
import numpy as np
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.optimize import minimize


def gram_matrix(X, k):
    """Computes the Gram matrix of `X`

    :type X: ndarray

    :param X: Two-dimensional numpy array, where rows are feature
              vectors

    :param k: The covariance function

    :returns: The gram matrix of `X` under the kernel `k`, that is,
              :math:`G_{ij} = k(X_i, X_j)`
    """
    # note: do not use squareform(pdist(X, k)) here, since it assumes
    # that dist(x,x) == 0, which might not be the case for an arbitrary k.
    return cdist(X, X, k)


def gram_matrix_sqexp(X, sigma2):
    r"""Computes the Gram matrix of `X` under the squared expontial kernel.
    Equivalent to, but more efficient than, calling ``gram_matrix(X,
    k_sqexp)``

    :type X: ndarray

    :param X: Two-dimensional numpy array, where rows are feature
              vectors

    :param sigma2: The variance parameter of the squared exponential kernel

    :returns: The gram matrix of `X` under the squared exponential
              kernel `k_sqexp` with variance parameter `sigma2`
              (:math:`=\sigma^2`), that is, :math:`G_{ij} = k_{sqexp}(X_i, X_j;
              \sigma^2)`

    """
    return np.exp(-0.5 * squareform(pdist(X, 'sqeuclidean')) / sigma2)


def median_dist(X):
    """Return the median of the pairwise (Euclidean) distances between
    each row of X
    """
    return np.median(pdist(X))


class gKDR(object):

    """Dimension reduction by the gKDR method.

    See link [Fukumizu1]_ (and in particular, [FL13]_) for details of the
    method.

    Note that this is a simpler and faster method than the original "KDR"
    method by the same authors (but with an added approximation).  The KDR
    method will be implemented separately.

    An instance of this class is callable, with the ``__call__`` method taking
    an input coordinate and mapping it to a reduced coordinate.

    Note that this class currently implements a *direct* translation of the
    Matlab implementation of KernelDeriv (see link above) into Python/NumPy.
    It is due to be replaced with a Fortran implementation, but this should not
    affect the interface.
    """

    def __init__(self, X, Y, K, EPS=1E-8, SGX=None, SGY=None):
        """Create a gKDR object

        Given some `M`-dimensional inputs (explanatory variables) `X`, and
        corresponding one-dimensional outputs (responses) `Y`, use the gKDR
        method to produce a reduced version of the input space with `K`
        dimensions.

        :type X: ndarray, of shape (N, M)
        :param X: `N` rows of `M` dimensional input vectors

        :type Y: ndarray, of shape (N,)
        :param Y: `N` response values

        :type K: integer
        :param K: The number of reduced dimensions to use (`0 <= K <= M`).

        :type EPS: float
        :param EPS: The regularization parameter, default `1e-08`; `EPS >= 0`

        :type SGX: float | NoneType
        :param SGX: Optional, default `None`. The kernel parameter representing
                    the scale of variation on the input space.  If `None`, then
                    the median distance between pairs of input points (`X`) is
                    used (as computed by
                    :func:`mogp_emulator.DimensionReduction.median_dist`).  If
                    a float is passed, then this must be positive.

        :type SGY: float | NoneType
        :param SGY: Optional, default `None`. The kernel parameter representing
                    the scale of variation on the output space.  If `None`,
                    then the median distance between pairs of output values
                    (`Y`) is used (as computed by
                    :func:`mogp_emulator.DimensionReduction.median_dist`).  If
                    a float is passed, then this must be positive.
        """

        # Note: see the Matlab implementation ...

        N, M = np.shape(X)

        assert(K >= 0 and K <= M)
        assert(EPS >= 0)
        assert(SGX is None or SGX > 0.0)
        assert(SGY is None or SGY > 0.0)

        Y = np.reshape(Y, (N, 1))

        if SGX is None:
            SGX = median_dist(X)
        if SGY is None:
            SGY = median_dist(Y)

        eye = np.eye(N)

        SGX2 = max(SGX*SGX, sys.float_info.min)
        SGY2 = max(SGY*SGY, sys.float_info.min)

        Kx = gram_matrix_sqexp(X, SGX2)
        Ky = gram_matrix_sqexp(Y, SGY2)

        Dx = np.reshape(np.tile(X, (N, 1)), (N, N, M), order='F').copy()
        Xij = Dx - np.transpose(Dx, (1, 0, 2))
        Xij = Xij / SGX2
        H = Xij * np.tile(Kx[:, :, np.newaxis], (1, 1, M))

        tmp = np.linalg.solve(Kx + N*EPS*eye, Ky)
        F = np.linalg.solve((Kx + N*EPS*eye).T, tmp.T).T

        Hm = np.reshape(H, (N, N*M), order='F')
        HH = np.reshape(Hm.T @ Hm, (N, M, N, M), order='F')
        HHm = np.reshape(np.transpose(HH, (0, 2, 1, 3)), (N*N, M, M),
                         order='F')
        Fm = np.tile(np.reshape(F, (N*N, 1, 1), order='F'), (1, M, M))
        R = np.reshape(np.sum(HHm * Fm, 0), (M, M), order='F')

        L, V = np.linalg.eig(R)
        idx = np.argsort(L, 0)[::-1]  # sort descending

        self.K = K
        self.B = V[:, idx]

    def __call__(self, X):
        """Calling a gKDR object with a vector of N inputs returns the inputs
        mapped to the reduced space.

        :type X: ndarray, of shape `(N, M)`
        :param X: `N` coordinates (rows) in the unreduced `M`-dimensional space

        :rtype: ndarray, of shape `(N, K)`
        :returns:  `N` coordinates (rows) in the reduced `K`-dimensional space
        """
        return X @ self.B[:, 0:self.K]


class KDR(object):
    """
    Dimension reduction using the KDR method
    """

    @classmethod
    def from_gKDR(cls, X, Y, gKDR, K=None, EPS=1e-8, SGX=None, SGY=None):
        """
        Construct a KDR object using a gKDR object as an initial guess for the
        projection matrix.
        """
        if K is None:
            K = gKDR.K
        B = gKDR.B[:, 0:K].copy()

        return cls(X, Y, K, EPS=EPS, SGX=SGX, SGY=SGY, B=B)

    def __init__(self, X, Y, K, EPS=1e-8, SGX=None, SGY=None, B=None):
        N, M = np.shape(X)
        Y = np.reshape(Y, (N, 1))
        self.K = K

        # If an initial guess for the projection matrix (B) is not provided,
        # generate a random one
        if B is None:
            B = np.random.random([M, K])

        # Determine Ky, SGY and SGX. If gKDR has been conducted this is
        # duplicated work. It would be preferred to eventually fetch SGY, SGX
        # and Ky from the gKDR object.
        if SGY is None:
            SGY = median_dist(Y)
        if SGX is None:
            SGX = median_dist(X)

        SGX2 = max(SGX*SGX, sys.float_info.min)
        SGY2 = max(SGY*SGY, sys.float_info.min)

        Ky = gram_matrix_sqexp(Y, SGY2)

        # orthogonalisation matrix
        unit = np.ones([N, N])
        eye = np.eye(N)
        Q = eye - unit/N

        # Orthogonalise Ky
        # Ky_o is the orthogonalised Gram matrix, corresponding to \hat{K}_y in
        # the paper
        Ky_o = self._orthogonalise(Ky, Q)

        # Flatten B for optimisation, scipy.optimize.minimize accepts a 1D
        # numpy array as the objective function arguments
        B_flat = B.flatten()
        # Minimise the objective function
        result = minimize(self._objective_function, B_flat,
                          args=(M, K, X, SGX2, N, EPS, Q, eye, Ky_o))
        # Reshape the optimised B from the minimisation result
        self.B = result.x.reshape([M, K])

    def __call__(self, X):
        return X @ self.B[:, 0:self.K]

    def _objective_function(self, B_flat, M, K, X, SGX2, N, EPS, Q, eye, Ky_o):
        B = B_flat.reshape([M, K])
        # Singular value decomposition of projection matrix
        B, *_ = np.linalg.svd(B)

        # Z corresponds to U in the paper
        Z = X @ B
        Kz = gram_matrix_sqexp(Z, sigma2=SGX2)
        Kz = self._orthogonalise(Kz, Q)

        mz = np.linalg.inv(Kz + N*EPS*eye)
        return np.sum(Ky_o * mz)

    @staticmethod
    def _orthogonalise(a, Q):
        o = Q @ a @ Q
        o = (o + o.T) / 2.
        return o
