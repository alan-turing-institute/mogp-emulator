import numpy as np
from scipy.spatial.distance import pdist, squareform


def gram_matrix(X, k):
    """Computes the Gram matrix of `X`

    :type X: ndarray

    :param X: Two-dimensional numpy array, where rows are feature
    vectors

    :param k: The covariance function

    :returns: The gram matrix of ``X`` under the kernel ``k``, that is
    G_ij = k(X_i, X_j)
    """
    return squareform(pdist(X, k))

def gram_matrix_sqexp(X, sigma2):
    """Computes the Gram matrix of `X` under the squared expontial kernel.

    :type X: ndarray

    :param X: Two-dimensional numpy array, where rows are feature
    vectors

    :param sigma2: The variance parameter of the squared exponential kernel

    :returns: The gram matrix of ``X`` under the kernel ``k``, that is
    G_ij = k(X_i, X_j)
    """
    return np.exp(-0.5 * squareform(pdist(X, 'sqeuclidean')) / sigma2)


def median_dist(X):
    """
    median of the pairwise (Euclidean) distances
    """
    return np.median(pdist(X))
    

class gKDR(object):

    """Dimension reduction by the gKDR method.

    See (reference) for details of the method.

    Note that this is a simpler and faster method than the original
    KDR method (but more approximate).  The KDR method will be
    implemented separately.

    An instance of this class is callable, with the ``__call__``
    method taking an input coordinate and mapping it to a reduced
    coordinate.
    """
    
    def __init__(self, X, Y, K, EPS=1E-8, SGX=None, SGY=None):
        """This is a *direct* translation of the Matlab implementation of
        KernelDeriv into Python/NumPy.
        """        
        N, M = np.shape(X)

        Y = np.reshape(Y, (N,1))
        
        if SGX is None:
            SGX = median_dist(X)
        if SGY is None:
            SGY = median_dist(Y)

        I = np.eye(N)

        SGX2 = SGX*SGX
        SGY2 = SGY*SGY

        Kx = gram_matrix_sqexp(X, SGX2)
        Ky = gram_matrix_sqexp(Y, SGY2)
        
        Dx = np.reshape(np.tile(X,(N,1)), (N,N,M), order='F').copy()
        Xij = Dx - np.transpose(Dx, (1,0,2))
        Xij = Xij / SGX2
        H = Xij * np.tile(Kx[:,:,np.newaxis], (1,1,M))
        
        tmp = np.linalg.solve(Kx + N*EPS*I, Ky)
        F = np.linalg.solve((Kx + N*EPS*I).T, tmp.T).T

        Hm = np.reshape(H,(N,N*M), order='F')       
        HH = np.reshape(Hm.T @ Hm, (N,M,N,M), order='F')      
        HHm = np.reshape(np.transpose(HH, (0,2,1,3)), (N*N,M,M), order='F')
        Fm = np.tile(np.reshape(F, (N*N,1,1), order='F'), (1,M,M))
        R = np.reshape(np.sum(HHm * Fm, 0), (M,M), order='F')
        
        L, V = np.linalg.eig(R)
        idx = np.argsort(L, 0)[::-1] # reversed

        self.K = K
        self.B = V[:, idx]
        
    def __call__(self, X):
        return X @ self.B[:,0:self.K]
