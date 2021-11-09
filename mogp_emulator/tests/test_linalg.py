import numpy as np
import pytest
from numpy.testing import assert_allclose
from ..linalg.cholesky import ChoInv, ChoInvPivot, cholesky_factor, fixed_cholesky
from ..linalg.cholesky import jit_cholesky, _check_cholesky_inputs, pivot_cholesky, _pivot_transpose
from scipy import linalg

@pytest.fixture
def A():
    return np.array([[2., 1., 0.4], [1., 2., 0.2], [0.4, 0.2, 2.]])
    
@pytest.fixture
def b():
    return np.array([2., 3., 1.])

def test_ChoInv(A, b):
    "test the ChoInv class"
    
    L = linalg.cholesky(A, lower=True)
    Ainv = ChoInv(L)
    
    assert_allclose(Ainv.L, L)
    
    x = np.linalg.solve(A, b)
    
    assert_allclose(Ainv.solve(b), x)
    
    assert_allclose(np.log(np.linalg.det(A)), Ainv.logdetK())

def test_ChoInnvPivot(A, b):
    "test the cho_solve routine using pivoting"

    L = np.linalg.cholesky(A)

    x = linalg.cho_solve((L, True), b)

    L_pivot, P = pivot_cholesky(A)
    
    Ainv = ChoInvPivot(L_pivot, P)

    x_pivot = Ainv.solve(b)

    assert_allclose(x, x_pivot)

    with pytest.raises(AssertionError):
        ChoInvPivot(L_pivot, np.array([0, 2, 1, 1], dtype=np.int32)).solve(b)

    with pytest.raises(ValueError):
        ChoInvPivot(L_pivot, np.array([0, 0, 1], dtype=np.int32)).solve(b)

def test_check_cholesky_inputs():
    "Test function that checks inputs to cholesky decomposition routines"
    
    A = np.array([[2., 1.], [1., 2.]])
    B = _check_cholesky_inputs(A)
    
    assert_allclose(A, B)

    A = np.array([[1., 2.], [1., 2.]])
    with pytest.raises(AssertionError):
        _check_cholesky_inputs(A)
    
    A = np.array([1., 2.])
    with pytest.raises(AssertionError):
        _check_cholesky_inputs(A)
        
    A = np.array([[1., 2., 3.], [4., 5., 6.]])
    with pytest.raises(AssertionError):
        _check_cholesky_inputs(A)
    
    input_matrix = np.array([[-1., 2., 2.], [2., 3., 2.], [2., 2., -3.]])
    with pytest.raises(linalg.LinAlgError):
        _check_cholesky_inputs(input_matrix)

def test_fixed_cholesky(A):
    "Test the cholesky routine with fixed nugget"
    
    L_expected = np.array([[2., 0., 0.], [6., 1., 0.], [-8., 5., 3.]])
    input_matrix = np.array([[4., 12., -16.], [12., 37., -43.], [-16., -43., 98.]])
    
    L_actual = fixed_cholesky(input_matrix)
    assert_allclose(L_actual, L_expected)
    
    L_actual, nugget = cholesky_factor(input_matrix, 0., "fixed")
    assert_allclose(L_actual.L, L_expected)
    assert nugget == 0.
    
    L_actual, nugget = cholesky_factor(input_matrix, 0., "fit")
    assert_allclose(L_actual.L, L_expected)
    assert nugget == 0.
    
    L_expected = np.array([[1.0000004999998751e+00, 0.0000000000000000e+00, 0.0000000000000000e+00],
                         [9.9999950000037496e-01, 1.4142132088085626e-03, 0.0000000000000000e+00],
                         [6.7379436301144941e-03, 4.7644444411381860e-06, 9.9997779980004420e-01]])
    input_matrix = np.array([[1. + 1.e-6        , 1.                , 0.0067379469990855 ],
                             [1.                , 1. + 1.e-6        , 0.0067379469990855 ],
                             [0.0067379469990855, 0.0067379469990855, 1. + 1.e-6         ]])
    L_actual = fixed_cholesky(input_matrix)
    assert_allclose(L_expected, L_actual)

def test_jit_cholesky():
    "Tests the stabilized Cholesky decomposition routine"
    
    L_expected = np.array([[2., 0., 0.], [6., 1., 0.], [-8., 5., 3.]])
    input_matrix = np.array([[4., 12., -16.], [12., 37., -43.], [-16., -43., 98.]])
    L_actual, jitter = jit_cholesky(input_matrix)
    assert_allclose(L_expected, L_actual)
    assert_allclose(jitter, 0.)
    
    L_expected = np.array([[1.0000004999998751e+00, 0.0000000000000000e+00, 0.0000000000000000e+00],
                         [9.9999950000037496e-01, 1.4142132088085626e-03, 0.0000000000000000e+00],
                         [6.7379436301144941e-03, 4.7644444411381860e-06, 9.9997779980004420e-01]])
    input_matrix = np.array([[1.                , 1.                , 0.0067379469990855],
                             [1.                , 1.                , 0.0067379469990855],
                             [0.0067379469990855, 0.0067379469990855, 1.                ]])
    L_actual, jitter = jit_cholesky(input_matrix)
    assert_allclose(L_expected, L_actual)
    assert_allclose(jitter, 1.e-6)
    
    L_actual, jitter = cholesky_factor(input_matrix, 0., "adaptive")
    assert_allclose(L_expected, L_actual.L)
    assert_allclose(jitter, 1.e-6)
    
    input_matrix = np.array([[1.e-6, 1., 0.], [1., 1., 1.], [0., 1., 1.e-10]])
    with pytest.raises(linalg.LinAlgError):
        jit_cholesky(input_matrix)

def test_pivot_cholesky():
    "Tests  pivoted cholesky decomposition routine"
    
    input_matrix = np.array([[4., 12., -16.], [12., 37., -43.], [-16., -43., 98.]])
    input_matrix_copy = np.copy(input_matrix)
    L_expected = np.array([[ 9.899494936611665 ,  0.                ,  0.                ],
                           [-4.3436559415745055,  4.258245303082538 ,  0.                ],
                           [-1.616244071283537 ,  1.1693999481734827,  0.1423336335961131]])
    Piv_expected = np.array([2, 1, 0], dtype = np.int32)
    
    L_actual, Piv_actual = pivot_cholesky(input_matrix)
    
    assert_allclose(L_actual, L_expected)
    assert np.array_equal(Piv_expected, Piv_actual)
    assert_allclose(input_matrix, input_matrix_copy)
    
    input_matrix = np.array([[1., 1., 1.e-6], [1., 1., 1.e-6], [1.e-6, 1.e-6, 1.]])
    input_matrix_copy = np.copy(input_matrix)
    L_expected = np.array([[1.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00],
                           [9.9999999999999995e-07, 9.9999999999949996e-01, 0.0000000000000000e+00],
                           [1.0000000000000000e+00, 0.0000000000000000e+00, 3.3333333333316667e-01]])
    Piv_expected = np.array([0, 2, 1], dtype=np.int32)               
    
    L_actual, Piv_actual = pivot_cholesky(input_matrix)
    
    assert_allclose(L_actual, L_expected)
    assert np.array_equal(Piv_expected, Piv_actual)
    assert_allclose(input_matrix, input_matrix_copy)
    
    L_actual, nugget = cholesky_factor(input_matrix, np.array([]), "pivot")
    assert_allclose(L_actual.L, L_expected)
    assert np.array_equal(Piv_expected, L_actual.P)
    assert len(nugget) == 0
    
def test_pivot_transpose():
    "Test function to invert pivot matrix"
    
    P = np.array([0, 2, 1], dtype = np.int32)
    
    Piv = _pivot_transpose(P)
    
    np.array_equal(Piv, P)
    
    P = np.array([1, 2, 1], dtype = np.int32)
    
    with pytest.raises(ValueError):
        _pivot_transpose(P)

    P = np.array([[0, 1, 2], [2, 1, 0]], dtype=np.int32)

    with pytest.raises(AssertionError):
        _pivot_transpose(P)

       
