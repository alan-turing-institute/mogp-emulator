import numpy as np
from numpy.testing import assert_allclose
import pytest
from ..GaussianProcess import GaussianProcess, PredictResult
from ..MultiOutputGP import MultiOutputGP
from ..validation import mahalanobis, standard_errors, pivoted_errors


def mock_predict(testing, unc=True, deriv=False, include_nugget=True, full_cov=False):
    mean = np.array([1.0, 2.0])
    if full_cov:
        unc = np.array([[0.1, 0.05], [0.05, 0.2]])
    else:
        unc = np.array([0.1, 0.2])
    return PredictResult(mean=mean, unc=unc, deriv=None)


def mock_predict_mogp(
    testing, unc=True, deriv=False, include_nugget=True, full_cov=False, processes=None
):
    mean = np.array([[2.0, 3.0], [1.0, 2.0]])
    if full_cov:
        unc = np.array([[[0.1, 0.05], [0.05, 0.2]], [[0.1, 0.05], [0.05, 0.2]]])
    else:
        unc = np.array([[0.1, 0.2], [0.1, 0.2]])
    return PredictResult(
        mean=mean,
        unc=unc,
        deriv=None,
    )


@pytest.fixture
def valid_inputs():
    return np.array([[1.0], [2.0]])


@pytest.fixture
def valid_targets():
    return np.array([0.5, 2.1])


@pytest.fixture
def valid_targets_mogp():
    return np.array([[1.5, 3.5], [0.5, 2.1]])


def test_standard_errors_GP(valid_inputs, valid_targets, monkeypatch):
    "test standard errors for a GP"

    monkeypatch.setattr("mogp_emulator.GaussianProcess.predict", mock_predict)

    gp = GaussianProcess(valid_inputs, valid_targets, nugget=0.0)

    errors = standard_errors(gp, valid_inputs, valid_targets)

    assert_allclose(errors, [0.5 / np.sqrt(0.1), -0.1 / np.sqrt(0.2)])


def test_standard_errors_MOGP(valid_inputs, valid_targets_mogp, monkeypatch):
    "test standard errors for a MOGP"

    monkeypatch.setattr("mogp_emulator.MultiOutputGP.predict", mock_predict_mogp)

    gp = MultiOutputGP(valid_inputs, valid_targets_mogp, nugget=0.0)

    errors = standard_errors(gp, valid_inputs, valid_targets_mogp)

    assert_allclose(
        errors,
        [
            [0.5 / np.sqrt(0.1), -0.5 / np.sqrt(0.2)],
            [0.5 / np.sqrt(0.1), -0.1 / np.sqrt(0.2)],
        ],
    )


def test_pivoted_errors_GP(valid_inputs, valid_targets, monkeypatch):
    "test correlated errors for a MOGP"
    
    monkeypatch.setattr("mogp_emulator.GaussianProcess.predict", mock_predict)
    
    gp = GaussianProcess(valid_inputs, valid_targets, nugget=0.0)

    errors = pivoted_errors(gp, valid_inputs, valid_targets, undo_pivot=False)
    
    A = np.linalg.cholesky([[0.2, 0.05], [0.05, 0.1]])
    b = np.linalg.solve(A, [-0.1, 0.5])
    
    assert_allclose(errors, b)
    
    errors = pivoted_errors(gp, valid_inputs, valid_targets, undo_pivot=True)
    
    assert_allclose(errors, b[::-1])