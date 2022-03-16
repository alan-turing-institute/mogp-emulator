import numpy as np
from numpy.testing import assert_allclose
import pytest
from ..GaussianProcess import GaussianProcess, PredictResult
from ..MultiOutputGP import MultiOutputGP
from ..validation import mahalanobis, standard_errors, pivoted_errors

targets1 = np.array([0.5, 2.1, 2.8])
targets2 = np.array([2.5, 2.9, 3.5])
mean1 = np.array([1.0, 2.0, 3.0])
mean2 = np.array([2.0, 3.0, 4.0])
err1 = mean1 - targets1
err2 = mean2 - targets2
unc1 = np.array([[0.1, 0.05, 0.02], [0.05, 0.2, 0.01], [0.02, 0.01, 0.15]])


def mock_predict(testing, unc=True, deriv=False, include_nugget=True, full_cov=False):
    mean = mean1
    if full_cov:
        unc = unc1
    else:
        unc = np.diag(unc1)
    return PredictResult(mean=mean, unc=unc, deriv=None)


def mock_predict_mogp(
    testing, unc=True, deriv=False, include_nugget=True, full_cov=False, processes=None
):
    mean = np.vstack([mean1, mean2])
    if full_cov:
        unc = np.stack([unc1] * 2)
    else:
        unc = np.vstack([np.diag(unc1)] * 2)
    return PredictResult(
        mean=mean,
        unc=unc,
        deriv=None,
    )


@pytest.fixture
def valid_inputs():
    return np.reshape(mean1, (-1, 1))


@pytest.fixture
def valid_targets():
    return targets1


@pytest.fixture
def valid_targets_mogp():
    return np.vstack([targets1, targets2])


def test_standard_errors_GP(valid_inputs, valid_targets, monkeypatch):
    "test standard errors for a GP"

    monkeypatch.setattr("mogp_emulator.GaussianProcess.predict", mock_predict)

    gp = GaussianProcess(valid_inputs, valid_targets, nugget=0.0)

    errors = standard_errors(gp, valid_inputs, valid_targets)

    assert_allclose(errors, err1 / np.sqrt(np.diag(unc1)))


def test_standard_errors_MOGP(valid_inputs, valid_targets_mogp, monkeypatch):
    "test standard errors for a MOGP"

    monkeypatch.setattr("mogp_emulator.MultiOutputGP.predict", mock_predict_mogp)

    gp = MultiOutputGP(valid_inputs, valid_targets_mogp, nugget=0.0)

    errors = standard_errors(gp, valid_inputs, valid_targets_mogp)

    assert_allclose(
        errors,
        [
            err1 / np.sqrt(np.diag(unc1)),
            err2 / np.sqrt(np.diag(unc1)),
        ],
    )


def test_pivoted_errors_GP(valid_inputs, valid_targets, monkeypatch):
    "test correlated errors for a MOGP"

    monkeypatch.setattr("mogp_emulator.GaussianProcess.predict", mock_predict)

    gp = GaussianProcess(valid_inputs, valid_targets, nugget=0.0)

    errors = pivoted_errors(gp, valid_inputs, valid_targets, undo_pivot=False)

    idx1 = np.array([1, 2, 0])
    idx2 = np.array([2, 0, 1])
    A = np.linalg.cholesky(unc1[idx1][:, idx1])
    b = np.linalg.solve(A, err1[idx1])

    assert_allclose(errors, b)

    errors = pivoted_errors(gp, valid_inputs, valid_targets, undo_pivot=True)

    assert_allclose(errors, b[idx2])


def test_pivoted_errors_MOGP(valid_inputs, valid_targets_mogp, monkeypatch):
    "test correlated errors for a MOGP"

    monkeypatch.setattr("mogp_emulator.MultiOutputGP.predict", mock_predict_mogp)

    gp = MultiOutputGP(valid_inputs, valid_targets_mogp, nugget=0.0)

    errors = pivoted_errors(gp, valid_inputs, valid_targets_mogp, undo_pivot=False)

    idx1 = np.array([1, 2, 0])
    idx2 = np.array([2, 0, 1])
    A = np.linalg.cholesky(unc1[idx1][:, idx1])
    b1 = np.linalg.solve(A, err1[idx1])
    b2 = np.linalg.solve(A, err2[idx1])

    assert_allclose(errors, np.vstack([b1, b2]))

    errors = pivoted_errors(gp, valid_inputs, valid_targets_mogp, undo_pivot=True)

    assert_allclose(errors, np.vstack([b1[idx2], b2[idx2]]))
