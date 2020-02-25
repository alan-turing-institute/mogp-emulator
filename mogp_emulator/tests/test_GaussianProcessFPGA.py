from ..GaussianProcessFPGA import GaussianProcessFPGA
from ..GaussianProcess import GaussianProcess
import numpy as np
from numpy.testing import assert_allclose
import pathlib
import pytest


@pytest.fixture()
def kernel_path():
    """
    Fixture pointing to a compiled FPGA kernel.

    To run the FPGA tests, compile the FPGA kernels for your accelerator and
    point the `path` variable in this fixture to your compiled kernel.
    """
    path = pathlib.Path(__file__).parent.absolute() / "kernel/prediction.aocx"
    return str(path)


@pytest.mark.fpga
class TestInit():
    def test_consistency(self, kernel_path):
        x = np.reshape(np.array([1., 2., 3.]), (1, 3))
        y = np.array([2.])

        gp = GaussianProcess(x, y)
        gpfpga = GaussianProcessFPGA(kernel_path, x, y)

        assert_allclose(gpfpga.inputs, gp.inputs)
        assert_allclose(gpfpga.targets, gp.targets)
        assert gpfpga.D == gp.D == 3
        assert gpfpga.n == gp.n == 1
        assert gpfpga.nugget == gp.nugget is None


@pytest.mark.fpga
class TestPredictSingle():
    def test_expectation(self, kernel_path):
        x = np.reshape(np.array([1., 2., 3., 2., 4., 1., 4., 2., 2.]), (3, 3))
        y = np.array([2., 3., 4.])

        gp = GaussianProcessFPGA(kernel_path, x, y)

        theta = np.zeros(4)
        gp._set_params(theta)

        x_star = np.array([[1., 3., 2.], [3., 2., 1.]])
        predict_expected = np.array([1.395386477054048, 1.7311400058360489])
        unc_expected = np.array([0.816675395381421, 0.8583559202639046])

        predict_actual, _, _ = gp._predict_single(x_star, False, False)

        assert_allclose(predict_actual, predict_expected)

    def test_variance(self, kernel_path):
        x = np.reshape(np.array([1., 2., 3., 2., 4., 1., 4., 2., 2.]), (3, 3))
        y = np.array([2., 3., 4.])

        gp = GaussianProcessFPGA(kernel_path, x, y)

        theta = np.zeros(4)
        gp._set_params(theta)

        x_star = np.array([[1., 3., 2.], [3., 2., 1.]])
        predict_expected = np.array([1.395386477054048, 1.7311400058360489])
        unc_expected = np.array([0.816675395381421, 0.8583559202639046])

        predict_actual, unc_actual, _ = gp._predict_single(x_star, False, True)

        assert_allclose(predict_actual, predict_expected)
        assert_allclose(unc_actual, unc_expected)
