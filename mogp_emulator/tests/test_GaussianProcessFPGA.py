from ..GaussianProcessFPGA import GaussianProcessFPGA
from ..GaussianProcess import GaussianProcess
import numpy as np
from numpy.testing import assert_allclose
import pathlib
import pytest


@pytest.fixture(scope="session")
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


@pytest.fixture(scope="class")
def example_gpfpga(kernel_path):
    x = np.reshape(np.array([1., 2., 3., 2., 4., 1., 4., 2., 2.]), (3, 3))
    y = np.array([2., 3., 4.])

    gp = GaussianProcessFPGA(kernel_path, x, y)

    theta = np.zeros(4)
    gp._set_params(theta)

    return gp


@pytest.mark.fpga
class TestPredictSingle():
    def test_expectation(self, example_gpfpga):
        gp = example_gpfpga

        x_star = np.array([[1., 3., 2.], [3., 2., 1.]])
        predict_expected = np.array([1.395386477054048, 1.7311400058360489])

        predict_actual, _, _ = gp._predict_single(x_star, False, False)

        assert_allclose(predict_actual, predict_expected)

    def test_variance(self, example_gpfpga):
        gp = example_gpfpga

        x_star = np.array([[1., 3., 2.], [3., 2., 1.]])
        predict_expected = np.array([1.395386477054048, 1.7311400058360489])
        unc_expected = np.array([0.816675395381421, 0.8583559202639046])

        predict_actual, unc_actual, _ = gp._predict_single(x_star, False, True)

        assert_allclose(predict_actual, predict_expected)
        assert_allclose(unc_actual, unc_expected)

    def test_deriv(self, example_gpfpga):
        gp = example_gpfpga

        x_star = np.array([[1., 3., 2.], [3., 2., 1.]])
        predict_expected = np.array([1.395386477054048, 1.7311400058360489])
        unc_expected = np.array([0.816675395381421, 0.8583559202639046])
        deriv_expected = np.array([0.73471011, -0.0858304,  0.05918638,
                                   1.14274266,  0.48175876,  1.52580682])

        predict_actual, unc_actual, deriv_actual = gp._predict_single(
            x_star, True, True
        )

        assert_allclose(predict_actual, predict_expected)
        assert_allclose(unc_actual, unc_expected)
        assert_allclose(deriv_actual, deriv_expected, rtol=1e-06)
