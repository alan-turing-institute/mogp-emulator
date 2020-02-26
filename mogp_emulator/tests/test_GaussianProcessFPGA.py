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
    x_star = np.array([[1., 3., 2.], [3., 2., 1.]])
    predict_expected = np.array([1.395386477054048, 1.7311400058360489])
    unc_expected = np.array([0.816675395381421, 0.8583559202639046])
    deriv_expected = np.array([0.73471011, -0.0858304,  0.05918638,
                               1.14274266,  0.48175876,  1.52580682])

    def test_expectation(self, example_gpfpga):
        gp = example_gpfpga
        predict_actual, _, _ = gp._predict_single(self.x_star, False, False)

        assert_allclose(predict_actual, self.predict_expected, atol=1.e-8,
                        rtol=1.e-5)

    def test_variance(self, example_gpfpga):
        gp = example_gpfpga
        predict_actual, unc_actual, _ = gp._predict_single(self.x_star, False,
                                                           True)

        assert_allclose(predict_actual, self.predict_expected, atol=1.e-8,
                        rtol=1.e-5)
        assert_allclose(unc_actual, self.unc_expected, atol=1.e-8, rtol=1.e-5)

    def test_deriv(self, example_gpfpga):
        gp = example_gpfpga
        predict_actual, unc_actual, deriv_actual = gp._predict_single(
            self.x_star, True, True
        )

        assert_allclose(predict_actual, self.predict_expected, atol=1.e-8,
                        rtol=1.e-5)
        assert_allclose(unc_actual, self.unc_expected, atol=1.e-8, rtol=1.e-5)
        assert_allclose(deriv_actual, self.deriv_expected, atol=1.e-8,
                        rtol=1.e-5)


@pytest.fixture(scope="class")
def example_gpfpga2(kernel_path):
    x = np.reshape(np.array([1., 2., 3., 2., 4., 1., 4., 2., 2.]), (3, 3))
    y = np.array([2., 3., 4.])

    gp = GaussianProcessFPGA(kernel_path, x, y)

    theta = np.ones(4)
    gp._set_params(theta)

    return gp


@pytest.mark.fpga
class TestPredictSingle2():
    x_star = np.array([4., 0., 2.])
    predict_expected = 0.0174176198731851
    unc_expected = 2.7182302871685224
    deriv_expected = np.array(
        [-8.88648350e-08, 9.46919992e-02, 2.96161460e-08]
    )

    def test_expectation(self, example_gpfpga2):
        gp = example_gpfpga2
        predict_actual, _, _ = gp._predict_single(self.x_star, False, False)

        assert_allclose(predict_actual, self.predict_expected, atol=1.e-8,
                        rtol=1.e-5)

    def test_variance(self, example_gpfpga2):
        gp = example_gpfpga2
        predict_actual, unc_actual, _ = gp._predict_single(self.x_star, False,
                                                           True)

        assert_allclose(predict_actual, self.predict_expected, atol=1.e-8,
                        rtol=1.e-5)
        assert_allclose(unc_actual, self.unc_expected, atol=1.e-8, rtol=1.e-5)

    def test_deriv(self, example_gpfpga2):
        gp = example_gpfpga2
        predict_actual, unc_actual, deriv_actual = gp._predict_single(
            self.x_star, True, True
        )

        assert_allclose(predict_actual, self.predict_expected, atol=1.e-8,
                        rtol=1.e-5)
        assert_allclose(unc_actual, self.unc_expected, atol=1.e-8, rtol=1.e-5)
        assert_allclose(deriv_actual, self.deriv_expected, atol=1.e-8,
                        rtol=1.e-5)
