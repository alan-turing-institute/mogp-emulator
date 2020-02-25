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
