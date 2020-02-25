from .GaussianProcess import GaussianProcess
from .prediction import default_container


class GaussianProcessFPGA(GaussianProcess):
    def __init__(self, kernel_path, *args):
        super().__init__(*args)

        self.cl_container = default_container(kernel_path)
