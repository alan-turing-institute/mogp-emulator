from .version import version as __version__

from .GaussianProcess import GaussianProcess
from .GaussianProcessGPU import GaussianProcessGPU
from .GaussianProcessGPU import UnavailableError
from .ExperimentalDesign import ExperimentalDesign, MonteCarloDesign, LatinHypercubeDesign
from .SequentialDesign import SequentialDesign, MICEDesign
from .MultiOutputGP import MultiOutputGP
from .fitting import fit_GP_MAP
from .MeanFunction import MeanFunction
from .HistoryMatching import HistoryMatching
from .DimensionReduction import gKDR
