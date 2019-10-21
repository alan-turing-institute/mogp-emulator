from .version import version as __version__

from .MultiOutputGP import MultiOutputGP
from .GaussianProcess import GaussianProcess
from .ExperimentalDesign import ExperimentalDesign, MonteCarloDesign, LatinHypercubeDesign
from .SequentialDesign import SequentialDesign, MICEDesign
