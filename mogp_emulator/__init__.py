from .version import version as __version__

from .GaussianProcess import GaussianProcess
from .MultiOutputGP import MultiOutputGP
from .fitting import fit_GP_MAP
from .MeanFunction import MeanFunction
from .ExperimentalDesign import MonteCarloDesign, LatinHypercubeDesign
from .SequentialDesign import MICEDesign
from .HistoryMatching import HistoryMatching
from .DimensionReduction import gKDR
