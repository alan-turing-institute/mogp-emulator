from .version import version as __version__

from .GaussianProcess import GaussianProcess
from .fitting import fit_GP_MLE
from .MeanFunction import MeanFunction
from .ExperimentalDesign import MonteCarloDesign, LatinHypercubeDesign
from .SequentialDesign import MICEDesign
from .HistoryMatching import HistoryMatching
from .DimensionReduction import gKDR
