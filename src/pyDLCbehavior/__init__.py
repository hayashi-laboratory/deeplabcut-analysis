import pkg_resources

__version__ = pkg_resources.get_distribution("pyDLCbehavior").version
from .analysis import YMazeAnalysis, NovelObjectRecognitionAnalysis
from .utility import setrois, glob_files, Roi
from .ymaze import YMazeScaler, BasicYMazeCollection, ArmCollection, ArmRegion
from .dataset import DLCDataset


__all__ = [
    "YMazeAnalysis",
    "NovelObjectRecognitionAnalysis",
    "setrois",
    "glob_files",
    "Roi",
    "YMazeScaler",
    "BasicYMazeCollection",
    "ArmCollection",
    "ArmRegion",
    "DLCDataset",
]
