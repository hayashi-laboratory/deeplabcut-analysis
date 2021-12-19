from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Any, Mapping, Union, NamedTuple
from datetime import timedelta
import pandas as pd


class FrameDimension(NamedTuple):
    width: int = -1
    height: int = -1


@dataclass
class DLCDataset:
    # file path
    csv_path: Path
    pkl_path: Path
    video_path: Path = field(default="")
    homedir: Path = field(init=False)

    # pickle data
    hyperparams: Dict[str, Any] = field(init=False, repr=False, default=None)
    dlc_model_config: Dict[str, Any] = field(init=False, repr=False, default=None)
    # video parameters
    FPS: float = field(init=False, default=None)
    frame_dimensions: FrameDimension = field(init=False, default_factory=FrameDimension)

    # raw_data
    raw_data: pd.DataFrame = field(init=False, repr=False, default=None)

    def __post_init__(self):
        # convert all path to patlib.Path
        self.csv_path = Path(self.csv_path)
        if not self.csv_path.exists():
            raise FileNotFoundError(self.csv_path)
        self.pkl_path = Path(self.pkl_path)
        if not self.pkl_path.exists():
            raise FileNotFoundError(self.pkl_path)
        self.video_path = Path(self.video_path)
        self.homedir = self.csv_path.parent

        # load pickle file that generate from DeepLabCut
        data: dict = pd.read_pickle(self.pkl_path).get("data", dict())

        if "DLC-model-config file" in data:
            self.dlc_model_config = data["DLC-model-config file"]
            del data["DLC-model-config file"]

        # load the video data from data
        self.FPS = data.get("fps")

        height, width = data.get("frame_dimensions", (-1, -1))

        self.frame_dimensions = FrameDimension(width, height)
        self.hyperparams = data
        self.load_csv()

    def __setstate__(self, d: Mapping[str, Any]) -> None:
        """A function that called when unpickling the NorDlcAnalysis class

        Args:
            d (Mapping[str, int]): data to be unpickled
        """

        # convert all path and file dir into pathlib.Path
        for key, val in d.items():
            if not isinstance(val, str):
                continue
            if key.lower().endswith(("path", "dir")):
                d[key] = Path(val)
        # convert the string path to pathlib.Path
        self.__dict__.update(d)

    def __getstate__(self) -> Dict[str, Any]:
        """Called when pickleing the DLC Dataset

        Returns:
            Dict[str, int]: dict to be pickled
        """
        return dict(
            csv_path=str(self.csv_path),
            pkl_path=str(self.pkl_path),
            video_path=str(self.video_path),
            homedir=str(self.homedir),
            hyperparams=self.hyperparams,
            dlc_model_config=self.dlc_model_config,
            FPS=self.FPS,
            frame_dimensions=self.frame_dimensions,
            raw_data=self.raw_data,
        )

    def load_csv(self) -> None:
        """Load the csv file and add a timestamp index"""
        # load data
        if self.csv_path.suffix in [".xlsx", ".xls"]:
            raw = pd.read_excel(self.csv_path, header=[0, 1, 2], index_col=0)
        else:
            raw = pd.read_csv(self.csv_path, header=[0, 1, 2], index_col=0)
        # Add timestamp index to raw data
        frame_interval = timedelta(milliseconds=1e3 / self.FPS)
        raw.index = raw.index * frame_interval
        self.raw_data = raw

    def to_pickle(self, pickle_path=None) -> Path:
        if pickle_path is None:
            pickle_path = self.homedir.joinpath(f"{self.csv_path.stem}_analysis.pkl.gz")
        else:
            pickle_path = Path(pickle_path)

        if pickle_path.suffix != ".gz":
            pickle_path = pickle_path.with_name(pickle_path.name + ".gz")

        print("Saving...")
        pd.to_pickle(self, pickle_path)
        print(f"Compressed pickle was saved at \033[92m{pickle_path}\033[0m.")
        return pickle_path

    @staticmethod
    def from_pickle(pickle_path: Union[str, Path]):
        if not Path(pickle_path).exists():
            raise FileNotFoundError(str(pickle_path))
        return pd.read_pickle(Path(pickle_path))
