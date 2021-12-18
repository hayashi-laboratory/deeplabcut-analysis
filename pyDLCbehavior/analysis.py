from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Mapping, OrderedDict

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage

from .dataset import DLCDataset
from .utility import Roi, setrois
from .ymaze import ArmCollection, ArmRegion, BasicYMazeCollection, YMazeScaler


@dataclass
class NovelObjectRecognitionAnalysis(DLCDataset):

    # ROI
    objects: List[Roi] = field(init=False, default_factory=list)

    # raw_data
    raw_data: pd.DataFrame = field(init=False, repr=False, default=None)
    scale_x: float = field(init=False, default=1)
    scale_y: float = field(init=False, default=1)

    # data after analysis
    nose2obj: int = field(init=False, default=None)
    offset: int = field(init=False, default=None)
    data: pd.DataFrame = field(init=False, repr=False, default=None)
    filter_data: pd.DataFrame = field(init=False, repr=False, default=None)
    climbing_filter: pd.DataFrame = field(init=False, repr=False, default=None)

    # the summary from center trajectory
    locomotion_data: pd.DataFrame = field(init=False, repr=False, default=None)
    total_distance: float = field(init=False, default=None)
    mean_speed: float = field(init=False, default=None)

    def __post_init__(self) -> None:
        # parse the path and load the pickle file
        super().__post_init__()
        self.preprocess()

    def __setstate__(self, d: Mapping[str, Any]) -> None:
        super().__setstate__(d)
        self.analyze()

    def __getstate__(self) -> Dict[str, Any]:
        d = super().__getstate__()
        d.update(
            {
                "objects": self.objects,
                "nose2obj": self.nose2obj,
                "offset": self.offset,
            }
        )
        return d

    @property
    def num_of_objects(self) -> int:
        return len(self.objects)

    @property
    def object_a(self) -> Roi:
        for o in self.objects:
            if o.name.lower().endswith("a"):
                return o
        raise AttributeError()

    @property
    def object_b(self) -> Roi:
        for o in self.objects:
            if o.name.lower().endswith("b"):
                return o
        raise AttributeError()

    def select_objects(self, num_of_objects: int = 2) -> None:
        """select the roi object manually.

        Args:
            num_of_objects (int, optional): How many object to be selected. Defaults to 2.
        """
        self.objects = setrois(self.video_path, num_of_objects)

    def add_object(self, obj: Roi) -> None:
        """Add a Roi object manually.

        Args:
            obj (Roi): A Roi instance contains (x,y, width, height, name)

        Raises:
            TypeError: Raise error if input is not Roi instance
        """
        if not isinstance(obj, Roi):
            raise TypeError("obj should be Roi instance")
        self.objects.append(obj)

    def preprocess(self) -> None:
        """filter the raw data by moving averaging"""
        # drop the scorer columns
        raw = self.raw_data.droplevel(0, axis=1)

        # Rolling the data by 5 seconds to get the reliable start time.
        # Using nose as target to calculate the rolling mean of likelihood
        # Get the index when mean of likelihood is larger than 0.99
        # default is the first items
        all_joints_names = self.dlc_model_config["all_joints_names"]
        bodypart = all_joints_names[0]
        if "nose" in all_joints_names:
            bodypart = "nose"
        elif "center" in all_joints_names:
            bodypart = "center"

        rolling_window = "5s"
        df_MVA = raw[bodypart, "likelihood"].rolling(rolling_window).mean()
        start = df_MVA[df_MVA > 0.99].index.min()
        # Time window for 10 minutes
        end = start + timedelta(minutes=10)
        raw = raw[start:end]
        # set data to class instance
        self.raw_data = raw

    def analyze(self, nose2obj: int = 4, offset: int = 15) -> None:
        if self.num_of_objects == 0:
            print("\033]91mPlease select objects before analyze\033]0m")
            return

        self.nose2obj = self.nose2obj or nose2obj
        self.offset = self.offset or offset
        # copy the data from raw_data
        data = self.raw_data.copy()

        # a convinient index for slice the multi-index
        idx = pd.IndexSlice

        # get body parts
        part_names = data.columns.get_level_values(0).unique()
        object_names = [c for c in part_names if "obj" in c.lower()]
        bodyparts = [c for c in part_names if "obj" not in c.lower()]

        # temp class that decide the distance to ROI and predict coordinates
        from collections import namedtuple

        Coords = namedtuple("Coords", ["name", "coords"])
        get_coord = lambda o: Coords(o, (data[o, "x"].mean(), data[o, "y"].mean()))
        # list of Coords(name = name, coords = (x, y))
        dlc_objs = [get_coord(o) for o in object_names]

        for roi_obj in self.objects:
            # automatic assign the object name by distance
            name = min(dlc_objs, key=lambda x: roi_obj.distance(*x.coords)).name
            if not roi_obj.name:
                roi_obj.name = f"ROI_{name}"
            data.loc[:, (roi_obj.name, "x")] = roi_obj.x
            data.loc[:, (roi_obj.name, "y")] = roi_obj.y

        part_names = data.columns.get_level_values(0).unique()
        # object real distance (cm)
        real_x_cm, real_y_cm = 20, 20
        scale_x = real_x_cm / abs(self.objects[0].x - self.objects[1].x)
        scale_y = real_y_cm / abs(self.objects[0].y - self.objects[1].y)
        self.scale_x, self.scale_y = scale_x, scale_y
        # scale all coordinates to real distance (cm)
        data.loc[:, idx[:, "x"]] = data.loc[:, idx[:, "x"]] * scale_x
        data.loc[:, idx[:, "y"]] = data.loc[:, idx[:, "y"]] * scale_y

        # Get the distance between each bodypart and object
        for roi_obj in self.objects:
            # create the new column names for assign the distance
            new_cols = pd.MultiIndex.from_tuples(
                [(b, f"distance_to_{roi_obj.name}") for b in bodyparts],
                names=["bodyparts", "coords"],
            )
            scale_obj_x = roi_obj.x * scale_x
            scale_obj_y = roi_obj.y * scale_y
            data[new_cols] = np.sqrt(
                np.square(data.loc[:, idx[bodyparts, "x"]].values - scale_obj_x)
                + np.square(data.loc[:, idx[bodyparts, "y"]].values - scale_obj_y)
            )
        data = data[part_names]

        # calculate the distance from nose to center
        data["nose2center"] = np.sqrt(
            np.square(np.diff(data.loc[:, idx[["nose", "center"], "x"]]))
            + np.square(np.diff(data.loc[:, idx[["nose", "center"], "y"]]))
        )
        self.data = data

        # first filter all data by 15cm
        # offset default is 15 cm
        filter_data = data[data["nose2center"] < offset]

        # took the center position
        center = (
            filter_data.loc[:, idx["center", ["x", "y"]]].droplevel(0, axis=1).copy()
        )
        center["length"] = np.sqrt(np.square(center.diff()).sum(axis=1))
        center["distance"] = center["length"].cumsum()
        # delta time
        dt = center.index[1:] - center.index[:-1]
        # convert nsdeltatime to seconds
        dt = dt.to_numpy().astype(float) / 1e9
        center["speed"] = np.nan
        center["speed"].iloc[1:] = center["length"].iloc[1:] / dt
        self.locomotion_data = center
        self.total_distance = center["length"].sum()
        self.mean_speed = center["speed"].mean()

        for roi_obj in self.objects:
            name = roi_obj.name
            mask = (filter_data["nose", f"distance_to_{name}"] < nose2obj) & ~(
                filter_data["center", f"distance_to_{name}"].isnull()
            )
            count = mask.sum()
            roi_obj.data = {"frame_count": count, "duration": count / self.FPS}

        self.filter_data = filter_data.copy()

        # TODO
        # this part is filter for ploting
        # filter the climbing data
        # select the nose is < 4 cm and center  > 4 cm
        filter_func = lambda roi_obj: (
            (filter_data[f"nose", f"distance_to_{roi_obj.name}"] < nose2obj)
            & (filter_data["center", f"distance_to_{roi_obj.name}"] > nose2obj)
        )
        # condition fit object A or object B
        distance_mask = filter_func(self.objects[0]) | filter_func(self.objects[1])

        filter_data = filter_data[distance_mask]

        self.climbing_filter = filter_data.copy()

    def __plot_objects(self, ax):
        ax.scatter(
            self.object_a.x * self.scale_x,
            self.object_a.y * self.scale_y,
            s=600,
            c="#ff2251",
            alpha=0.5,
        )

        ax.scatter(
            self.object_b.x * self.scale_x,
            self.object_b.y * self.scale_y,
            s=600,
            c="#00b48c",
            alpha=0.5,
        )

    def plot_trajectory(self):
        nose = self.filter_data.nose
        center = self.filter_data.center

        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111)
        ax.plot(nose.x.values, nose.y.values, alpha=0.75)
        ax.plot(center.x.values, center.y.values, alpha=0.75)
        self.__plot_objects(ax)
        ax.set_ylim(40, 0)
        ax.set_xlim(0, 40)
        # fig.set_title(f"file: {self.csv_path.stem}")
        fig.tight_layout()
        fig.savefig(
            self.homedir.joinpath(f"{self.csv_path.stem}_scatter.png"),
            transparent=True,
        )
        return fig

    def plot_filter_scatter(self):
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111)

        nose = self.climbing_filter.nose
        center = self.climbing_filter.center
        ax.scatter(nose.x.values, nose.y.values, alpha=0.4)
        ax.scatter(center.x.values, center.y.values, alpha=0.4)
        self.__plot_objects(ax)
        ax.set_xlim(0, 40)
        ax.set_ylim(40, 0)
        fig.tight_layout()
        fig.savefig(
            self.homedir.joinpath(f"{self.csv_path.stem}_filtered.png"),
            transparent=True,
        )
        return fig

    def plot_heatmap(self):
        from matplotlib import colors
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        nose_np = self.climbing_filter.nose[["x", "y"]].values
        nose_np_y = nose_np[:, 1]
        nose_np_x = nose_np[:, 0]
        # make histrogram2d
        counts, y, x = np.histogram2d(
            nose_np_y,
            nose_np_x,
            self.frame_dimensions,
        )

        # Filter by Gaussian_filter from scipy
        counts = ndimage.gaussian_filter(counts, sigma=8)

        # Heatmap with customized colormap
        fig = plt.figure(figsize=(5, 5))

        ax = fig.add_subplot(111)
        norm = colors.LogNorm(vmin=np.nanmin(counts), vmax=np.nanmax(counts))
        cmap = colors.LinearSegmentedColormap.from_list(
            "custom",
            ["grey", "#edf8e9", "#74c476", "#006d2c"],
            N=64,
        )
        # colmap = mpl.cm.ScalarMappable(norm = norm, cmap = cmap)
        im = ax.imshow(counts, cmap=cmap)
        # create an axes on the right side of ax. The width of cax will be 5%
        # of ax and the padding between cax and ax will be fixed at 0.05 inch.
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        fig.colorbar(im, cax=cax)

        fig.tight_layout()
        fig.savefig(
            self.homedir.joinpath(f"{self.csv_path.stem}_heatmap.png"),
            transparent=True,
        )
        return fig

    @property
    def summary_df(self) -> pd.DataFrame:
        z1 = self.object_a.data["frame_count"]
        z2 = self.object_b.data["frame_count"]
        total = z1 + z2
        return pd.DataFrame.from_dict(
            OrderedDict(
                filename=[self.csv_path.stem],
                FrameNumber_exploration_Zone1=[z1],
                FrameNumber_exploration_Zone2=[z2],
                video_fps=[self.FPS],
                Time_exploration_Zone1=[self.object_a.data["duration"]],
                Time_exploration_Zone2=[self.object_b.data["duration"]],
                Zone1_x=[self.object_a.x * self.scale_x],
                Zone1_y=[self.object_a.y * self.scale_y],
                Zone2_x=[self.object_a.x * self.scale_x],
                Zone2_y=[self.object_a.y * self.scale_y],
                Zone1_coord=[(self.object_a.x, self.object_a.y)],
                Zone2_coord=[(self.object_b.x, self.object_b.y)],
                video_shape_w=[self.frame_dimensions[0]],
                video_shape_h=[self.frame_dimensions[0]],
                Discrimination_index_Zone1_to_Zone2=[(z1 - z2) / total],
                Discrimination_index_Zone2_to_Zone1=[(z2 - z1) / total],
                total_distance=[self.total_distance],
                mean_speed=[self.mean_speed],
            )
        )


@dataclass
class YMazeAnalysis(DLCDataset):
    # analysis time window
    time: int = 8
    savedir: Path = field(init=False)

    # scale parameters
    scale_x: float = field(init=False, default=1)
    scale_y: float = field(init=False, default=1)

    # data after analysis
    labeled_data: pd.DataFrame = field(init=False, repr=False, default=None)
    arms: ArmCollection = field(init=False, default_factory=ArmCollection)
    ymaze_center: np.ndarray = field(init=False, repr=False, default=None)
    alternation_data: Dict[str, Any] = field(
        init=False, repr=False, default_factory=dict
    )

    # the summary from center trajectory
    locomotion_data: pd.DataFrame = field(init=False, repr=False, default=None)
    total_distance: float = field(init=False, default=None)
    mean_speed: float = field(init=False, default=None)

    def __post_init__(self) -> None:
        super().__post_init__()
        self.savedir = self.homedir.joinpath("save", f"{self.time}min")
        if not self.savedir.exists():
            self.savedir.mkdir(parents=True)
        self.preprocess()

    def __setstate__(self, d: Mapping[str, Any]) -> None:
        super().__setstate__(d)
        self.savedir = self.homedir.joinpath("save", f"{self.time}min")
        self.analyze()

    def __getstate__(self) -> Dict[str, Any]:
        d = super().__getstate__()
        d.update({"time": self.time, "arms": self.arms})
        return d

    def preprocess(self) -> None:
        raw = self.raw_data.droplevel(0, axis=1)
        # Rolling the data by 5 seconds to get the reliable start time.
        # Using nose as target to calculate the rolling mean of likelihood
        # Get the index when mean of likelihood is larger than 0.99
        # default is the first items
        rolling_window = "3s"
        nose_MVA = raw["Nose", "likelihood"].rolling(rolling_window).mean()
        withers_MVA = raw["Withers", "likelihood"].rolling(rolling_window).mean()
        start = raw[(nose_MVA > 0.95) & (withers_MVA > 0.95)].index.min()
        # Time window for 8 minutes
        end = start + timedelta(minutes=self.time)
        raw = raw[start:end]
        # set filter data to raw_data
        self.raw_data = raw

    def analyze(self):
        # copy data
        data = self.raw_data.copy()
        # index slice
        idx = pd.IndexSlice

        # filter the data by distance from nose to withers
        nose2wither = np.sqrt(
            np.square(np.diff(data.loc[:, idx[["Nose", "Withers"], "x"]]))
            + np.square(np.diff(data.loc[:, idx[["Nose", "Withers"], "y"]]))
        )
        data = data[nose2wither < np.square(100)]

        arm_tags = ["A", "B", "C"]
        ref_pt = []
        #temp_arms = ArmCollection()
        for a in arm_tags:
            name = [f"Arm{a}_{c+1}" for c in range(4)]
            mask = data.loc[:, idx[name, "likelihood"]] > 0.98
            xy = data.loc[mask.values, idx[name, ["x","y"]]].median()
            arm_center = xy.loc[idx[name, ["x"]]].mean(), xy.loc[idx[name, ["y"]]].mean()
            ref_pt.append(arm_center)

        # fit the abstract arm by DeepLabCut coordinates
        arms: ArmCollection = BasicYMazeCollection()
        arms.fit(ref_pt)

        # mean pos_x, pos_y
        middle_body_pos = (data.Nose[["x", "y"]] + data.Withers[["x", "y"]]) / 2

        data["mask"] = "undefined"
        for arm in arms.values():
            mask = middle_body_pos.apply(lambda pt: arm.contains(pt.x, pt.y), axis=1)
            data.loc[mask, "mask"] = arm.name

        ymaze_center = arms.center
        calc_dist2center = lambda pt: np.square(pt - ymaze_center).sum() < np.square(
            100
        )
        mask = middle_body_pos.apply(calc_dist2center, axis=1, raw=True)
        data.loc[mask, "mask"] = "o"

        ###################################################
        # Get the property of locomotor activity.
        # Length x between center and mean point of arm1 and 2 is around 40 cm * sqrt(3) / 2
        # Lenght y between center and mean point of arm1 and 2 is around 40 cm * 1 / 2
        # Total distance is calculated following ARIMA model (moving average)
        ###################################################

        xdiff, ydiff = np.absolute(np.diff([ymaze_center, arms["A"].center], axis=0)[0])

        scale_x, scale_y = 40 * 0.5 * np.sqrt(3) / xdiff, 40 * 0.5 / ydiff

        withers = data.loc[:, idx["Withers", ["x", "y"]]].droplevel(0, axis=1).copy()

        # Set the rolling time window as 5
        rolling_window = "5s"
        withers = withers.rolling(rolling_window).mean().dropna()  # Moving average

        scaler = YMazeScaler([arms["A"].center, arms["B"].center, arms["C"].center])

        withers[["x", "y"]] = scaler.scale(withers.values)
        # withers["x"] = withers["x"] * scale_x
        # withers["y"] = withers["y"] * scale_y

        withers["length"] = np.sqrt(np.square(withers.diff()).sum(axis=1))
        withers["distance"] = withers["length"].cumsum()
        # delta time
        dt = withers.index[1:] - withers.index[:-1]
        # convert nsdeltatime to seconds
        dt = dt.to_numpy().astype(float) / 1e9
        withers["speed"] = np.nan
        withers["speed"].iloc[1:] = withers["length"].iloc[1:] / dt

        self.locomotion_data = withers
        self.total_distance = withers["length"].sum()
        self.mean_speed = withers["speed"].mean()

        self.ymaze_center = ymaze_center
        self.arms = arms
        self.labeled_data = data

        # analysis the alternation
        self.analyze_alternation()

    def analyze_alternation(self):
        data = self.labeled_data
        labels = data["mask"].copy()
        start = labels.isin(["A", "B", "C", "o"]).index.min()
        labels = labels.loc[start:]
        labels[~labels.isin(["A", "B", "C", "o"])] = np.nan
        labels = labels.ffill()
        labels.index.name = "time"
        onset = (
            labels[labels != labels.shift(1)]
            .reset_index()
            .rename(columns={"mask": "onset", "time": "onset_time"})
        )
        offset = (
            labels[labels != labels.shift(-1)]
            .reset_index()
            .rename(columns={"mask": "offset", "time": "offset_time"})
        )

        alt_df = pd.concat([onset, offset], axis=1)
        alt_df["duration"] = alt_df["offset_time"] - alt_df["onset_time"]
        total_row = len(alt_df)
        num_of_arms = (alt_df.onset != "o").sum()
        #### Fetch the alternation, which is the count of arm entries that a mouse does not entry in the same arm over 3 entries.
        #### Compared to next 2 entries.
        arm_label = alt_df[alt_df.onset != "o"].onset

        alt_mask = (
            (arm_label != arm_label.shift(-1))
            & (arm_label != arm_label.shift(-2))
            & (arm_label.shift(-1) != arm_label.shift(-2))
        )
        alt = alt_mask.sum()
        # Alternation ratio calculation
        alternation_ratio = alt / (num_of_arms - 2) * 100
        alternation_ratio = np.round(alternation_ratio, 3)

        # Store each varibale in alt_dict
        alternation_data = {
            "spontaneous_alternation": alt,  # Spontaneous_alternation
            "alternation_ratio": alternation_ratio,  # Alternation_ratio
            "arm_entries": num_of_arms,  # Arm_entries
            "alternation_order": alt_df,
        }
        self.alternation_data = alternation_data

    def plot_ymaze(self) -> mpl.figure.Figure:
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111)
        data = self.labeled_data
        withers = data["Withers"]
        ax.plot(withers.x, withers.y, alpha=0.2)

        import re
        from itertools import cycle

        colors = str(plt.rcParams["axes.prop_cycle"])
        colors = cycle(
            map(lambda s: s.strip(" '"), re.findall(r"'#.*'", colors)[0].split(","))
        )
        [next(colors) for _ in range(2)]

        for arm, c in zip(self.arms.values(), colors):
            ax.text(*arm.center, arm.name, size=12)
            arm.draw_area(ax, facecolor=c + "70", zorder=2)
        ax.scatter(*self.ymaze_center, c="red")

        for label in ["o", "A", "B", "C"]:
            filter_withers = data.loc[data["mask"] == label, "Withers"]
            ax.plot(filter_withers.x, filter_withers.y)

        self.arms.draw_ymaze_contour(ax, facecolor="0.9", edgecolor="0.5")

        width, height = self.frame_dimensions
        ax.set_xlim(0, width)
        ax.set_ylim(height, 0)
        filename = self.csv_path.stem.split("_")[0]
        ax.set_title(f"Ymaze appearance {filename}")
        fig.tight_layout()
        fig.savefig(
            self.savedir.joinpath(f"{filename}_Ymaze_apperance_{self.time}.png")
        )
        return fig

    def plot_alternations(self) -> mpl.figure.Figure:
        data = self.labeled_data
        labels = data["mask"].copy()
        start = labels.isin(["A", "B", "C", "o"]).index.min()
        labels = labels.loc[start:]
        labels[~labels.isin(["A", "B", "C", "o"])] = np.nan
        labels = labels.ffill()

        fig = plt.figure()
        ax = fig.add_subplot(111)

        from collections import defaultdict

        dic = defaultdict(lambda: 5, A=1, B=2, C=3, o=4)

        label_nums = labels.replace(dic)
        ax.plot(label_nums)
        ax.plot(label_nums.diff())
        # set labels
        group_labels = ["A", "B", "C", "Center", "Others"]
        ax.set_yticks([1, 2, 3, 4, 5])
        ax.set_yticklabels(group_labels)
        ax.set_ylim(0, 6)
        filename = self.csv_path.stem.split("_")[0]
        ax.set_title(f"Ymaze appearance {filename}")
        fig.tight_layout()
        fig.savefig(
            self.savedir.joinpath(f"{filename}_Ymaze_alternation_{self.time}min.png")
        )
        return fig

    @property
    def summary_df(self) -> pd.DataFrame:
        return pd.DataFrame.from_dict(
            OrderedDict(
                filename=[self.csv_path.stem],
                video_shape=[self.frame_dimensions],
                video_fps=[self.FPS],
                arm_entries=[self.alternation_data["arm_entries"]],
                spontaneous_alternation=[
                    self.alternation_data["spontaneous_alternation"]
                ],
                total_distance=[self.total_distance],
                mean_speed=[self.mean_speed],
            )
        )


if __name__ == "__main__":
    pkl = "data/YMAZE/00228_DLC_resnet50_Ymaze_project-DLC-2020-10-13Oct13shuffle1_140000_meta.pickle"
    csv = "data/YMAZE/00228_DLC_resnet50_Ymaze_project-DLC-2020-10-13Oct13shuffle1_140000.csv"
    ymaze = YMazeAnalysis(csv, pkl)
    ymaze.analyze()
    ymaze.plot_ymaze()
    ymaze.plot_alternations()
    pickle_path = ymaze.to_pickle()
    ymaze = YMazeAnalysis.from_pickle(pickle_path)
