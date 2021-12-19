from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, OrderedDict

import cv2
from numpy import ndarray

__all__ = ["Roi", "setrois", "glob_files"]


@dataclass
class Roi:
    """A basic class that store the region of interests in pixel value"""

    order: int = -1
    x: int = -1
    y: int = -1
    width: int = -1
    height: int = -1
    name: str = ""
    # save the data from NorDlcAnalysis
    data: Dict = field(init=False, default_factory=dict)

    def to_dict(self) -> OrderedDict[str, int]:
        """Get the dict of all attributes

        Returns:
            OrderDict[str, int]: dict of parameters
        """
        return OrderedDict(
            order=self.order,
            x=self.x,
            y=self.y,
            width=self.width,
            height=self.height,
            name=self.name,
            data=self.data,
        )

    def __setstate__(self, d: Mapping[str, int]) -> None:
        """A function that called when unpickling the Roi class

        Args:
            d (Mapping[str, int]): data to be unpickled
        """
        self.__dict__.update(d)

    def __getstate__(self) -> Mapping[str, int]:
        """Called when pickleind the Roi

        Returns:
            Mapping[str, int]: dict to be pickled
        """
        return self.to_dict()

    def to_json_str(self) -> str:
        """Get the json serialized string

        Returns:
            str: json serialized string
        """
        import json

        return json.dumps(self.to_dict())

    def draw(self, src: ndarray) -> ndarray:
        """Draw the ROI rectangle and label on the source image

        Args:
            src (ndarray): source image to be draw (8U_C1, 8U_C3)

        Returns:
            ndarray: labeled image
        """
        if not isinstance(src, ndarray):
            raise TypeError("src is not a ndarray")
        color = (0, 255, 255)
        pt1 = (self.x, self.y)
        pt2 = (self.x + self.width, self.y + self.height)
        cv2.putText(
            src,
            f"Roi{self.order}",
            (self.x, self.y - 3),
            cv2.FONT_HERSHEY_DUPLEX,
            0.7,
            color,
            thickness=2,
        )
        return cv2.rectangle(src, pt1, pt2, color, thickness=2)

    def distance(self, x: int, y: int) -> int:
        """Calculate the euclidean distance of x, y coordinate to Roi (pixel-wise)

        Args:
            x (int): x coordinate (pixel position)
            y (int): y coordinate (pixel position)

        Returns:
            int: the euclidean distance
        """
        from math import floor, pow, sqrt

        return floor(sqrt(pow(self.x - x, 2) + pow(self.y - y, 2)))


# define all functions used in this script
def glob_files(parent: Path) -> Dict[str, List[Path]]:
    """Glob files in all sub-directories and sorted by suffix

    Args:
        path (Path): parent directory to be glob

    Returns:
        Dict[str, List[Path]]: A dict collection with suffix as key
    """

    path_dict = {}
    for file in Path(parent).glob("**/*"):
        if not file.is_file() or file.name.startswith("."):
            continue
        key = file.suffix
        path_dict[key] = path_dict.get(key, list())
        path_dict[key].append(file)
    return path_dict


def setrois(videopath: str, num_of_objects: int) -> List["Roi"]:
    """Manual selection region of interest from video

    Args:
        videopath (str): video to be selected
        num_of_objects (int): number of ROI to be selected

    Raises:
        FileNotFoundError: File is not available
        FileNotFoundError: Opencv fail to open the video
        KeyboardInterrupt: Roi selection was cancelled

    Returns:
        List[Roi]: A list of Roi instance.
    """
    from IPython.display import clear_output

    if not isinstance(videopath, Path):
        videopath = Path(videopath)

    if not (videopath.exists() and videopath.is_file()):
        raise FileNotFoundError(
            f"{videopath.name} is not existed. Please check your videopath"
        )

    cap = cv2.VideoCapture(str(videopath))

    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open the video {videopath.name}")

    while True:
        ok, frame = cap.read()
        if ok:
            break
    cap.release()

    windowname = videopath.name.split("_")[1]
    rois = []

    cv2.namedWindow(windowname)
    cv2.startWindowThread()
    try:
        for i in range(num_of_objects):
            print(f"\033[92m[SYSTEM] Please select Zone-{i}\033[0m")
            x, y, width, height = cv2.selectROI(windowname, frame, False, False)
            cv2.waitKey(1)
            if x * y * width * height == 0:
                break
            roi = Roi(i, x, y, width, height)
            frame = roi.draw(frame)
            rois.append(roi)
            clear_output()
        frame_height = frame.shape[0]
        msg = "Press `Space or Enter` to confirm ROI.\nPress `R` to re-select.\nPress `esc or c` to break the process"
        print("\033[92m" + msg + "\033[0m")
        putText = lambda frame, msg, y_pos: cv2.putText(
            frame,
            msg,
            (2, y_pos),
            cv2.FONT_HERSHEY_PLAIN,
            1.25,
            color=(10, 255, 255),
            thickness=2,
            lineType=cv2.LINE_AA,
        )
        msg_lines = msg.split("\n")
        for i, m in enumerate(msg_lines):
            text_y_pos = frame_height - (len(msg_lines) - i) * 22 + 10
            putText(frame, m, y_pos=text_y_pos)

        while True:
            cv2.imshow(windowname, frame)
            ret = cv2.waitKey(33) & 0xFF

            if ret in (27, 99):  # esc or c
                raise KeyboardInterrupt("Cancel the selection")

            if ret in (13, 32):  # space or enter
                break

            if ret == 114:  # r
                return setrois(videopath, num_of_objects)
        return rois

    finally:
        clear_output()
        cv2.destroyAllWindows()
        cv2.waitKey(1)
