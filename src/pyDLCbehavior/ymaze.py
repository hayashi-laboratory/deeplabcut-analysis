from dataclasses import dataclass, field
from numbers import Number
from typing import (
    Any,
    Dict,
    Generator,
    Hashable,
    List,
    NamedTuple,
    OrderedDict,
    Tuple,
    Union,
)

import cv2
import matplotlib as mpl
import numpy as np
from matplotlib.patches import Polygon
from math import sqrt

__all__ = [
    "Point",
    "Line",
    "sort_coordinates",
    "PixelScaler",
    "YMazeScaler",
    "ArmRegion",
    "ArmCollection",
    "BasicYMazeCollection",
]


SQRT3 = sqrt(3)


class Point(NamedTuple):
    x: Number = np.nan
    y: Number = np.nan

    def __add__(self, other: "Point"):
        if not isinstance(other, self.__class__):
            raise NotImplementedError()
        return Point(self.x + other.x, self.y + other.y)

    def __sub__(self, other: "Point"):
        if not isinstance(other, self.__class__):
            raise NotImplementedError()
        return Point(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar):
        if not isinstance(scalar, Number):
            raise NotImplementedError()
        return Point(self.x * scalar, self.y * scalar)

    def __neg__(self):
        return Point(self.x * (-1), self.y * (-1))

    def __repr__(self):
        return f"Point(x={self.x:.2f},y={self.y:.2f})"


def P(r: float, theta: float, origin=(0, 0)) -> Point:
    """Get x, y coordinate by radius and length

    Args:
        r (float): distance to origin
        theta (float): rotation degree from axis-x (deg)

    Returns:
        Point: x, y point
    """
    from math import cos, sin, radians

    x = np.round(cos(radians(theta)) * r + origin[0], 3)
    y = np.round(sin(radians(theta)) * r + origin[1], 3)
    return Point(x, y)


def sort_coordinates(
    points: Union[List[Tuple[Number, Number]], np.ndarray],
) -> np.ndarray:
    """Sort the point collections in clock-wise order

    Args:
        points (Union[List[Tuple[Number,Number], np.ndarray]): A list or numpy array that contains (N, 2) of points

    Raises:
        ValueError: if input size is not (N, 2)

    Returns:
        np.ndarray: a numpy array of sorted points coordinates
    """
    if isinstance(points, list):
        points = np.array(points)

    if points.ndim != 2 and points[1] != 2:
        raise ValueError(f"input size{points.shape} is not (N, 2)")

    sub_np = points - points.mean(axis=0)
    degree = -np.degrees(np.arctan2(sub_np[:, 1], sub_np[:, 0])) - 135
    degree = np.mod(degree, 360)
    return points[degree.argsort()].astype(np.float32)


@dataclass(frozen=True)
class Line:

    pt1: Point = field(default_factory=Point)
    pt2: Point = field(default_factory=Point)

    def __post_init__(self):
        object.__setattr__(self, "pt1", Point(*self.pt1))
        object.__setattr__(self, "pt2", Point(*self.pt2))

    def __getstate__(self):
        return dict(pt1=self.pt1, pt2=self.pt2)

    def __setstate__(self, d):
        self.__dict__.update(d)

    @property
    def slope(self):
        return np.Inf if self.xdiff == 0 else self.ydiff / self.xdiff

    @property
    def middle_pt(self) -> Point:
        x1, y1, x2, y2 = self.pt1, self.pt2
        return Point((x1 + x2) / 2, (y1 + y2) / 2)

    @property
    def dist(self):
        from math import pow, sqrt

        return sqrt(pow(self.xdiff, 2), pow(self.ydiff, 2))

    @property
    def constant(self):
        m = self.slope
        x1, y1 = self.pt1
        if m == 0:
            return y1
        if m == np.Inf:
            return np.nan
        return y1 - m * x1

    @property
    def xdiff(self):
        return self.pt1[0] - self.pt2[0]

    @property
    def ydiff(self):
        return self.pt1[1] - self.pt2[1]

    def __call__(self, x: float) -> float:
        if self.slope == np.Inf:
            return np.Inf
        if self.slope == 0:
            return 0
        return self.slope * x + self.constant

    def draw_line(self, ax, *args, **kwargs):
        array = np.array([self.pt1, self.pt2])
        # print("self.pt1 is ", array[0, :])
        ax.plot(array[:, 0], array[:, 1], *args, **kwargs)

    def cal_intersection(self, other: "Line") -> Point:
        """
        Calcuate the intersection point of two lines.
        If vertical_cross is True, calculate the intersection point of orthogonal lines from two lines, that passed the median point.
        :otherlines Line:
        :vertical_cross Bool:
        :RETURN (x, y) Tuple:
        """
        xdiff = self.xdiff, other.xdiff
        ydiff = self.ydiff, other.ydiff

        def det(a, b):
            return a[0] * b[1] - a[1] * b[0]

        div = det(xdiff, ydiff)
        if div == 0:
            return Point()

        d = (det(self.pt1, self.pt2), det(other.pt1, other.pt2))
        x = det(d, xdiff) / div
        y = det(d, ydiff) / div
        return Point(x, y)


@dataclass
class PixelScaler:
    target: List[Tuple[int, int]]
    reference: List[Tuple[int, int]]
    warp_mat: np.ndarray = field(init=False, default=np.array([(1, 0, 0), (0, 1, 0)]))

    def __post_init__(self):
        target_num = len(self.target)
        ref_num = len(self.reference)
        if (ref_num != target_num) or (ref_num < 2) or (len(self.reference[0]) != 2):
            raise ValueError(
                f"shape are not (N, 2), N > 1.\ntarget   : {self.target}\nreference: {self.reference}"
            )

        ref_np = sort_coordinates(self.reference)
        target_np = sort_coordinates(self.target)

        if ref_num == 2:
            ref_diff = np.abs(ref_np[1] - ref_np[0])
            target_dff = np.abs(target_np[1] - target_np[0])
            scale = ref_diff / target_dff
            self.warp_mat *= scale[None, :]
        else:
            step = (ref_num - 3) // 2 + 1
            self.warp_mat = cv2.getAffineTransform(target_np[::step], ref_np[::step])

        self.reference = list(map(tuple, ref_np))
        self.target = list(map(tuple, target_np))

    def scale(self, xy_coords: Union[np.ndarray, List[Point]]) -> np.ndarray:
        """Re-scale the xy-coordinates from target scale to reference scale.

        Args:
            xy_coords (np.ndarray): A xy-coordinate array with shape (..., 2)
        Raises:
            TypeError: ndim > 1 and array.shape[-1]

        Returns:
            np.ndarray: Return the rescale value (..., 2)
        """
        if not isinstance(xy_coords, np.ndarray):
            xy_coords = np.array(xy_coords)
        shape = xy_coords.shape
        ndim = xy_coords.ndim
        if ndim < 2 or shape[-1] != 2:
            raise TypeError(f"src shape{xy_coords.shape} is not (N, 2) or (M, N, 2)")
        pad_width = [(0, 0) for _ in range(ndim - 1)]
        pad_width.append((0, 1))
        return np.pad(xy_coords, pad_width=pad_width, constant_values=1).dot(
            self.warp_mat.T
        )


@dataclass
class YMazeScaler(PixelScaler):

    reference: List[Tuple[int, int]] = field(init=False)

    def __post_init__(self):
        r = 35 + 6 / SQRT3
        self.reference = np.array([P(r, 150), P(r, 270), P(r, 30)])
        super().__post_init__()


@dataclass
class ArmRegion:

    name: str = ""
    points: List[Point] = field(default_factory=list)

    def __post_init__(self):
        if len(self.points):
            self.sort()

    def __setstate__(self, d):
        self.__dict__.update(d)

    def __getstate__(self) -> Dict[str, Any]:
        return dict(name=self.name, points=self.points)

    @property
    def center(self) -> None:
        if not len(self.points):
            return Point()
        return Point(*np.array(self.points).mean(axis=0))

    def is_valid(self, x, y) -> bool:
        if x is None or y is None:
            return False
        if np.isnan(x) or np.isnan(y):
            return False
        return True

    def get_paired_lines(self) -> List[Line]:
        num = len(self.points)
        if not num:
            return list()
        return [Line(self.points[i - 1], self.points[i]) for i in range(num)]

    def sort(self):
        self.points = [tuple(pt) for pt in sort_coordinates(self.points)]

    def add_point(self, x: int, y: int):
        """Add a new point to area corner

        Args:
            x (int): x-coordinates
            y (int): y-coordinates

        Raises:
            ValueError: Value should be numeric and not nan.
        """
        if not self.is_valid(x, y):
            raise ValueError(f"invalid value ({x}, {y})")
        self.points.append(Point(x, y))
        self.sort()

    def add_points(self, xy_coords: List[Point]):
        self.points.extend([Point(*pt) for pt in xy_coords])
        self.sort()

    def contains(self, x: float, y: float) -> bool:
        """A function to check whether point is in Arm area (closed)

        Args:
            x (float): x-coordinates
            y (float): y-coordinates
        """
        if len(self.points) < 3:
            return False
        polygon = mpl.path.Path(self.points)
        return polygon.contains_point((x, y))

    def draw_area(self, ax, *args, **kwargs) -> None:
        if not self.points:
            return
        poly = Polygon(self.points, *args, **kwargs)
        ax.add_patch(poly)

    def merge(self, other: "ArmRegion") -> "ArmRegion":
        if not isinstance(other, self.__class__):
            raise NotImplementedError()
        self.points.extend(other.points)
        self.sort()
        return self

    def copy(self) -> "ArmRegion":
        r = ArmRegion(self.name)
        r.points = [p for p in self.points]
        return r


@dataclass
class ArmCollection:

    data: OrderedDict[str, ArmRegion] = field(default_factory=OrderedDict)
    area: ArmRegion = field(init=False, repr=False, default_factory=ArmRegion)

    @property
    def center(self):
        return self.area.center

    @property
    def each_arm_center(self):
        return [a.center for a in self.data.values()]

    def __getstate__(self):
        return dict(data=self.data, area=self.area)

    def __setstate__(self, d):
        self.__dict__.update(d)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, key: Hashable):
        return self.data[key]

    def __iter__(self) -> Generator[ArmRegion, None, None]:
        for key in self.data.keys():
            yield key

    def keys(self) -> Generator[str, None, None]:
        return self.data.keys()

    def values(self) -> Generator[ArmRegion, None, None]:
        return self.data.values()

    def items(self) -> Generator[Tuple[str, ArmRegion], None, None]:
        return self.data.items()

    def get(self, key, default=None):
        return self.data.get(key, default)

    def remove(self, item: Any) -> None:
        self.data = OrderedDict([(k, v) for k, v in self.data.items() if v != item])

    def append(self, arm_region: ArmRegion):
        if not isinstance(arm_region, ArmRegion):
            raise NotImplementedError
        self.data[arm_region.name] = arm_region
        self.area.merge(arm_region)

    def sort(self) -> None:
        self.area.sort()

    def draw_contour(self, ax, *arg, **kwargs):
        poly = Polygon(self.area.points, *arg, **kwargs)
        ax.add_patch(poly)

    # def get_ymaze_verts_by_coords(self):
    #     lines = self.area.get_paired_lines()
    #     center = np.array(self.center)
    #     is_close = lambda pt: 1e4 > np.sum(np.square(np.array(pt) - center))
    #     temp = self.area.copy()
    #     for i in range(len(lines)):
    #         l1 = lines[i - 2]
    #         l2 = lines[i]
    #         x, y = l1.cal_intersection(l2)
    #         if is_close((x, y)):
    #             temp.add_point(x, y)
    #     return temp.points


@dataclass
class BasicYMazeCollection(ArmCollection):
    def __post_init__(self):
        r = 12 / SQRT3
        p1, p2, p3 = P(r, 90), P(r, 210), P(r, 330)
        a1, a2, a3, a4 = P(30, 150, p2), P(40, 150, p2), P(40, 150, p1), P(30, 150, p1)
        b1, b2, b3, b4 = P(30, 30, p1), P(40, 30, p1), P(40, 30, p3), P(30, 30, p3)
        c1, c2, c3, c4 = P(30, 270, p3), P(40, 270, p3), P(40, 270, p2), P(30, 270, p2)
        o_pts = [p1, p2, p3]
        a_pts = [a1, a2, a3, a4]
        b_pts = [b1, b2, b3, b4]
        c_pts = [c1, c2, c3, c4]
        all_pts = o_pts + a_pts + b_pts + c_pts
        self.data = OrderedDict(
            A=ArmRegion("A", a_pts),
            B=ArmRegion("B", b_pts),
            C=ArmRegion("C", c_pts),
        )
        self.area.points = all_pts
        self.sort()

    def fit(
        self,
        reference_points: List[Point],
    ) -> None:
        """Fit the mask of YMaze to the center of 3 arms.
        Args:
            reference_points (List[Point]): The reference center points with the order [ArmA, ArmB, ArmC]
        """
        s = PixelScaler(self.each_arm_center, reference_points)
        np2pts = lambda l: [Point(*p) for p in l]
        for v in self.data.values():
            v.points = np2pts(s.scale(v.points))
        self.area.points = np2pts(s.scale(self.area.points))
