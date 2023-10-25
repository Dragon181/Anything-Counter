from dataclasses import dataclass
from typing import TypeVar, Generic, Tuple, List, Union, Dict, Optional

from nptyping import Float32, Int32, NDArray, Shape, UInt8


ImageArr = NDArray[Shape['* height, * width, 3 bgr'], UInt8]

Blob = Union[
    NDArray[Shape['1 batch, 3 bgr, * height, * width'], UInt8],  # noqa: F722
]


Coordinate = TypeVar('Coordinate', int, float)


@dataclass
class Point(Generic[Coordinate]):
    x: Coordinate  # noqa: VNE001
    y: Coordinate  # noqa: VNE001

    @property
    def as_tuple(self) -> Tuple[Coordinate, Coordinate]:
        return self.x, self.y


@dataclass
class Box(Generic[Coordinate]):
    top_left: Point[Coordinate]
    bottom_right: Point[Coordinate]

    @property
    def center(self) -> Point[Coordinate]:
        if isinstance(self.top_left.x, int):
            return Point(
                x=(self.top_left.x + self.bottom_right.x) // 2,
                y=(self.top_left.y + self.bottom_right.y) // 2,
            )
        return Point(
            x=(self.top_left.x + self.bottom_right.x) / 2,
            y=(self.top_left.y + self.bottom_right.y) / 2,
        )


@dataclass
class Detection:
    absolute_box: Box[int]
    relative_box: Box[float]
    score: Optional[float]
    label_as_str: str
    label_as_int: int


Detections = List[Detection]


@dataclass
class TrackingResult:
    intersection_dict: Dict[str, bool]
    detections: Detections


TrackingResults = Dict[int, TrackingResult]


@dataclass
class CountResult:
    in_count: int = 0
    out_count: int = 0


AreaCountResult = Dict[str, CountResult]


@dataclass
class Line:
    start: Point[float]
    end: Point[float]
