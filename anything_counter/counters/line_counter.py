from typing import List, Optional

from anything_counter.anything_counter.counter import Counter
from anything_counter.anything_counter.models import TrackingResults, CountResult, Line, Point


def ccw(a: Point, b: Point, c: Point) -> int:
    return (b.x - a.x) * (c.y - a.y) - (c.x - a.x) * (b.y - a.y)


def is_intersecting(line1: Line, line2: Line) -> Optional[int]:

    o1 = ccw(line1.start, line1.end, line2.start)
    o2 = ccw(line1.start, line1.end, line2.end)
    o3 = ccw(line2.start, line2.end, line1.start)
    o4 = ccw(line2.start, line2.end, line1.end)

    if o1 * o2 < 0 and o3 * o4 < 0:
        direction = 1 if o1 * o2 > 0 else -1
        return direction
    return None


class LineCounter(Counter):
    def __init__(self, list_of_points: List[List[float]]):
        list_of_points = sorted(
            [Point(x=x, y=y) for (x, y) in list_of_points], key=lambda point: point.x ** 2 + point.y ** 2)
        self._line = Line(start=list_of_points[0], end=list_of_points[1])
        self._counter = CountResult()

    def count(self, tracking_results: TrackingResults) -> CountResult:
        for track, dets in tracking_results.items():
            if len(dets) < 2:
                continue
            last_det = dets[-2]
            now_det = dets[-1]
            track_line = Line(last_det.relative_box.center, now_det.relative_box.center)
            intersection = is_intersecting(self._line, track_line)
            if intersection == 1:
                self._counter.in_count += 1
            elif intersection == -1:
                self._counter.out_count += 1

        return self._counter
