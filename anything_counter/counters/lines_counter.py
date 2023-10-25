import logging
from typing import Optional, Dict

from anything_counter.anything_counter.counter import Counter
from anything_counter.anything_counter.models import TrackingResults, CountResult, Line, Point, AreaCountResult


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


class LinesCounter(Counter):
    def __init__(self, dict_of_lines: Dict[str, Dict[str, float]]):
        self._lines = {
            line_name: Line(start=Point(x=line['x1'], y=line['y1']), end=Point(x=line['x2'], y=line['y2']))
            for line_name, line in dict_of_lines.items()
        }
        self._counters: AreaCountResult = {line_name: CountResult() for line_name in dict_of_lines.keys()}

    def count(self, tracking_results: TrackingResults) -> AreaCountResult:
        for line_name, line in self._lines.items():
            for track, track_result in tracking_results.items():

                intersection_dict = track_result.intersection_dict
                dets = track_result.detections

                if len(dets) < 2:
                    continue

                last_det = dets[-2]
                now_det = dets[-1]
                track_line = Line(last_det.relative_box.center, now_det.relative_box.center)
                intersection = is_intersecting(line, track_line)

                if intersection and intersection_dict[line_name]:
                    continue

                if intersection == 1:
                    self._counters[line_name].in_count += 1
                    logging.info(
                        f'IN event on {line_name}! \n IN: {self._counters[line_name].in_count}\n OUT: {self._counters[line_name].out_count}'
                    )
                elif intersection == -1:
                    self._counters[line_name].out_count += 1
                    logging.info(
                        f'OUT event on {line_name}! \n IN: {self._counters[line_name].in_count}\n OUT: {self._counters[line_name].out_count}')

        return self._counters
