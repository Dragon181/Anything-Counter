import logging
from typing import Dict

from anything_counter.anything_counter.counter import Counter
from anything_counter.anything_counter.models import TrackingResults, CountResult, Line, Point, AreaCountResult
from anything_counter.utils.functions import is_intersecting


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
