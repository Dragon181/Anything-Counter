from typing import Dict

import cv2

from anything_counter.anything_counter.models import (
    TrackingResults, ImageArr, Point, Line, Detections, AreaCountResult
)
from anything_counter.anything_counter.visualizer import Visualizer


class OpenCVVisualizer(Visualizer):
    def __init__(self, dict_of_lines: Dict[str, Dict[str, float]]):
        self._lines = {
            line_name: Line(start=Point(x=line['x1'], y=line['y1']), end=Point(x=line['x2'], y=line['y2']))
            for line_name, line in dict_of_lines.items()
        }

    def draw_path(self, detections: Detections, image: ImageArr):

        last_detection = detections[0]
        for detection in detections[1:]:
            cv2.line(
                image, last_detection.absolute_box.center.as_tuple,
                detection.absolute_box.center.as_tuple, (255, 0, 0), 10
            )
            last_detection = detection
        return image

    def draw_lines_with_counters(self, image, counter):
        h, w = image.shape[:2]

        for line_name, line in self._lines.items():
            start_point = (int(line.start.x * w), int(line.start.y * h))
            end_point = (int(line.end.x * w), int(line.end.y * h))

            # draw line
            cv2.line(image, start_point, end_point, (0, 255, 255), 10)

            # write line name
            cv2.putText(
                image, line_name, (start_point[0], start_point[1] + 100), cv2.FONT_HERSHEY_SIMPLEX,
                5, (0, 0, 0), 15, cv2.LINE_AA
            )

            # write in and out count
            cv2.putText(
                image, f'{counter[line_name].in_count}', (start_point[0] - 100, start_point[1] + 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                5, (0, 255, 0), 10, cv2.LINE_AA
            )
            cv2.putText(
                image, f'{counter[line_name].out_count}', end_point, cv2.FONT_HERSHEY_SIMPLEX,
                5, (0, 0, 255), 10, cv2.LINE_AA
            )
        return image

    def paint(self, tracking_results: TrackingResults, counter: AreaCountResult, image: ImageArr) -> ImageArr:
        image = self.draw_lines_with_counters(image=image, counter=counter)
        for track_id, track_result in tracking_results.items():
            detections = track_result.detections
            cv2.rectangle(
                image, detections[-1].absolute_box.top_left.as_tuple, detections[-1].absolute_box.bottom_right.as_tuple,
                (255, 0, 0), 5,
            )
            cv2.putText(
                image, str(track_id), detections[-1].absolute_box.top_left.as_tuple, cv2.FONT_HERSHEY_SIMPLEX,
                5, (0, 255, 0), 10, cv2.LINE_AA
            )
            image = self.draw_path(detections, image)
        return image

    def visualize(self, image: ImageArr) -> None:
        cv2.imshow('Frame', cv2.resize(image, (0, 0), fx=0.2, fy=0.2))

