from typing import Tuple, List

import cv2

from anything_counter.anything_counter.models import TrackingResults, CountResult, ImageArr, Point, Line
from anything_counter.anything_counter.visualizer import Visualizer


class OpenCVVisualizer(Visualizer):
    def __init__(self, list_of_points: List[List[float]]):
        list_of_points = sorted(
            [Point(x=x, y=y) for (x, y) in list_of_points], key=lambda point: point.x ** 2 + point.y ** 2)
        self._line = Line(start=list_of_points[0], end=list_of_points[1])

    def draw_counter(self, counter: CountResult, image: ImageArr) -> ImageArr:
        cv2.putText(
            image, f'IN:{counter.in_count}', (50, 150), cv2.FONT_HERSHEY_SIMPLEX,
            5, (0, 255, 0), 10, cv2.LINE_AA
        )
        cv2.putText(
            image, f'OUT:{counter.out_count}', (50, 300), cv2.FONT_HERSHEY_SIMPLEX,
            5, (0, 0, 255), 10, cv2.LINE_AA
        )
        return image

    def paint(self, tracking_results: TrackingResults, counter: CountResult, image: ImageArr) -> ImageArr:

        h, w = image.shape[:2]
        start_point = (int(self._line.start.x * w), int(self._line.start.y * h))
        end_point = (int(self._line.end.x * w), int(self._line.end.y * h))

        cv2.line(image, start_point, end_point, (0, 255, 255), 10)
        for track_id, detections in tracking_results.items():

            cv2.rectangle(
                image, detections[-1].absolute_box.top_left.as_tuple, detections[-1].absolute_box.bottom_right.as_tuple,
                (255, 0, 0), 5,
            )
            cv2.putText(
                image, str(track_id), detections[-1].absolute_box.top_left.as_tuple, cv2.FONT_HERSHEY_SIMPLEX,
                5, (0, 255, 0), 10, cv2.LINE_AA
            )
        return self.draw_counter(counter, image)

    def visualize(self, image: ImageArr) -> None:
        cv2.imshow('Frame', cv2.resize(image, (0, 0), fx=0.2, fy=0.2))

