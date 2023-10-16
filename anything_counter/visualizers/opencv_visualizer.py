import cv2

from anything_counter.anything_counter.models import TrackingResults, CountResult, ImageArr
from anything_counter.anything_counter.visualizer import Visualizer


class OpenCVVisualizer(Visualizer):
    def paint(self, tracking_results: TrackingResults, counter: CountResult, image: ImageArr) -> ImageArr:
        for track_id, detections in tracking_results.items():

            cv2.rectangle(
                image, detections[-1].absolute_box.top_left.as_tuple, detections[-1].absolute_box.bottom_right.as_tuple,
                (255, 0, 0), 5,
            )
            cv2.putText(
                image, str(track_id), detections[-1].absolute_box.top_left.as_tuple, cv2.FONT_HERSHEY_SIMPLEX,
                5, (0, 255, 0), 10, cv2.LINE_AA
            )
        return image

    def visualize(self, image: ImageArr) -> None:
        cv2.imshow('Frame', cv2.resize(image, (512, 512)))

