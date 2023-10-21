import cv2

from anything_counter.anything_counter.counter import Counter
from anything_counter.anything_counter.detector import Detector
from anything_counter.anything_counter.loader import Loader
from anything_counter.anything_counter.models import Detections, TrackingResults, CountResult
from anything_counter.anything_counter.tracker import Tracker
from anything_counter.anything_counter.visualizer import Visualizer


class AnythingCounter:
    def __init__(
        self,
        loader: Loader,
        detector: Detector,
        tracker: Tracker,
        counter: Counter,
        visualizer: Visualizer,
    ):
        self._loader = loader
        self._detector = detector
        self._tracker = tracker
        self._counter = counter
        self._visualizer = visualizer

    def run(self) -> None:

        for image in self._loader.load():
            detections: Detections = self._detector.detect(image=image)
            tracking_results: TrackingResults = self._tracker.track(detections=detections, image=image)

            count_result: CountResult = self._counter.count(tracking_results=tracking_results)

            image_result = self._visualizer.paint(tracking_results=tracking_results, counter=count_result, image=image)
            self._visualizer.visualize(image=image_result)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        self._loader.cap.release()
        cv2.destroyAllWindows()
