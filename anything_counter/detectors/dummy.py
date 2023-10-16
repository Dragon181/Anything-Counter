from anything_counter.anything_counter.detector import Detector
from anything_counter.anything_counter.models import Detections, ImageArr


class DummyDetector(Detector):
    def detect(self, image: ImageArr) -> Detections:
        return []
