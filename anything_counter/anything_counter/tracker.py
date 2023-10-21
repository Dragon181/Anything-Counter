from abc import ABC, abstractmethod

from anything_counter.anything_counter.models import Detections, TrackingResults, ImageArr


class Tracker(ABC):

    @abstractmethod
    def track(self, detections: Detections, image: ImageArr) -> TrackingResults:
        raise NotImplementedError
