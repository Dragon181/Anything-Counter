from abc import ABC, abstractmethod

from anything_counter.anything_counter.models import Detections, TrackingResults


class Tracker(ABC):

    @abstractmethod
    def track(self, detections: Detections) -> TrackingResults:
        raise NotImplementedError