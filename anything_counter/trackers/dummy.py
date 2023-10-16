from anything_counter.anything_counter.models import Detections, TrackingResults
from anything_counter.anything_counter.tracker import Tracker


class DummyTracker(Tracker):
    def track(self, detections: Detections) -> TrackingResults:
        return []
