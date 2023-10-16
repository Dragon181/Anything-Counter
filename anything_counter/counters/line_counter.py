from anything_counter.anything_counter.counter import Counter
from anything_counter.anything_counter.models import TrackingResults, CountResult


class LineCounter(Counter):
    def __init__(self):
        pass

    def count(self, tracking_results: TrackingResults) -> CountResult:
        pass
