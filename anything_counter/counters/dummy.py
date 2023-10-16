from anything_counter.anything_counter.counter import Counter
from anything_counter.anything_counter.models import CountResult, TrackingResults


class DummyCounter(Counter):
    def count(self, tracking_results: TrackingResults) -> CountResult:
        return CountResult()
