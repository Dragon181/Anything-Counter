from abc import ABC, abstractmethod

from anything_counter.anything_counter.models import TrackingResults, AreaCountResult


class Counter(ABC):
    @abstractmethod
    def count(self, tracking_results: TrackingResults) -> AreaCountResult:
        raise NotImplementedError
