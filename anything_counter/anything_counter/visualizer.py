from abc import ABC, abstractmethod

from anything_counter.anything_counter.models import ImageArr, TrackingResults, AreaCountResult


class Visualizer(ABC):

    @abstractmethod
    def paint(self, tracking_results: TrackingResults, counter: AreaCountResult, image: ImageArr) -> ImageArr:
        raise NotImplementedError

    @abstractmethod
    def visualize(self, image: ImageArr) -> None:
        raise NotImplementedError
