from anything_counter.anything_counter.models import TrackingResults, CountResult, ImageArr
from anything_counter.anything_counter.visualizer import Visualizer


class DummyVisualizer(Visualizer):
    def paint(self, tracking_results: TrackingResults, counter: CountResult, image: ImageArr) -> ImageArr:
        return image

    def visualize(self, image: ImageArr) -> None:
        pass
