from abc import ABC, abstractmethod

from anything_counter.anything_counter.models import ImageArr, Detections


class Detector(ABC):

    @abstractmethod
    def detect(self, image: ImageArr) -> Detections:
        raise NotImplementedError
