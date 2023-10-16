from abc import ABC, abstractmethod
from typing import Iterator

from anything_counter.anything_counter.models import ImageArr


class Loader(ABC):

    @abstractmethod
    def load(self) -> Iterator[ImageArr]:
        raise NotImplementedError
