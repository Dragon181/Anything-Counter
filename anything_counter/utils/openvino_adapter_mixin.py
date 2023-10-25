import logging
from abc import abstractmethod
from pathlib import Path
from typing import Any

from openvino.runtime import Core

from anything_counter.anything_counter.models import Blob, ImageArr


class OpenVINOAdapterMixin:
    def __init__(
            self,
            model: str,
            weights: str,
            scale_factor: float,
            width: int,
            height: int,
    ):
        if not Path(model).exists():
            raise FileNotFoundError(f'Model file "{model}" not found')

        if not Path(weights).exists():
            raise FileNotFoundError(f'Weights file "{weights}" not found')

        logging.info('Creating inference engine')
        self.__core = Core()

        logging.info('Loading network')
        self.__model = self.__core.read_model(model=Path(model))

        self.__compiled_model = self.__core.compile_model(
            model=self.__model,
            device_name='CPU',
            config={'PERFORMANCE_HINT': 'LATENCY'},
        )
        self.__infer_request = self.__compiled_model.create_infer_request()

        self._scale_factor: float = scale_factor

        self._width = width
        self._height = height

        logging.info(f'Image input height: {self._height}')
        logging.info(f'Image input width: {self._width}')

    @abstractmethod
    def _pre_processing(self, image: ImageArr) -> Blob:
        raise NotImplementedError

    def __infer(self, data: Any) -> Any:
        self.__infer_request.infer([data])
        return self.__infer_request.get_output_tensor().data

    @abstractmethod
    def _post_processing(self, output: Any, original_image: ImageArr) -> Any:
        raise NotImplementedError

    def _predict(self, image: ImageArr) -> Any:
        prepared_input = self._pre_processing(image=image)
        output = self.__infer(data=prepared_input)
        return self._post_processing(output=output, original_image=image)
