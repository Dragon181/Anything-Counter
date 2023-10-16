from typing import Any

import cv2  # type: ignore
import numpy as np

from anything_counter.anything_counter.detector import Detector
from anything_counter.anything_counter.models import Blob, Detections, ImageArr, Detection, Box, Point
from anything_counter.utils.openvino_adapter_mixin import OpenVINOAdapterMixin


class OpenVINODetector(Detector, OpenVINOAdapterMixin):
    def __init__(
            self,
            model: str,
            weights: str,
            scale_factor: float,
            score_threshold: float,
            nms_threshold: float,
            width: int,
            height: int,
    ) -> None:
        super().__init__(model, weights, scale_factor, width, height)
        self._score_threshold = score_threshold
        self._nms_threshold = nms_threshold

    def _pre_processing(self, image: ImageArr) -> Blob:
        image = cv2.resize(image, (self._width, self._height))
        image = image.transpose((2, 0, 1))  # BHWC to BCHW
        image = np.expand_dims(image, axis=0)
        return image

    def _post_processing(self, output: Any, original_image: ImageArr) -> Any:
        image_height, image_width = original_image.shape[:2]
        output = output[output[:, :, :, 2] > self._score_threshold]
        detections = []

        for _, _, conf, x_min, y_min, x_max, y_max in output:
            detections.append(
                Detection(
                    absolute_box=Box[int](
                        top_left=Point(x=int(x_min * image_width), y=int(y_min * image_height)),
                        bottom_right=Point(x=int(x_max * image_width), y=int(y_max * image_height))),
                    relative_box=Box[float](top_left=Point(x=x_min, y=y_min), bottom_right=Point(x=x_max, y=y_max)),
                    score=conf,
                    label_as_str='person',
                    label_as_int=0,
                ),
            )
        # for detection in detections:
        #     cv2.rectangle(
        #         original_image, detection.absolute_box.top_left.as_tuple, detection.absolute_box.bottom_right.as_tuple,
        #         (255, 0, 0), 5
        #     )
        #
        # cv2.imshow('out', cv2.resize(original_image, (512, 512)))
        # cv2.waitKey()

        return detections

    def detect(self, image: ImageArr) -> Detections:
        detections: Detections = self._predict(image=image)
        return detections
