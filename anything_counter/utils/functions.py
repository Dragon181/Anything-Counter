from typing import List

import cv2
import numpy as np
from nptyping import Float32, NDArray, Shape, UInt8

from anything_counter.anything_counter.models import Box, Detection, Detections, ImageArr, Point


def parse_yolo_output(
        output: NDArray[Shape['1, 6300, 6'], Float32],
        score_threshold: float,
        nms_threshold: float,
        ratio_x: float,
        ratio_y: float,
        original_image: ImageArr,
) -> Detections:
    original_height, original_width = original_image.shape[:2]

    out = output[0]  # `[x_center, y_center, width, height, score, classes...]`
    out = out[out[:, 4] > score_threshold, :]

    if len(out) == 0:
        return []

    # non max suppression detected boxes
    indices_arr: List[List[int]] = cv2.dnn.NMSBoxes(
        bboxes=out[:, :4].tolist(),
        scores=out[:, 4].tolist(),
        score_threshold=score_threshold,
        nms_threshold=nms_threshold,
    )
    indices: NDArray[Shape['*'], UInt8] = np.array(indices_arr, dtype=int).ravel()  # noqa: F722

    boxes = out[indices, :4]
    boxes = np.hstack((boxes[:, :2] - boxes[:, 2:] // 2, boxes[:, :2] + boxes[:, 2:] // 2))  # xywh -> x1y1x2y2
    boxes = (boxes * [ratio_x, ratio_y, ratio_x, ratio_y]).astype(int)  # to absolute input image size
    boxes = boxes.clip(min=0, max=original_height if original_height == original_width else None)

    scores: List[float] = out[indices, 4].tolist()
    labels: List[int] = out[indices, 5:].argmax(axis=1).tolist()

    detections: Detections = []

    for (ax1, ay1, ax2, ay2), score, label in zip(boxes, scores, labels):
        detections.append(
            Detection(
                absolute_box=Box[int](top_left=Point(x=ax1, y=ay1), bottom_right=Point(x=ax2, y=ay2)),
                relative_box=Box[float](
                    top_left=Point(x=ax1 / original_width, y=ay1 / original_height),
                    bottom_right=Point(x=ax2 / original_width, y=ay2 / original_height)),
                score=score,
                label_as_str='plate',
                label_as_int=0,
            ),
        )
    return detections


def xywh2xyxy(xywh: NDArray[Shape['1, 4'], UInt8]) -> NDArray[Shape['1, 4'], UInt8]:
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    xyxy = np.copy(xywh)
    xyxy[:, 0] = xywh[:, 0] - xywh[:, 2] / 2  # top left x
    xyxy[:, 1] = xywh[:, 1] - xywh[:, 3] / 2  # top left y
    xyxy[:, 2] = xywh[:, 0] + xywh[:, 2] / 2  # bottom right x
    xyxy[:, 3] = xywh[:, 1] + xywh[:, 3] / 2  # bottom right y
    return xyxy

