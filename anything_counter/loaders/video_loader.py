from typing import Iterator

import cv2  # type: ignore

from anything_counter.anything_counter.loader import Loader
from anything_counter.anything_counter.models import ImageArr


class VideoLoader(Loader):
    def __init__(self, video_path: str):
        self.cap = cv2.VideoCapture(video_path)

    def load(self) -> Iterator[ImageArr]:
        if self.cap.isOpened() == False:
            print("Error opening video stream or file")

        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if frame is None:
                break
            if ret == True:
                yield frame
