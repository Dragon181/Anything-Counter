from typing import Iterator

import cv2

from anything_counter.anything_counter.loader import Loader
from anything_counter.anything_counter.models import ImageArr


class VideoLoader(Loader):
    def __init__(self, video_path: str):
        self.cap = cv2.VideoCapture(video_path)

    def load(self) -> Iterator[ImageArr]:
        if not self.cap.isOpened():
            print("Error opening video stream or file")

        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if frame is None:
                break
            if ret:
                yield frame

    def end_stream(self) -> None:
        self.cap.release()
        cv2.destroyAllWindows()
