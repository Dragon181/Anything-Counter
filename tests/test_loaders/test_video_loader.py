import numpy as np
import pytest
from anything_counter.loaders.video_loader import VideoLoader


@pytest.fixture
def video_loader():
    video_path = '../../assets/tests/output.avi'
    loader = VideoLoader(video_path)
    yield loader
    loader.end_stream()


def test_video_loader_open(video_loader):
    assert video_loader.cap.isOpened()


def test_video_loader_load(video_loader):
    for i, frame in enumerate(video_loader.load()):
        assert isinstance(frame, np.ndarray)
        assert frame.shape == (480, 640, 3)
        if i < 3:
            assert np.all(frame == np.zeros((480, 640, 3), dtype=np.uint8))
        else:
            assert np.all(frame == np.ones((480, 640, 3), dtype=np.uint8) * 255)


def test_video_loader_close(video_loader):
    video_loader.end_stream()
    assert not video_loader.cap.isOpened()
