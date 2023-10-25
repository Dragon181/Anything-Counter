from collections import defaultdict
from typing import Dict

import numpy as np

from anything_counter.anything_counter.models import Detections, TrackingResults, TrackingResult, Detection, Box, Point, \
    ImageArr
from anything_counter.anything_counter.tracker import Tracker
from anything_counter.trackers.sort import Sort


class SortTracker(Tracker):
    def __init__(self, max_age: int, min_hits: int, iou_threshold: float):
        self._sort = Sort(max_age=max_age, min_hits=min_hits, iou_threshold=iou_threshold)

        self._max_age = max_age
        self._tracking_results: TrackingResults = {}
        self._miss_track: Dict[int, int] = defaultdict(int)

    def track(self, detections: Detections, image: ImageArr) -> TrackingResults:
        height, width = image.shape[:2]
        dets = [
            np.array([
                det.absolute_box.top_left.x, det.absolute_box.top_left.y,
                det.absolute_box.bottom_right.x, det.absolute_box.bottom_right.y, det.score
            ]) for det in detections]

        tracks = []

        if dets:
            tracks = self._sort.update(dets=np.stack(dets))

            for track in tracks:
                track_id = int(track[-1])
                x1, y1, x2, y2 = track[:4]
                detection = Detection(
                    absolute_box=Box(top_left=Point(x=int(x1), y=int(y1)), bottom_right=Point(x=int(x2), y=int(y2))),
                    relative_box=Box(
                        top_left=Point(x=x1 / width, y=y1 / height), bottom_right=Point(x=x2 / width, y=y2 / height)),
                    score=None,
                    label_as_str='person',
                    label_as_int=0,
                )

                if not self._tracking_results.get(track_id):
                    self._tracking_results[track_id] = TrackingResult(
                        intersection_dict=defaultdict(bool), detections=[detection]
                    )
                else:
                    self._tracking_results[track_id].detections.append(detection)

                if not self._miss_track.get(track_id):
                    self._miss_track[track_id] = 0

        all_tracks = set(track[-1] for track in tracks)
        track_ids = list(self._miss_track.keys())
        for track_id in track_ids:
            if track_id not in all_tracks:
                self._miss_track[track_id] += 1
                if self._miss_track[track_id] > self._max_age:
                    del self._tracking_results[track_id]
                    del self._miss_track[track_id]

        return self._tracking_results
