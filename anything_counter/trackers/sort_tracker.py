from collections import defaultdict
from typing import Dict

import numpy as np
from nptyping import NDArray, Shape, UInt8

from anything_counter.anything_counter.models import (
    Detections, TrackingResults, TrackingResult, Detection, Box, Point, ImageArr,
)
from anything_counter.anything_counter.tracker import Tracker
from anything_counter.trackers.sort import Sort
from anything_counter.utils.functions import intersection_over_union


class SortTracker(Tracker):
    def __init__(self, max_age: int, min_hits: int, iou_threshold: float):
        self._sort = Sort(max_age=max_age, min_hits=min_hits, iou_threshold=iou_threshold)

        self._max_age = max_age
        self._tracking_results: TrackingResults = {}
        self._miss_track: Dict[int, int] = defaultdict(int)

    def _update_track_result(self, track_id: int, detection: Detection):
        if not self._tracking_results.get(track_id):
            self._tracking_results[track_id] = TrackingResult(
                intersection_dict=defaultdict(bool), detections=[detection]
            )
        else:
            self._tracking_results[track_id].detections.append(detection)

    def _clean_track_results(self, tracks: NDArray[Shape['* count, 5 shapes'], UInt8]):
        all_tracks = set(track[-1] for track in tracks)
        track_ids = list(self._miss_track.keys())
        for track_id in track_ids:
            if track_id not in all_tracks:
                self._miss_track[track_id] += 1
                if self._miss_track[track_id] > self._max_age:
                    del self._tracking_results[track_id]
                    del self._miss_track[track_id]

    def _get_det_score(self, bbox, detections):
        # TODO: do some smarter score saving

        max_iou = -1.0
        max_score = None

        for det in detections:
            det_bbox = det.as_array[:4]
            score = det.as_array[4]

            iou = intersection_over_union(bbox, det_bbox)
            if iou > max_iou:
                max_iou = iou
                max_score = score

        return max_score

    def track(self, detections: Detections, image: ImageArr) -> TrackingResults:
        height, width = image.shape[:2]
        dets = [det.as_array for det in detections]

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
                    score=self._get_det_score(track[:4], detections),
                    label_as_str='person',
                    label_as_int=0,
                )

                self._update_track_result(track_id=track_id, detection=detection)

                if not self._miss_track.get(track_id):
                    self._miss_track[track_id] = 0
        else:
            self._sort.update(dets=np.empty((0, 5)))

        self._clean_track_results(tracks)

        return self._tracking_results
