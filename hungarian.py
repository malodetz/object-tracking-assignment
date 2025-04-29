import numpy as np
from scipy.optimize import linear_sum_assignment

from common import compute_iou


class HungarianTracker:
    def __init__(self, max_age=15, min_hits=3, distance_threshold=200):
        self.max_age = max_age
        self.min_hits = min_hits
        self.distance_threshold = distance_threshold
        self.tracks = []
        self.next_id = 0

    def update(self, detections):
        if not self.tracks:
            for det in detections:
                if len(det.get("bounding_box", [])) >= 4:  # Valid detection
                    self.tracks.append(
                        {
                            "track_id": self.next_id,
                            "x": det["x"],
                            "y": det["y"],
                            "bounding_box": det["bounding_box"],
                            "age": 1,
                            "total_hits": 1,
                            "consecutive_misses": 0,
                            "last_detection": det,
                        }
                    )
                    det["track_id"] = self.next_id
                    self.next_id += 1
                else:
                    det["track_id"] = None
            return detections
        cost_matrix = np.zeros((len(detections), len(self.tracks)))
        for i, det in enumerate(detections):
            for j, track in enumerate(self.tracks):
                cost_matrix[i, j] = np.sqrt(
                    (det["x"] - track["x"]) ** 2 + (det["y"] - track["y"]) ** 2
                )
                if (
                    len(det.get("bounding_box", [])) >= 4
                    and len(track.get("bounding_box", [])) >= 4
                ):
                    iou = compute_iou(det["bounding_box"], track["bounding_box"])
                    cost_matrix[i, j] *= 1 - iou * 0.5
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        unmatched_tracks = list(range(len(self.tracks)))
        unmatched_detections = list(range(len(detections)))
        matches = []
        for row, col in zip(row_indices, col_indices):
            if cost_matrix[row, col] < self.distance_threshold:
                matches.append((row, col))
                if col in unmatched_tracks:
                    unmatched_tracks.remove(col)
                if row in unmatched_detections:
                    unmatched_detections.remove(row)
        for det_idx, track_idx in matches:
            det = detections[det_idx]
            track = self.tracks[track_idx]
            track["x"] = det["x"]
            track["y"] = det["y"]
            if len(det.get("bounding_box", [])) >= 4:
                track["bounding_box"] = det["bounding_box"]
            track["age"] += 1
            track["total_hits"] += 1
            track["consecutive_misses"] = 0
            track["last_detection"] = det
            det["track_id"] = track["track_id"]
        for idx in unmatched_tracks:
            track = self.tracks[idx]
            track["age"] += 1
            track["consecutive_misses"] += 1
        new_tracks = []
        for track in self.tracks:
            if track["consecutive_misses"] <= self.max_age:
                new_tracks.append(track)
        self.tracks = new_tracks
        for idx in unmatched_detections:
            det = detections[idx]
            if len(det.get("bounding_box", [])) >= 4:
                self.tracks.append(
                    {
                        "track_id": self.next_id,
                        "x": det["x"],
                        "y": det["y"],
                        "bounding_box": det["bounding_box"],
                        "age": 1,
                        "total_hits": 1,
                        "consecutive_misses": 0,
                        "last_detection": det,
                    }
                )
                det["track_id"] = self.next_id
                self.next_id += 1
            else:
                det["track_id"] = None
        for det in detections:
            if det.get("track_id") is None and len(det.get("bounding_box", [])) >= 4:
                det["track_id"] = self.next_id
                self.next_id += 1
        return detections
