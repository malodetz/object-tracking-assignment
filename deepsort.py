import os

import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment

from common import compute_iou


class KalmanFilter:
    def __init__(self, dt=1.0):
        self.dt = dt
        self.state = np.zeros(4)
        self.F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]])
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        self.Q = np.eye(4) * 0.01
        self.R = np.eye(2) * 1.0
        self.P = np.eye(4) * 100.0

    def predict(self):
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.state[:2]

    def update(self, measurement):
        z = np.array(measurement)
        y = z - self.H @ self.state
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.state = self.state + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P
        return self.state[:2]


class DeepSortTracker:
    def __init__(
        self, max_age=30, min_hits=3, iou_threshold=0.3, frames_dir="save_frames_dir"
    ):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.frames_dir = frames_dir
        self.tracks = []
        self.next_id = 0

    def update(self, frame_id, detections):
        frame_path = os.path.join(self.frames_dir, f"frame_{frame_id}.png")
        frame = None
        if os.path.exists(frame_path):
            try:
                frame = Image.open(frame_path)
            except:
                print(f"Warning: Could not open frame {frame_path}")
        features = []
        for det in detections:
            if len(det.get("bounding_box", [])) >= 4 and frame is not None:
                feature = self._extract_feature(frame, det["bounding_box"])
                features.append(feature)
            else:
                features.append(None)
        for track in self.tracks:
            track["kalman_filter"].predict()
        if not self.tracks:
            for i, det in enumerate(detections):
                if len(det.get("bounding_box", [])) >= 4:
                    kf = KalmanFilter()
                    kf.state[:2] = np.array([det["x"], det["y"]])
                    kf.update([det["x"], det["y"]])

                    self.tracks.append(
                        {
                            "track_id": self.next_id,
                            "kalman_filter": kf,
                            "x": det["x"],
                            "y": det["y"],
                            "bounding_box": det["bounding_box"],
                            "feature": features[i],
                            "age": 1,
                            "total_hits": 1,
                            "consecutive_misses": 0,
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
                motion_dist = np.sqrt(
                    (det["x"] - track["x"]) ** 2 + (det["y"] - track["y"]) ** 2
                )
                if features[i] is not None and track["feature"] is not None:
                    appearance_dist = 1.0 - np.dot(features[i], track["feature"])
                else:
                    appearance_dist = 1.0
                combined_dist = 0.3 * motion_dist + 0.7 * appearance_dist
                if (
                    len(det.get("bounding_box", [])) >= 4
                    and len(track.get("bounding_box", [])) >= 4
                ):
                    iou = compute_iou(det["bounding_box"], track["bounding_box"])
                    cost_matrix[i, j] *= 1 - iou * 0.5
                cost_matrix[i, j] = combined_dist
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        unmatched_tracks = list(range(len(self.tracks)))
        unmatched_detections = list(range(len(detections)))

        matches = []
        for row, col in zip(row_indices, col_indices):
            if cost_matrix[row, col] != float("inf"):
                matches.append((row, col))
                if col in unmatched_tracks:
                    unmatched_tracks.remove(col)
                if row in unmatched_detections:
                    unmatched_detections.remove(row)
        for det_idx, track_idx in matches:
            det = detections[det_idx]
            track = self.tracks[track_idx]
            track["kalman_filter"].update([det["x"], det["y"]])
            track["x"] = det["x"]
            track["y"] = det["y"]
            if len(det.get("bounding_box", [])) >= 4:
                track["bounding_box"] = det["bounding_box"]
            if features[det_idx] is not None:
                track["feature"] = features[det_idx]
            track["age"] += 1
            track["total_hits"] += 1
            track["consecutive_misses"] = 0
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
                kf = KalmanFilter()
                kf.state[:2] = np.array([det["x"], det["y"]])
                kf.update([det["x"], det["y"]])
                self.tracks.append(
                    {
                        "track_id": self.next_id,
                        "kalman_filter": kf,
                        "x": det["x"],
                        "y": det["y"],
                        "bounding_box": det["bounding_box"],
                        "feature": features[idx],
                        "age": 1,
                        "total_hits": 1,
                        "consecutive_misses": 0,
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

    def _extract_feature(self, frame, bbox):
        try:
            x1, y1, x2, y2 = bbox
            x1, y1, x2, y2 = (
                max(0, int(x1)),
                max(0, int(y1)),
                min(frame.width, int(x2)),
                min(frame.height, int(y2)),
            )
            if x2 <= x1 or y2 <= y1:
                return np.zeros(128, dtype=np.float32)
            region = frame.crop((x1, y1, x2, y2))
            region = region.resize((64, 64))
            region_np = np.array(region)
            feature = np.zeros(128, dtype=np.float32)
            if region_np.ndim == 3:
                for i in range(min(3, region_np.shape[2])):
                    hist, _ = np.histogram(
                        region_np[:, :, i].flatten(),
                        bins=42,
                        range=(0, 255),
                        density=True,
                    )
                    feature[i * 42 : (i + 1) * 42] = hist
            else:
                hist, _ = np.histogram(
                    region_np.flatten(), bins=128, range=(0, 255), density=True
                )
                feature = hist
            norm = np.linalg.norm(feature)
            if norm > 0:
                feature = feature / norm
            return feature
        except Exception as e:
            print(f"Error extracting feature: {e}")
            return np.zeros(128, dtype=np.float32)
