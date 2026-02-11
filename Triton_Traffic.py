import cv2
import numpy as np
import json
from collections import deque
from tqdm import tqdm
from dataclasses import dataclass, field
from typing import Optional, Dict, Deque, List, Tuple

import tritonclient.http as httpclient
from tritonclient.http import InferInput, InferRequestedOutput


# -------------------------
# CONFIG
# -------------------------
TRITON_URL = "localhost:8000"
MODEL_NAME = "person_detection_yolo26"

VIDEO_PATH = "NewOrleans.mp4"
OUTPUT_PATH = "traffic_tracked_output_triton.mp4"

TARGET_SIZE = 640
PAD_VALUE = 114

# === DETECTION THRESHOLDS ===
TRAFFIC_CONF_THRES = 0.3
IOU_THRES = 0.45

# === STABILITY SETTINGS ===
TRACK_HISTORY = 100
MISSING_TOLERANCE = 75
IOU_MATCH_THRESHOLD = 0.3
BOX_SMOOTH_ALPHA = 0.7

# Traffic and road related classes (IDs that are valid)
TRAFFIC_CLASSES = {
    0: 'person',
    1: 'bicycle',
    2: 'car',
    3: 'motorcycle',
    4: 'airplane',
    5: 'bus',
    6: 'train',
    7: 'truck',
    8: 'boat',
    9: 'traffic light',
    10: 'fire hydrant',
    11: 'stop sign',
    12: 'parking meter',
    13: 'bench'
}

COLORS = {
    'person': (255, 0, 0),
    'bicycle': (0, 255, 255),
    'car': (0, 255, 0),
    'motorcycle': (255, 0, 255),
    'bus': (0, 165, 255),
    'truck': (0, 128, 255),
    'traffic light': (0, 0, 255),
    'stop sign': (0, 0, 200),
    'fire hydrant': (255, 255, 0),
    'parking meter': (180, 180, 180),
    'train': (255, 100, 0),
    'airplane': (200, 200, 0),
    'boat': (255, 200, 100),
    'bench': (128, 128, 128)
}


# -------------------------
# TRITON CLIENT
# -------------------------
triton_client = httpclient.InferenceServerClient(url=TRITON_URL, verbose=False)


# -------------------------
# PREPROCESS (LETTERBOX)
# -------------------------
def preprocess(frame):
    h, w, _ = frame.shape
    scale = min(TARGET_SIZE / h, TARGET_SIZE / w)
    nh, nw = int(h * scale), int(w * scale)

    resized = cv2.resize(frame, (nw, nh))
    canvas = np.full((TARGET_SIZE, TARGET_SIZE, 3), PAD_VALUE, dtype=np.uint8)

    px = (TARGET_SIZE - nw) // 2
    py = (TARGET_SIZE - nh) // 2
    canvas[py:py + nh, px:px + nw] = resized

    img = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))[None]

    return img, {
        "scale": scale,
        "pad_x": px,
        "pad_y": py,
        "orig_w": w,
        "orig_h": h
    }


# -------------------------
# TRITON INFER
# -------------------------
def triton_infer(img):
    inp = InferInput("images", img.shape, "FP32")
    inp.set_data_from_numpy(img)
    out = InferRequestedOutput("output0")
    resp = triton_client.infer(MODEL_NAME, inputs=[inp], outputs=[out])
    return resp.as_numpy("output0")


# -------------------------
# POSTPROCESS
# output shape: [1, 300, 6] => x1, y1, x2, y2, conf, cls
# -------------------------
def postprocess(output, meta):
    dets = []
    preds = output[0]

    for p in preds:
        x1, y1, x2, y2, conf, cls = p
        cls = int(cls)

        if conf < TRAFFIC_CONF_THRES:
            continue

        if cls not in TRAFFIC_CLASSES:
            continue

        # Undo padding + scaling
        x1 = (x1 - meta["pad_x"]) / meta["scale"]
        x2 = (x2 - meta["pad_x"]) / meta["scale"]
        y1 = (y1 - meta["pad_y"]) / meta["scale"]
        y2 = (y2 - meta["pad_y"]) / meta["scale"]

        # Clamp
        x1 = max(0, min(meta["orig_w"], x1))
        x2 = max(0, min(meta["orig_w"], x2))
        y1 = max(0, min(meta["orig_h"], y1))
        y2 = max(0, min(meta["orig_h"], y2))

        dets.append({
            "box": (int(x1), int(y1), int(x2), int(y2)),
            "class": TRAFFIC_CLASSES[cls],
            "confidence": float(conf)
        })

    return dets


# -------------------------
# TRACKED OBJECT CLASS
# -------------------------
@dataclass
class TrackedObject:
    track_id: int
    object_class: str
    box: Tuple[int, int, int, int]
    confidence_history: Deque[float] = field(default_factory=lambda: deque(maxlen=TRACK_HISTORY))
    frames_missing: int = 0
    smoothed_box: Tuple[int, int, int, int] = None
    first_seen_frame: int = 0
    last_seen_frame: int = 0

    def __post_init__(self):
        self.smoothed_box = self.box

    def update_box(self, new_box: Tuple[int, int, int, int], confidence: float, frame_num: int):
        if self.smoothed_box is None:
            self.smoothed_box = new_box
        else:
            self.smoothed_box = tuple(
                int(BOX_SMOOTH_ALPHA * new + (1 - BOX_SMOOTH_ALPHA) * old)
                for new, old in zip(new_box, self.smoothed_box)
            )
        self.box = new_box
        self.confidence_history.append(confidence)
        self.frames_missing = 0
        self.last_seen_frame = frame_num

    def get_average_confidence(self) -> float:
        if not self.confidence_history:
            return 0.0
        return sum(self.confidence_history) / len(self.confidence_history)

    def mark_missing(self):
        self.frames_missing += 1

    def is_expired(self) -> bool:
        return self.frames_missing > MISSING_TOLERANCE


# -------------------------
# TRAFFIC OBJECT TRACKER
# -------------------------
class TrafficObjectTracker:
    def __init__(self):
        self.tracked_objects: Dict[int, TrackedObject] = {}
        self.next_id = 0
        self.frame_count = 0
        self.class_statistics: Dict[str, int] = {cls: 0 for cls in TRAFFIC_CLASSES.values()}

    def _iou(self, boxA: tuple, boxB: tuple) -> float:
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        inter = max(0, xB - xA) * max(0, yB - yA)
        areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        union = areaA + areaB - inter
        return inter / union if union > 0 else 0

    def update(self, detections: List[Dict]) -> List[TrackedObject]:
        self.frame_count += 1

        for obj in self.tracked_objects.values():
            obj.mark_missing()

        matched_track_ids = set()
        matched_det_indices = set()

        matches = []
        for det_idx, detection in enumerate(detections):
            det_box = detection["box"]
            det_class = detection["class"]

            for track_id, tracked in self.tracked_objects.items():
                if tracked.object_class != det_class:
                    continue

                iou_score = self._iou(det_box, tracked.smoothed_box)
                if iou_score > IOU_MATCH_THRESHOLD:
                    matches.append((iou_score, det_idx, track_id))

        matches.sort(reverse=True, key=lambda x: x[0])

        for iou_score, det_idx, track_id in matches:
            if det_idx in matched_det_indices or track_id in matched_track_ids:
                continue
            detection = detections[det_idx]
            self.tracked_objects[track_id].update_box(
                detection["box"],
                detection["confidence"],
                self.frame_count
            )
            matched_track_ids.add(track_id)
            matched_det_indices.add(det_idx)

        for det_idx, detection in enumerate(detections):
            if det_idx not in matched_det_indices:
                class_name = detection["class"]
                new_object = TrackedObject(
                    track_id=self.next_id,
                    object_class=class_name,
                    box=detection["box"],
                    first_seen_frame=self.frame_count
                )
                new_object.update_box(
                    detection["box"],
                    detection["confidence"],
                    self.frame_count
                )
                self.tracked_objects[self.next_id] = new_object
                self.class_statistics[class_name] += 1
                self.next_id += 1

        expired_ids = [tid for tid, obj in self.tracked_objects.items() if obj.is_expired()]
        for tid in expired_ids:
            del self.tracked_objects[tid]

        return [obj for obj in self.tracked_objects.values() if obj.frames_missing == 0]

    def get_statistics(self) -> Dict[str, int]:
        return {
            "total_tracked": self.next_id,
            "currently_active": len([o for o in self.tracked_objects.values() if o.frames_missing == 0]),
            "class_counts": self.class_statistics.copy()
        }


# -------------------------
# VISUALIZATION
# -------------------------
def draw_annotations(frame: np.ndarray, tracked_objects: List[TrackedObject],
                     frame_width: int, frame_height: int,
                     show_track_id: bool = True) -> np.ndarray:
    for obj in tracked_objects:
        x1, y1, x2, y2 = obj.smoothed_box
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame_width, x2), min(frame_height, y2)

        color = COLORS.get(obj.object_class, (0, 255, 0))

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        avg_conf = obj.get_average_confidence()
        if show_track_id:
            label = f"ID:{obj.track_id} {obj.object_class} {avg_conf:.2f}"
        else:
            label = f"{obj.object_class} {avg_conf:.2f}"

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        (label_width, label_height), baseline = cv2.getTextSize(
            label, font, font_scale, thickness
        )

        label_y1 = max(0, y1 - label_height - baseline - 5)
        label_y2 = y1
        label_x1 = x1
        label_x2 = min(frame_width, x1 + label_width + 10)

        cv2.rectangle(frame, (label_x1, label_y1), (label_x2, label_y2), color, -1)

        cv2.putText(
            frame, label, (x1 + 5, y1 - 5),
            font, font_scale, (255, 255, 255), thickness
        )

    return frame


def draw_statistics_panel(frame: np.ndarray, tracker: TrafficObjectTracker,
                          frame_width: int, frame_height: int) -> np.ndarray:
    stats = tracker.get_statistics()

    panel_width = 350
    panel_x = frame_width - panel_width - 10
    panel_y = 10
    line_height = 25
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2

    lines = [
        f"Frame: {tracker.frame_count}",
        f"Active Objects: {stats['currently_active']}",
        f"Total Tracked: {stats['total_tracked']}",
        "--- Class Counts ---"
    ]

    for cls, count in sorted(stats['class_counts'].items(), key=lambda x: x[1], reverse=True):
        if count > 0:
            lines.append(f"{cls}: {count}")

    panel_height = len(lines) * line_height + 20

    overlay = frame.copy()
    cv2.rectangle(overlay, (panel_x, panel_y),
                  (panel_x + panel_width, panel_y + panel_height),
                  (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    cv2.rectangle(frame, (panel_x, panel_y),
                  (panel_x + panel_width, panel_y + panel_height),
                  (255, 255, 255), 2)

    y = panel_y + line_height
    for i, line in enumerate(lines):
        color = (0, 255, 255) if i < 3 else (255, 255, 255)
        if "---" in line:
            color = (0, 255, 0)
        cv2.putText(frame, line, (panel_x + 10, y),
                    font, font_scale, color, thickness)
        y += line_height

    return frame


# -------------------------
# MAIN
# -------------------------
def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out = cv2.VideoWriter(
        OUTPUT_PATH,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps, (W, H)
    )

    tracker = TrafficObjectTracker()

    for frame_idx in tqdm(range(total_frames), desc="Processing"):
        ret, frame = cap.read()
        if not ret:
            break

        img, meta = preprocess(frame)
        output = triton_infer(img)
        detections = postprocess(output, meta)

        tracked_objects = tracker.update(detections)

        frame = draw_annotations(frame, tracked_objects, W, H, show_track_id=True)
        frame = draw_statistics_panel(frame, tracker, W, H)

        out.write(frame)

    cap.release()
    out.release()

    final_stats = tracker.get_statistics()
    print(f"\n✅ Saved output video to: {OUTPUT_PATH}")
    print(f"Total frames: {total_frames}")
    print(f"Total objects tracked: {final_stats['total_tracked']}")

    # Class-wise analysis as JSON
    class_analysis = {
        "total_frames": total_frames,
        "total_objects_tracked": final_stats["total_tracked"],
        "class_counts": final_stats["class_counts"]
    }
    print(json.dumps(class_analysis, indent=2))


if __name__ == "__main__":
    main()
