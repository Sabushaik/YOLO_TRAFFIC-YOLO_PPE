

#--------------PERSON PPE __CODE --------------
import cv2
import numpy as np
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
MODEL_NAME = "person_ppe_astec"

VIDEO_PATH = "HP_Overview_1.mp4"
OUTPUT_PATH = "HP_Overview_1_triton_ppe_tracking.mp4"

TARGET_SIZE = 640
PAD_VALUE = 114

# === DETECTION THRESHOLDS ===
PERSON_CONF_THRES = 0.16
PPE_CONF_THRES = 0.10
IOU_THRES = 0.45  # not used in Triton output, kept for parity

# === STABILITY SETTINGS ===
TRACK_HISTORY = 20
PPE_VOTE_THRESHOLD = 0.4
MISSING_TOLERANCE = 8
IOU_MATCH_THRESHOLD = 0.3
BOX_SMOOTH_ALPHA = 0.7

# === EXTRA NMS (SECOND STAGE) ===
PERSON_NMS_IOU = 0.3
PPE_NMS_IOU = 0.25

CLASS_MAP = {
    0: "glasses", 1: "gloves", 2: "helmet",
    3: "no-glasses", 4: "no-gloves", 5: "no-helmet",
    6: "no-shoes", 7: "no-vest", 8: "shoes",
    9: "vest", 10: "person"
}

PPE_PRESENT_CLASSES = {"glasses", "gloves", "helmet", "shoes", "vest"}
PPE_ABSENT_CLASSES = {"no-glasses", "no-gloves", "no-helmet", "no-shoes", "no-vest"}
PPE_ITEMS = ["helmet", "glasses", "vest", "gloves", "shoes"]

DISPLAY_NAME = {
    "helmet": "Hardhat", "glasses": "Goggles", "vest": "Vest",
    "gloves": "Gloves", "shoes": "Shoes"
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

        if conf < min(PERSON_CONF_THRES, PPE_CONF_THRES):
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

        dets.append({"bbox": [int(x1), int(y1), int(x2), int(y2)], "conf": float(conf), "cls": cls})

    return dets


# -------------------------
# TRACKED PERSON CLASS
# -------------------------
@dataclass
class TrackedPerson:
    track_id: int
    box: Tuple[int, int, int, int]
    ppe_history: Dict[str, Deque[Optional[bool]]] = field(default_factory=dict)
    ppe_confidence: Dict[str, Deque[float]] = field(default_factory=dict)
    frames_missing: int = 0
    smoothed_box: Tuple[int, int, int, int] = None

    def __post_init__(self):
        for item in PPE_ITEMS:
            self.ppe_history[item] = deque(maxlen=TRACK_HISTORY)
            self.ppe_confidence[item] = deque(maxlen=TRACK_HISTORY)
        self.smoothed_box = self.box

    def update_box(self, new_box: Tuple[int, int, int, int]):
        if self.smoothed_box is None:
            self.smoothed_box = new_box
        else:
            self.smoothed_box = tuple(
                int(BOX_SMOOTH_ALPHA * new + (1 - BOX_SMOOTH_ALPHA) * old)
                for new, old in zip(new_box, self.smoothed_box)
            )
        self.box = new_box
        self.frames_missing = 0

    def add_ppe_observation(self, item: str, detected: bool, confidence: float = 1.0):
        self.ppe_history[item].append(detected)
        self.ppe_confidence[item].append(confidence)

    def fill_missing_ppe_observations(self, observed_items: set):
        for item in PPE_ITEMS:
            if item not in observed_items:
                self.ppe_history[item].append(None)
                self.ppe_confidence[item].append(0.0)

    def get_stable_ppe_status(self) -> Dict[str, bool]:
        status = {}
        for item in PPE_ITEMS:
            history = list(self.ppe_history[item])
            confidences = list(self.ppe_confidence[item])

            if not history:
                status[item] = False
                continue

            weighted_true = 0.0
            weighted_false = 0.0
            total_weight = 0.0

            for val, conf in zip(history, confidences):
                if val is None:
                    continue
                weight = max(conf, 0.1)
                total_weight += weight
                if val:
                    weighted_true += weight
                else:
                    weighted_false += weight

            if total_weight == 0:
                status[item] = False
            else:
                true_ratio = weighted_true / total_weight
                status[item] = true_ratio >= PPE_VOTE_THRESHOLD

        return status

    def mark_missing(self):
        self.frames_missing += 1

    def is_expired(self) -> bool:
        return self.frames_missing > MISSING_TOLERANCE


# -------------------------
# PERSON TRACKER
# -------------------------
class PersonTracker:
    def __init__(self):
        self.tracked_persons: Dict[int, TrackedPerson] = {}
        self.next_id = 0

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

    def _is_inside(self, inner: tuple, outer: tuple) -> bool:
        xA = max(inner[0], outer[0])
        yA = max(inner[1], outer[1])
        xB = min(inner[2], outer[2])
        yB = min(inner[3], outer[3])
        if xB <= xA or yB <= yA:
            return False
        inter_area = (xB - xA) * (yB - yA)
        inner_area = (inner[2] - inner[0]) * (inner[3] - inner[1])
        return inter_area / inner_area > 0.5 if inner_area > 0 else False

    def _get_vertical_position(self, ppe_box: tuple, person_box: tuple) -> str:
        ppe_center_y = (ppe_box[1] + ppe_box[3]) / 2
        person_height = person_box[3] - person_box[1]
        person_top = person_box[1]
        relative_y = (ppe_center_y - person_top) / person_height if person_height > 0 else 0.5
        if relative_y < 0.33:
            return "top"
        elif relative_y < 0.66:
            return "middle"
        else:
            return "bottom"

    def update(self, detected_persons: List[tuple], ppe_detections: List[dict]) -> List[TrackedPerson]:
        for person in self.tracked_persons.values():
            person.mark_missing()

        matched_track_ids = set()
        matched_det_indices = set()

        matches = []
        for det_idx, det_box in enumerate(detected_persons):
            for track_id, tracked in self.tracked_persons.items():
                iou_score = self._iou(det_box, tracked.smoothed_box)
                if iou_score > IOU_MATCH_THRESHOLD:
                    matches.append((iou_score, det_idx, track_id))

        matches.sort(reverse=True, key=lambda x: x[0])

        for iou_score, det_idx, track_id in matches:
            if det_idx in matched_det_indices or track_id in matched_track_ids:
                continue
            self.tracked_persons[track_id].update_box(detected_persons[det_idx])
            matched_track_ids.add(track_id)
            matched_det_indices.add(det_idx)

        for det_idx, det_box in enumerate(detected_persons):
            if det_idx not in matched_det_indices:
                new_person = TrackedPerson(track_id=self.next_id, box=det_box)
                self.tracked_persons[self.next_id] = new_person
                self.next_id += 1

        self._associate_ppe(ppe_detections)

        expired_ids = [tid for tid, p in self.tracked_persons.items() if p.is_expired()]
        for tid in expired_ids:
            del self.tracked_persons[tid]

        return [p for p in self.tracked_persons.values() if p.frames_missing == 0]

    def _associate_ppe(self, ppe_detections: List[dict]):
        person_observed_ppe: Dict[int, set] = {tid: set() for tid in self.tracked_persons}

        for ppe in ppe_detections:
            ppe_box = ppe["box"]
            ppe_name = ppe["name"]
            ppe_conf = ppe.get("confidence", 1.0)

            if ppe_name.startswith("no-"):
                base_item = ppe_name.replace("no-", "")
                is_present = False
            else:
                base_item = ppe_name
                is_present = True

            if base_item not in PPE_ITEMS:
                continue

            best_person_id = None
            best_score = 0

            for track_id, tracked in self.tracked_persons.items():
                if tracked.frames_missing > 0:
                    continue

                p_box = tracked.smoothed_box
                iou_score = self._iou(ppe_box, p_box)
                inside = self._is_inside(ppe_box, p_box)
                score = iou_score + (0.5 if inside else 0)

                if inside:
                    position = self._get_vertical_position(ppe_box, p_box)
                    if base_item in ["helmet", "glasses"] and position == "top":
                        score += 0.3
                    elif base_item == "vest" and position == "middle":
                        score += 0.3
                    elif base_item in ["shoes"] and position == "bottom":
                        score += 0.3

                if score > best_score:
                    best_score = score
                    best_person_id = track_id

            if best_person_id is not None and best_score > 0.1:
                self.tracked_persons[best_person_id].add_ppe_observation(
                    base_item, is_present, ppe_conf
                )
                person_observed_ppe[best_person_id].add(base_item)

        for track_id, tracked in self.tracked_persons.items():
            if tracked.frames_missing == 0:
                tracked.fill_missing_ppe_observations(person_observed_ppe[track_id])


# -------------------------
# VISUALIZATION
# -------------------------
def draw_annotations(frame: np.ndarray, tracked_persons: List[TrackedPerson],
                     frame_width: int, frame_height: int) -> np.ndarray:
    for person in tracked_persons:
        x1, y1, x2, y2 = person.smoothed_box
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame_width, x2), min(frame_height, y2)

        ppe_status = person.get_stable_ppe_status()
        full_ppe = all(ppe_status.values())
        box_color = (0, 255, 0) if full_ppe else (0, 0, 255)

        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 3)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        padding = 6
        line_h = 22

        lines = []
        for item in PPE_ITEMS:
            label = f"{DISPLAY_NAME[item]}: {'Y' if ppe_status[item] else 'N'}"
            lines.append((label, ppe_status[item]))

        max_w = max(cv2.getTextSize(t, font, font_scale, thickness)[0][0] for t, _ in lines)
        block_h = len(lines) * line_h + padding
        block_w = max_w + padding * 2

        block_x1 = max(0, x1)
        block_y1 = max(0, y1 - block_h - 5)
        block_x2 = min(frame_width, block_x1 + block_w)
        block_y2 = block_y1 + block_h

        if block_x2 >= frame_width:
            block_x1 = max(0, frame_width - block_w)
            block_x2 = frame_width

        if block_y2 > block_y1 and block_x2 > block_x1:
            overlay = frame[block_y1:block_y2, block_x1:block_x2].copy()
            cv2.rectangle(overlay, (0, 0), (overlay.shape[1], overlay.shape[0]), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, frame[block_y1:block_y2, block_x1:block_x2],
                           0.3, 0, frame[block_y1:block_y2, block_x1:block_x2])

        y = block_y1 + line_h
        for text, ok in lines:
            color = (0, 255, 0) if ok else (0, 0, 255)
            cv2.putText(frame, text, (block_x1 + padding, y), font, font_scale, color, thickness)
            y += line_h

    return frame


def apply_class_nms(boxes, scores, iou_thres):
    if len(boxes) == 0:
        return []

    boxes_xywh = [[x1, y1, x2 - x1, y2 - y1] for (x1, y1, x2, y2) in boxes]

    keep = cv2.dnn.NMSBoxes(
        boxes_xywh,
        scores,
        score_threshold=0.0,
        nms_threshold=iou_thres
    )

    return keep.flatten().tolist() if len(keep) > 0 else []


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

    tracker = PersonTracker()

    for frame_idx in tqdm(range(total_frames), desc="Processing"):
        ret, frame = cap.read()
        if not ret:
            break

        img, meta = preprocess(frame)
        output = triton_infer(img)
        dets = postprocess(output, meta)

        raw_person_boxes = []
        raw_person_scores = []
        ppe_by_class = {}

        for det in dets:
            x1, y1, x2, y2 = det["bbox"]
            confidence = det["conf"]
            cls_id = det["cls"]
            name = CLASS_MAP.get(cls_id, str(cls_id))

            if name == "person" and confidence >= PERSON_CONF_THRES:
                raw_person_boxes.append((x1, y1, x2, y2))
                raw_person_scores.append(confidence)

            elif name in PPE_PRESENT_CLASSES or name in PPE_ABSENT_CLASSES:
                if confidence >= PPE_CONF_THRES:
                    ppe_by_class.setdefault(name, {"boxes": [], "scores": []})
                    ppe_by_class[name]["boxes"].append((x1, y1, x2, y2))
                    ppe_by_class[name]["scores"].append(confidence)

        keep_person = apply_class_nms(raw_person_boxes, raw_person_scores, PERSON_NMS_IOU)
        detected_persons = [raw_person_boxes[i] for i in keep_person]

        ppe_detections = []
        for name, data in ppe_by_class.items():
            keep_ppe = apply_class_nms(data["boxes"], data["scores"], PPE_NMS_IOU)
            for i in keep_ppe:
                ppe_detections.append({
                    "box": data["boxes"][i],
                    "name": name,
                    "confidence": data["scores"][i]
                })

        tracked_persons = tracker.update(detected_persons, ppe_detections)
        frame = draw_annotations(frame, tracked_persons, W, H)

        out.write(frame)

    cap.release()
    out.release()
    print(f"\n✅ Saved output video to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
