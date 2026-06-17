"""Hand/object detection helpers for Ze-Ri VLA handoff.

All heavy dependencies are optional.  The supervisor can run in three modes:
  - Ultralytics YOLO hand model when --hand_model_path is provided.
  - MediaPipe Hands when mediapipe is installed.
  - Disabled detector mode for state-machine testing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class HandDetection:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    label: str = "hand"

    @property
    def cx(self) -> float:
        return 0.5 * (self.x1 + self.x2)

    @property
    def cy(self) -> float:
        return 0.5 * (self.y1 + self.y2)

    @property
    def area(self) -> float:
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)


class BaseHandDetector:
    def detect(self, rgb: np.ndarray) -> list[HandDetection]:
        raise NotImplementedError


class DisabledHandDetector(BaseHandDetector):
    def detect(self, rgb: np.ndarray) -> list[HandDetection]:
        return []


class YoloHandDetector(BaseHandDetector):
    def __init__(self, model_path: str, conf_threshold: float = 0.35, device: str | None = None):
        from ultralytics import YOLO

        self.model = YOLO(model_path)
        self.conf_threshold = float(conf_threshold)
        self.device = device

    def detect(self, rgb: np.ndarray) -> list[HandDetection]:
        kwargs: dict[str, Any] = {"verbose": False, "conf": self.conf_threshold}
        if self.device:
            kwargs["device"] = self.device
        results = self.model.predict(rgb, **kwargs)
        detections: list[HandDetection] = []
        if not results:
            return detections

        result = results[0]
        names = getattr(result, "names", {}) or getattr(self.model, "names", {}) or {}
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return detections

        for box in boxes:
            xyxy = box.xyxy[0].detach().cpu().numpy().astype(float).tolist()
            conf = float(box.conf[0].detach().cpu().item()) if box.conf is not None else 0.0
            cls_id = int(box.cls[0].detach().cpu().item()) if box.cls is not None else -1
            label = str(names.get(cls_id, cls_id)).lower()
            # Keep hand-specific labels, but do not reject a single-class custom model
            # whose class name is 0.
            if names and "hand" not in label and len(names) > 1:
                continue
            if conf < self.conf_threshold:
                continue
            detections.append(HandDetection(*xyxy, confidence=conf, label=label))
        return detections


class MediaPipeHandDetector(BaseHandDetector):
    def __init__(self, min_detection_confidence: float = 0.45, max_num_hands: int = 2):
        import mediapipe as mp

        self.mp = mp
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=0.45,
        )
        self.conf = float(min_detection_confidence)

    def detect(self, rgb: np.ndarray) -> list[HandDetection]:
        h, w = rgb.shape[:2]
        result = self.hands.process(rgb)
        detections: list[HandDetection] = []
        if not result.multi_hand_landmarks:
            return detections
        for lm in result.multi_hand_landmarks:
            xs = [p.x * w for p in lm.landmark]
            ys = [p.y * h for p in lm.landmark]
            detections.append(
                HandDetection(
                    x1=max(0.0, min(xs)),
                    y1=max(0.0, min(ys)),
                    x2=min(float(w - 1), max(xs)),
                    y2=min(float(h - 1), max(ys)),
                    confidence=self.conf,
                    label="hand_mediapipe",
                )
            )
        return detections


def build_hand_detector(
    detector_type: str,
    hand_model_path: str = "",
    conf_threshold: float = 0.35,
    device: str = "",
) -> BaseHandDetector:
    detector_type = (detector_type or "auto").strip().lower()
    hand_model_path = (hand_model_path or "").strip()

    if detector_type in {"none", "disabled", "off"}:
        return DisabledHandDetector()

    if detector_type in {"yolo", "auto"} and hand_model_path:
        try:
            return YoloHandDetector(hand_model_path, conf_threshold=conf_threshold, device=device or None)
        except Exception as e:
            print(f"[WARN] failed to initialize YOLO hand detector, disabling hand detector: {e}")
            return DisabledHandDetector()

    if detector_type in {"mediapipe", "auto"}:
        try:
            return MediaPipeHandDetector(min_detection_confidence=conf_threshold)
        except Exception as e:
            print(f"[WARN] failed to initialize MediaPipe hand detector, disabling hand detector: {e}")
            return DisabledHandDetector()

    return DisabledHandDetector()
