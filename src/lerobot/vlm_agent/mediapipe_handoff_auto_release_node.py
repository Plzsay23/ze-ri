#!/usr/bin/env python3
"""
MediaPipe-based automatic handoff release node for Ze-Ri.

This node does not open the robot motor serial port.
It only watches the VLA handoff image/status and publishes release_and_home.

Input:
  /zeri/vla/<arm>/status          std_msgs/String JSON
  /zeri/vla/<arm>/handoff_image   sensor_msgs/Image

Output:
  /zeri/vla/<arm>/release_and_home std_msgs/String
  /zeri/vla/<arm>/auto_handoff_status std_msgs/String JSON
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String


@dataclass
class HandBox:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float

    @property
    def cx(self) -> float:
        return 0.5 * (self.x1 + self.x2)

    @property
    def cy(self) -> float:
        return 0.5 * (self.y1 + self.y2)


class MediaPipeHandoffAutoRelease(Node):
    def __init__(self) -> None:
        super().__init__("mediapipe_handoff_auto_release")

        self.declare_parameter("arm", "left")
        self.declare_parameter("status_topic", "")
        self.declare_parameter("image_topic", "")
        self.declare_parameter("release_topic", "")
        self.declare_parameter("auto_status_topic", "")
        self.declare_parameter("roi", [0.05, 0.05, 0.95, 0.95])
        self.declare_parameter("stable_frames_required", 5)
        self.declare_parameter("min_detection_confidence", 0.45)
        self.declare_parameter("min_tracking_confidence", 0.45)
        self.declare_parameter("max_num_hands", 2)
        self.declare_parameter("release_cooldown_sec", 5.0)
        self.declare_parameter("debug_period_sec", 0.5)

        # If true, the node can release during VLA "running" state.
        # This avoids waiting for duration_sec/awaiting_handoff_verify when the arm has already reached handoff pose.
        self.declare_parameter("allow_release_while_running", True)
        self.declare_parameter("min_run_sec_before_release", 7.0)
        self.declare_parameter("stable_hand_sec_required", 1.0)

        self.arm = str(self.get_parameter("arm").value).strip() or "left"

        status_topic = str(self.get_parameter("status_topic").value).strip()
        image_topic = str(self.get_parameter("image_topic").value).strip()
        release_topic = str(self.get_parameter("release_topic").value).strip()
        auto_status_topic = str(self.get_parameter("auto_status_topic").value).strip()

        self.status_topic = status_topic or f"/zeri/vla/{self.arm}/status"
        self.image_topic = image_topic or f"/zeri/vla/{self.arm}/handoff_image"
        self.release_topic = release_topic or f"/zeri/vla/{self.arm}/release_and_home"
        self.auto_status_topic = auto_status_topic or f"/zeri/vla/{self.arm}/auto_handoff_status"

        self.roi = self._parse_roi(self.get_parameter("roi").value)
        self.stable_frames_required = int(self.get_parameter("stable_frames_required").value)
        self.release_cooldown_sec = float(self.get_parameter("release_cooldown_sec").value)
        self.debug_period_sec = float(self.get_parameter("debug_period_sec").value)
        self.allow_release_while_running = bool(self.get_parameter("allow_release_while_running").value)
        self.min_run_sec_before_release = float(self.get_parameter("min_run_sec_before_release").value)
        self.stable_hand_sec_required = float(self.get_parameter("stable_hand_sec_required").value)

        min_det = float(self.get_parameter("min_detection_confidence").value)
        min_track = float(self.get_parameter("min_tracking_confidence").value)
        max_hands = int(self.get_parameter("max_num_hands").value)

        try:
            import mediapipe as mp
        except Exception as e:
            raise RuntimeError(
                "mediapipe is not importable in this Python environment. "
                "Run this node in the venv where mediapipe is installed."
            ) from e

        self.mp = mp
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            min_detection_confidence=min_det,
            min_tracking_confidence=min_track,
        )

        self.current_request_id: str | None = None
        self.current_policy_id: str | None = None
        self.current_task: str | None = None
        self.vla_ready_for_handoff = False
        self.released_for_request: str | None = None
        self.last_release_at = 0.0
        self.stable_frames = 0
        self.hand_seen_since: float | None = None
        self.request_started_at: float | None = None
        self.last_debug_at = 0.0

        self.release_pub = self.create_publisher(String, self.release_topic, 10)
        self.auto_status_pub = self.create_publisher(String, self.auto_status_topic, 10)

        self.create_subscription(String, self.status_topic, self._status_cb, 10)
        self.create_subscription(Image, self.image_topic, self._image_cb, 1)

        self.get_logger().info(
            f"MediaPipe auto handoff started | arm={self.arm} | "
            f"status={self.status_topic} | image={self.image_topic} | "
            f"release={self.release_topic} | roi={self.roi} | "
            f"stable_frames_required={self.stable_frames_required}"
        )
        self._publish_status("ready", "node_started")

    @staticmethod
    def _parse_roi(value: Any) -> tuple[float, float, float, float]:
        try:
            if isinstance(value, str):
                v = json.loads(value)
            else:
                v = value

            if isinstance(v, (list, tuple)) and len(v) == 4:
                x1, y1, x2, y2 = [float(x) for x in v]
                return (
                    max(0.0, min(1.0, x1)),
                    max(0.0, min(1.0, y1)),
                    max(0.0, min(1.0, x2)),
                    max(0.0, min(1.0, y2)),
                )
        except Exception:
            pass
        return (0.05, 0.05, 0.95, 0.95)

    def _publish_status(self, status: str, reason: str, extra: dict[str, Any] | None = None) -> None:
        payload: dict[str, Any] = {
            "source": "mediapipe_handoff_auto_release_node",
            "arm": self.arm,
            "status": status,
            "reason": reason,
            "request_id": self.current_request_id,
            "policy_id": self.current_policy_id,
            "task": self.current_task,
            "vla_ready_for_handoff": self.vla_ready_for_handoff,
            "stable_frames": self.stable_frames,
            "stable_hand_sec": 0.0 if self.hand_seen_since is None else max(0.0, time.time() - self.hand_seen_since),
            "request_elapsed_sec": None if self.request_started_at is None else max(0.0, time.time() - self.request_started_at),
            "stamp_sec": time.time(),
        }
        if extra:
            payload.update(extra)
        msg = String()
        msg.data = json.dumps(payload, ensure_ascii=False)
        self.auto_status_pub.publish(msg)
        self.get_logger().info(msg.data)

    @staticmethod
    def _parse_json_msg(msg: String) -> dict[str, Any] | None:
        try:
            return json.loads(str(msg.data))
        except Exception:
            return None

    def _status_cb(self, msg: String) -> None:
        payload = self._parse_json_msg(msg)
        if payload is None:
            return

        status = str(payload.get("status") or "")
        request_id = payload.get("request_id")
        policy_id = payload.get("policy_id")
        task = payload.get("task")

        if request_id and request_id != self.current_request_id:
            self.current_request_id = str(request_id)
            self.current_policy_id = str(policy_id) if policy_id is not None else None
            self.current_task = str(task) if task is not None else None
            self.released_for_request = None
            self.stable_frames = 0
            self.hand_seen_since = None
            self.request_started_at = time.time()

        if status in {"accepted", "running"}:
            elapsed = 0.0 if self.request_started_at is None else time.time() - self.request_started_at
            if self.allow_release_while_running and elapsed >= self.min_run_sec_before_release:
                if not self.vla_ready_for_handoff:
                    self.vla_ready_for_handoff = True
                    self.stable_frames = 0
                    self.hand_seen_since = None
                    self._publish_status(
                        "waiting_for_hand",
                        "running_release_window_open",
                        {"min_run_sec_before_release": self.min_run_sec_before_release},
                    )
            else:
                self.vla_ready_for_handoff = False
                self.stable_frames = 0
                self.hand_seen_since = None
            return

        if status in {"handoff_pose_reached", "awaiting_handoff_verify"}:
            if not self.vla_ready_for_handoff:
                self.vla_ready_for_handoff = True
                self.stable_frames = 0
                self._publish_status("waiting_for_hand", "vla_handoff_pose_ready")
            return

        if status in {
            "home_return_started",
            "home_return_finished",
            "succeeded",
            "failed",
            "timeout",
            "rejected",
            "stopped",
            "idle",
        }:
            self.vla_ready_for_handoff = False
            self.stable_frames = 0
            self.hand_seen_since = None

    def _ros_image_to_rgb(self, msg: Image) -> np.ndarray | None:
        try:
            h, w = int(msg.height), int(msg.width)
            enc = str(msg.encoding).lower()
            data = np.frombuffer(bytes(msg.data), dtype=np.uint8)

            if enc in {"rgb8", "bgr8"}:
                arr = data.reshape(h, w, 3)
                if enc == "bgr8":
                    arr = arr[..., ::-1]
                return np.ascontiguousarray(arr)

            if enc in {"rgba8", "bgra8"}:
                arr = data.reshape(h, w, 4)[..., :3]
                if enc == "bgra8":
                    arr = arr[..., ::-1]
                return np.ascontiguousarray(arr)

            if enc in {"mono8", "8uc1"}:
                arr = data.reshape(h, w)
                return np.repeat(arr[..., None], 3, axis=2)

        except Exception as e:
            self.get_logger().warning(f"failed to convert image: {e}")

        return None

    def _detect_hands(self, rgb: np.ndarray) -> list[HandBox]:
        h, w = rgb.shape[:2]
        result = self.hands.process(rgb)

        if not result.multi_hand_landmarks:
            return []

        boxes: list[HandBox] = []
        for hand_lm in result.multi_hand_landmarks:
            xs = [float(p.x) * w for p in hand_lm.landmark]
            ys = [float(p.y) * h for p in hand_lm.landmark]
            boxes.append(
                HandBox(
                    x1=max(0.0, min(xs)),
                    y1=max(0.0, min(ys)),
                    x2=min(float(w - 1), max(xs)),
                    y2=min(float(h - 1), max(ys)),
                    confidence=1.0,
                )
            )
        return boxes

    def _inside_roi(self, box: HandBox, rgb: np.ndarray) -> bool:
        h, w = rgb.shape[:2]
        x1n, y1n, x2n, y2n = self.roi
        x1, y1, x2, y2 = x1n * w, y1n * h, x2n * w, y2n * h
        return x1 <= box.cx <= x2 and y1 <= box.cy <= y2

    def _image_cb(self, msg: Image) -> None:
        if not self.vla_ready_for_handoff:
            return

        if self.current_request_id and self.released_for_request == self.current_request_id:
            return

        now = time.time()
        if now - self.last_release_at < self.release_cooldown_sec:
            return

        rgb = self._ros_image_to_rgb(msg)
        if rgb is None:
            return

        boxes = self._detect_hands(rgb)
        boxes_in_roi = [b for b in boxes if self._inside_roi(b, rgb)]

        if boxes_in_roi:
            self.stable_frames += 1
            if self.hand_seen_since is None:
                self.hand_seen_since = now

            stable_hand_sec = now - self.hand_seen_since
            best = max(boxes_in_roi, key=lambda b: (b.x2 - b.x1) * (b.y2 - b.y1))

            if now - self.last_debug_at >= self.debug_period_sec:
                self.last_debug_at = now
                self._publish_status(
                    "hand_detected",
                    "hand_inside_roi",
                    {
                        "num_hands": len(boxes),
                        "num_hands_in_roi": len(boxes_in_roi),
                        "stable_hand_sec": round(stable_hand_sec, 3),
                        "stable_hand_sec_required": self.stable_hand_sec_required,
                        "bbox": [
                            round(best.x1, 1),
                            round(best.y1, 1),
                            round(best.x2, 1),
                            round(best.y2, 1),
                        ],
                    },
                )
        else:
            self.stable_frames = 0
            self.hand_seen_since = None
            if now - self.last_debug_at >= self.debug_period_sec:
                self.last_debug_at = now
                self._publish_status(
                    "waiting_for_hand",
                    "no_hand_in_roi",
                    {"num_hands": len(boxes)},
                )
            return

        stable_hand_sec = 0.0 if self.hand_seen_since is None else now - self.hand_seen_since
        if self.stable_frames < self.stable_frames_required:
            return
        if stable_hand_sec < self.stable_hand_sec_required:
            return

        out = String()
        out.data = "release_and_home"
        self.release_pub.publish(out)

        self.last_release_at = now
        self.released_for_request = self.current_request_id
        self.vla_ready_for_handoff = False

        self._publish_status(
            "release_and_home_requested",
            "stable_hand_detected",
            {"stable_frames_required": self.stable_frames_required},
        )


def main() -> None:
    rclpy.init()
    node = MediaPipeHandoffAutoRelease()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
