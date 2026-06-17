#!/usr/bin/env python3
"""
YOLO-based automatic handoff release node for Ze-Ri.

It does not control motors directly.
It watches /zeri/vla/<arm>/handoff_image and publishes:
  /zeri/vla/<arm>/release_and_home
"""

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String


class YoloHandoffAutoRelease(Node):
    def __init__(self) -> None:
        super().__init__("yolo_handoff_auto_release_node")

        self.declare_parameter("arm", "left")
        self.declare_parameter("model_path", str(Path.home() / "ze-ri/models/hand_yolo.pt"))
        self.declare_parameter("device", 0)
        self.declare_parameter("conf", 0.25)
        self.declare_parameter("imgsz", 640)
        self.declare_parameter("roi", [0.0, 0.0, 1.0, 1.0])
        self.declare_parameter("stable_frames_required", 2)
        self.declare_parameter("stable_hand_sec_required", 0.2)
        self.declare_parameter("min_run_sec_before_release", 0.0)
        self.declare_parameter("allow_release_while_running", True)
        self.declare_parameter("debug_period_sec", 0.2)
        self.declare_parameter("release_cooldown_sec", 5.0)

        self.arm = str(self.get_parameter("arm").value)
        self.model_path = str(self.get_parameter("model_path").value)
        self.device = self.get_parameter("device").value
        self.conf = float(self.get_parameter("conf").value)
        self.imgsz = int(self.get_parameter("imgsz").value)
        self.roi = self._parse_roi(self.get_parameter("roi").value)

        self.stable_frames_required = int(self.get_parameter("stable_frames_required").value)
        self.stable_hand_sec_required = float(self.get_parameter("stable_hand_sec_required").value)
        self.min_run_sec_before_release = float(self.get_parameter("min_run_sec_before_release").value)
        self.allow_release_while_running = bool(self.get_parameter("allow_release_while_running").value)
        self.debug_period_sec = float(self.get_parameter("debug_period_sec").value)
        self.release_cooldown_sec = float(self.get_parameter("release_cooldown_sec").value)

        from ultralytics import YOLO

        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"YOLO hand model not found: {self.model_path}")

        self.model = YOLO(self.model_path)

        self.status_topic = f"/zeri/vla/{self.arm}/status"
        self.image_topic = f"/zeri/vla/{self.arm}/handoff_image"
        self.release_topic = f"/zeri/vla/{self.arm}/release_and_home"
        self.auto_status_topic = f"/zeri/vla/{self.arm}/auto_handoff_status"

        self.release_pub = self.create_publisher(String, self.release_topic, 10)
        self.auto_status_pub = self.create_publisher(String, self.auto_status_topic, 10)

        self.create_subscription(String, self.status_topic, self._status_cb, 10)
        self.create_subscription(Image, self.image_topic, self._image_cb, 1)

        self.current_request_id = None
        self.current_policy_id = None
        self.request_started_at = None
        self.vla_ready_for_handoff = False
        self.released_for_request = None

        self.stable_frames = 0
        self.hand_seen_since = None
        self.last_debug_at = 0.0
        self.last_release_at = 0.0

        self.get_logger().info(
            f"YOLO auto handoff started | arm={self.arm} | model={self.model_path} | "
            f"image={self.image_topic} | release={self.release_topic} | roi={self.roi}"
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
        return (0.0, 0.0, 1.0, 1.0)

    def _publish_status(self, status: str, reason: str, extra: dict[str, Any] | None = None) -> None:
        now = time.time()
        payload = {
            "source": "yolo_handoff_auto_release_node",
            "arm": self.arm,
            "status": status,
            "reason": reason,
            "request_id": self.current_request_id,
            "policy_id": self.current_policy_id,
            "vla_ready_for_handoff": self.vla_ready_for_handoff,
            "stable_frames": self.stable_frames,
            "stable_hand_sec": 0.0 if self.hand_seen_since is None else max(0.0, now - self.hand_seen_since),
            "request_elapsed_sec": None if self.request_started_at is None else max(0.0, now - self.request_started_at),
            "stamp_sec": now,
        }
        if extra:
            payload.update(extra)

        msg = String()
        msg.data = json.dumps(payload, ensure_ascii=False)
        self.auto_status_pub.publish(msg)
        self.get_logger().info(msg.data)

    @staticmethod
    def _json_loads_safe(text: str) -> dict[str, Any] | None:
        try:
            return json.loads(text)
        except Exception:
            return None

    def _status_cb(self, msg: String) -> None:
        payload = self._json_loads_safe(msg.data)
        if payload is None:
            return

        status = str(payload.get("status") or "")
        request_id = payload.get("request_id")
        policy_id = payload.get("policy_id")

        if request_id and request_id != self.current_request_id:
            self.current_request_id = str(request_id)
            self.current_policy_id = str(policy_id) if policy_id is not None else None
            self.request_started_at = time.time()
            self.released_for_request = None
            self.stable_frames = 0
            self.hand_seen_since = None

        if status in {"accepted", "running"}:
            elapsed = 0.0 if self.request_started_at is None else time.time() - self.request_started_at
            if self.allow_release_while_running and elapsed >= self.min_run_sec_before_release:
                if not self.vla_ready_for_handoff:
                    self.vla_ready_for_handoff = True
                    self._publish_status("waiting_for_hand", "running_release_window_open")
            else:
                self.vla_ready_for_handoff = False
                self.stable_frames = 0
                self.hand_seen_since = None
            return

        if status in {"handoff_pose_reached", "awaiting_handoff_verify"}:
            if not self.vla_ready_for_handoff:
                self.vla_ready_for_handoff = True
                self.stable_frames = 0
                self.hand_seen_since = None
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

    def _image_to_rgb(self, msg: Image) -> np.ndarray | None:
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

            self.get_logger().warning(f"unsupported image encoding: {enc}")
        except Exception as e:
            self.get_logger().warning(f"failed to convert image: {e}")

        return None

    def _bbox_in_roi(self, xyxy: list[float], image_shape: tuple[int, int, int]) -> bool:
        h, w = image_shape[:2]
        x1, y1, x2, y2 = xyxy
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        rx1, ry1, rx2, ry2 = self.roi
        return (rx1 * w) <= cx <= (rx2 * w) and (ry1 * h) <= cy <= (ry2 * h)

    def _detect_hand(self, rgb: np.ndarray) -> tuple[bool, dict[str, Any]]:
        results = self.model.predict(
            source=rgb,
            conf=self.conf,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )

        best_conf = 0.0
        best_box = None
        count = 0
        count_in_roi = 0

        if not results:
            return False, {"num_boxes": 0, "num_boxes_in_roi": 0}

        boxes = getattr(results[0], "boxes", None)
        if boxes is None:
            return False, {"num_boxes": 0, "num_boxes_in_roi": 0}

        for b in boxes:
            xyxy = b.xyxy[0].detach().cpu().numpy().astype(float).tolist()
            conf = float(b.conf[0].detach().cpu().item()) if b.conf is not None else 0.0
            count += 1

            if not self._bbox_in_roi(xyxy, rgb.shape):
                continue

            count_in_roi += 1
            if conf > best_conf:
                best_conf = conf
                best_box = xyxy

        return count_in_roi > 0, {
            "num_boxes": count,
            "num_boxes_in_roi": count_in_roi,
            "best_conf": round(best_conf, 4),
            "best_bbox": None if best_box is None else [round(x, 1) for x in best_box],
        }

    def _image_cb(self, msg: Image) -> None:
        if not self.vla_ready_for_handoff:
            return

        if self.current_request_id and self.released_for_request == self.current_request_id:
            return

        now = time.time()
        if now - self.last_release_at < self.release_cooldown_sec:
            return

        rgb = self._image_to_rgb(msg)
        if rgb is None:
            return

        detected, meta = self._detect_hand(rgb)

        if detected:
            self.stable_frames += 1
            if self.hand_seen_since is None:
                self.hand_seen_since = now

            stable_hand_sec = now - self.hand_seen_since

            if now - self.last_debug_at >= self.debug_period_sec:
                self.last_debug_at = now
                self._publish_status(
                    "hand_detected",
                    "yolo_hand_inside_roi",
                    {
                        **meta,
                        "stable_hand_sec": round(stable_hand_sec, 3),
                        "stable_hand_sec_required": self.stable_hand_sec_required,
                    },
                )
        else:
            self.stable_frames = 0
            self.hand_seen_since = None
            if now - self.last_debug_at >= self.debug_period_sec:
                self.last_debug_at = now
                self._publish_status("waiting_for_hand", "no_yolo_hand", meta)
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
            "stable_yolo_hand_detected",
            {
                "stable_frames_required": self.stable_frames_required,
                "stable_hand_sec_required": self.stable_hand_sec_required,
            },
        )


def main() -> None:
    rclpy.init()
    node = YoloHandoffAutoRelease()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
