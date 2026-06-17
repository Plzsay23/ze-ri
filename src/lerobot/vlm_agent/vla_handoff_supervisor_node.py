#!/usr/bin/env python3
"""Ze-Ri VLA handoff supervisor.

Sidecar ROS2 node.  It does not open the SO-101 motor serial port.  It watches
VLA client status/snapshots, verifies that the policy reached a plausible
handoff pose, detects a human hand approaching the handoff ROI, then requests
release_and_home from the VLA client.
"""

from __future__ import annotations

import json
import math
import os
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String

try:
    from lerobot.vlm_agent.handoff_detectors import HandDetection, build_hand_detector
except Exception:
    from handoff_detectors import HandDetection, build_hand_detector  # type: ignore


@dataclass
class ArmRuntime:
    arm: str
    phase: str = "idle"
    request_id: str | None = None
    policy_id: str | None = None
    task: str | None = None
    phase_started_at: float = 0.0
    last_status: dict[str, Any] = field(default_factory=dict)
    last_state: dict[str, Any] = field(default_factory=dict)
    last_rgb: np.ndarray | None = None
    hand_dist_history: deque[float] = field(default_factory=lambda: deque(maxlen=12))
    stable_hand_frames: int = 0
    released: bool = False
    last_release_at: float = 0.0


class VLAHandoffSupervisor(Node):
    def __init__(self) -> None:
        super().__init__("vla_handoff_supervisor")

        self.declare_parameter("reference_manifest_path", "config/vla/handoff_reference_manifest.json")
        self.declare_parameter("status_topic", "/zeri/vla/handoff/status")
        self.declare_parameter("grasp_verify_timeout_sec", 3.0)
        self.declare_parameter("human_hand_timeout_sec", 10.0)
        self.declare_parameter("timer_period_sec", 0.05)
        self.declare_parameter("require_reference", False)
        self.declare_parameter("require_gripper_closed", False)

        self.declare_parameter("hand_detector_type", "auto")
        self.declare_parameter("hand_model_path", "")
        self.declare_parameter("hand_detector_device", "0")
        self.declare_parameter("hand_conf_threshold", 0.40)
        self.declare_parameter("approach_window_frames", 8)
        self.declare_parameter("stable_frames_required", 10)
        self.declare_parameter("handoff_roi", "[0.2, 0.15, 0.8, 0.9]")
        self.declare_parameter("max_roi_center_distance_norm", 0.30)
        self.declare_parameter("release_debounce_sec", 2.0)

        self.declare_parameter("left_status_topic", "/zeri/vla/left/status")
        self.declare_parameter("right_status_topic", "/zeri/vla/right/status")
        self.declare_parameter("left_state_snapshot_topic", "/zeri/vla/left/state_snapshot")
        self.declare_parameter("right_state_snapshot_topic", "/zeri/vla/right/state_snapshot")
        self.declare_parameter("left_handoff_image_topic", "/zeri/vla/left/handoff_image")
        self.declare_parameter("right_handoff_image_topic", "/zeri/vla/right/handoff_image")
        self.declare_parameter("left_release_and_home_topic", "/zeri/vla/left/release_and_home")
        self.declare_parameter("right_release_and_home_topic", "/zeri/vla/right/release_and_home")
        self.declare_parameter("left_stop_topic", "/zeri/vla/left/stop")
        self.declare_parameter("right_stop_topic", "/zeri/vla/right/stop")

        self.reference_manifest_path = str(self.get_parameter("reference_manifest_path").value)
        self.grasp_verify_timeout_sec = float(self.get_parameter("grasp_verify_timeout_sec").value)
        self.human_hand_timeout_sec = float(self.get_parameter("human_hand_timeout_sec").value)
        self.require_reference = bool(self.get_parameter("require_reference").value)
        self.require_gripper_closed = bool(self.get_parameter("require_gripper_closed").value)
        self.approach_window_frames = int(self.get_parameter("approach_window_frames").value)
        self.stable_frames_required = int(self.get_parameter("stable_frames_required").value)
        self.max_roi_center_distance_norm = float(self.get_parameter("max_roi_center_distance_norm").value)
        self.release_debounce_sec = float(self.get_parameter("release_debounce_sec").value)
        self.roi_norm = self._parse_roi(str(self.get_parameter("handoff_roi").value))

        self.references = self._load_references(self.reference_manifest_path)

        detector_type = str(self.get_parameter("hand_detector_type").value)
        hand_model_path = str(self.get_parameter("hand_model_path").value)
        detector_device = str(self.get_parameter("hand_detector_device").value)
        hand_conf_threshold = float(self.get_parameter("hand_conf_threshold").value)
        self.hand_detector = build_hand_detector(
            detector_type=detector_type,
            hand_model_path=hand_model_path,
            conf_threshold=hand_conf_threshold,
            device=detector_device,
        )
        self.get_logger().info(
            f"hand detector={self.hand_detector.__class__.__name__} | "
            f"reference_manifest={self.reference_manifest_path}"
        )

        self.arms = {"left": ArmRuntime("left"), "right": ArmRuntime("right")}
        self.status_pub = self.create_publisher(String, str(self.get_parameter("status_topic").value), 10)
        self.release_pubs = {
            "left": self.create_publisher(String, str(self.get_parameter("left_release_and_home_topic").value), 10),
            "right": self.create_publisher(String, str(self.get_parameter("right_release_and_home_topic").value), 10),
        }
        self.stop_pubs = {
            "left": self.create_publisher(String, str(self.get_parameter("left_stop_topic").value), 10),
            "right": self.create_publisher(String, str(self.get_parameter("right_stop_topic").value), 10),
        }

        self.create_subscription(String, str(self.get_parameter("left_status_topic").value), lambda m: self._status_cb("left", m), 10)
        self.create_subscription(String, str(self.get_parameter("right_status_topic").value), lambda m: self._status_cb("right", m), 10)
        self.create_subscription(String, str(self.get_parameter("left_state_snapshot_topic").value), lambda m: self._state_cb("left", m), 10)
        self.create_subscription(String, str(self.get_parameter("right_state_snapshot_topic").value), lambda m: self._state_cb("right", m), 10)
        self.create_subscription(Image, str(self.get_parameter("left_handoff_image_topic").value), lambda m: self._image_cb("left", m), 1)
        self.create_subscription(Image, str(self.get_parameter("right_handoff_image_topic").value), lambda m: self._image_cb("right", m), 1)

        self.create_timer(float(self.get_parameter("timer_period_sec").value), self._timer_cb)
        self._publish("idle", "handoff_supervisor_ready")

    @staticmethod
    def _parse_roi(text: str) -> tuple[float, float, float, float]:
        try:
            data = json.loads(text)
            if len(data) == 4:
                x1, y1, x2, y2 = [float(v) for v in data]
                return (max(0.0, x1), max(0.0, y1), min(1.0, x2), min(1.0, y2))
        except Exception:
            pass
        return (0.2, 0.15, 0.8, 0.9)

    def _load_references(self, path: str) -> dict[str, Any]:
        if not path:
            return {}
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            self.get_logger().warning(f"reference manifest not found: {path}")
            return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            tasks = payload.get("tasks", payload)
            if not isinstance(tasks, dict):
                return {}
            return tasks
        except Exception as e:
            self.get_logger().error(f"failed to load reference manifest {path}: {e}")
            return {}

    def _publish(self, status: str, reason: str, arm: str | None = None, extra: dict[str, Any] | None = None) -> None:
        payload = {
            "status": status,
            "reason": reason,
            "arm": arm,
            "stamp_sec": time.time(),
        }
        if arm in self.arms:
            rt = self.arms[arm]
            payload.update({"phase": rt.phase, "request_id": rt.request_id, "policy_id": rt.policy_id, "task": rt.task})
        if extra:
            payload.update(extra)
        msg = String()
        msg.data = json.dumps(payload, ensure_ascii=False)
        self.status_pub.publish(msg)
        self.get_logger().info(msg.data)

    @staticmethod
    def _parse_json_msg(msg: String) -> dict[str, Any] | None:
        try:
            return json.loads(str(msg.data))
        except Exception:
            return None

    def _status_cb(self, arm: str, msg: String) -> None:
        payload = self._parse_json_msg(msg)
        if payload is None:
            return
        rt = self.arms[arm]
        rt.last_status = payload
        status = str(payload.get("status") or "")

        if status in {"accepted", "running"}:
            rt.request_id = payload.get("request_id") or rt.request_id
            rt.policy_id = payload.get("policy_id") or rt.policy_id
            rt.task = payload.get("task") or rt.task
            rt.phase = "vla_running"
            rt.phase_started_at = time.time()
            rt.released = False
            rt.stable_hand_frames = 0
            rt.hand_dist_history.clear()
            return

        if status in {"handoff_pose_reached", "awaiting_handoff_verify"}:
            # Enter verification once.  Later heartbeats must not reset timeout.
            if rt.phase not in {"verify_grasp", "wait_human_hand", "release_requested"}:
                rt.phase = "verify_grasp"
                rt.phase_started_at = time.time()
                rt.request_id = payload.get("request_id") or rt.request_id
                rt.policy_id = payload.get("policy_id") or rt.policy_id
                rt.task = payload.get("task") or rt.task
                rt.released = False
                rt.stable_hand_frames = 0
                rt.hand_dist_history.clear()
                self._publish("verify_grasp_started", "handoff_pose_reached", arm)
            return

        if status in {"home_return_finished", "succeeded", "failed", "timeout", "rejected", "stopped"}:
            rt.phase = "idle"
            rt.phase_started_at = time.time()
            rt.stable_hand_frames = 0
            rt.hand_dist_history.clear()
            return

    def _state_cb(self, arm: str, msg: String) -> None:
        payload = self._parse_json_msg(msg)
        if payload is None:
            return
        rt = self.arms[arm]
        rt.last_state = payload
        if rt.phase == "verify_grasp":
            ok, detail = self._verify_grasp(rt)
            if ok:
                rt.phase = "wait_human_hand"
                rt.phase_started_at = time.time()
                self._publish("wait_human_hand_started", "grasp_verified", arm, detail)

    def _image_cb(self, arm: str, msg: Image) -> None:
        rgb = self._ros_image_to_rgb(msg)
        if rgb is None:
            return
        rt = self.arms[arm]
        rt.last_rgb = rgb
        if rt.phase == "wait_human_hand" and not rt.released:
            ok, detail = self._detect_handoff_intent(rt, rgb)
            if ok:
                self._request_release_and_home(arm, detail)

    @staticmethod
    def _ros_image_to_rgb(msg: Image) -> np.ndarray | None:
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
        except Exception:
            return None
        return None

    @staticmethod
    def _as_float_dict(value: Any) -> dict[str, float]:
        if not isinstance(value, dict):
            return {}
        out = {}
        for k, v in value.items():
            try:
                if isinstance(v, list) and len(v) == 1:
                    v = v[0]
                out[str(k)] = float(v)
            except Exception:
                continue
        return out

    @staticmethod
    def _lookup_feature(values: dict[str, float], key: str) -> float | None:
        if key in values:
            return values[key]
        key_base = key.removesuffix(".pos")
        for k, v in values.items():
            kb = k.removesuffix(".pos")
            if kb == key_base or k.endswith(key) or key.endswith(k):
                return v
        return None

    def _verify_grasp(self, rt: ArmRuntime) -> tuple[bool, dict[str, Any]]:
        ref = self.references.get(str(rt.policy_id or "")) or self.references.get(str(rt.task or ""))
        if not ref:
            if self.require_reference:
                return False, {"verify_reason": "missing_reference", "policy_id": rt.policy_id}
            return True, {"verify_reason": "reference_not_required"}

        state = self._as_float_dict(rt.last_state.get("action_state"))
        raw_ticks = self._as_float_dict(rt.last_state.get("raw_ticks"))

        if "final_raw_ticks_mean" in ref or "raw_ticks_mean" in ref:
            mean = self._as_float_dict(ref.get("final_raw_ticks_mean") or ref.get("raw_ticks_mean"))
            current = raw_ticks
        else:
            mean = self._as_float_dict(ref.get("final_state_mean") or ref.get("state_mean"))
            current = state

        tolerance = self._as_float_dict(ref.get("tolerance") or ref.get("final_state_tolerance"))
        if not mean:
            if self.require_reference:
                return False, {"verify_reason": "reference_has_no_mean"}
            return True, {"verify_reason": "reference_no_mean_but_allowed"}

        errors: dict[str, float] = {}
        missing: list[str] = []
        failed: list[str] = []
        score_sum = 0.0
        score_count = 0

        for key, target in mean.items():
            cur = self._lookup_feature(current, key)
            if cur is None:
                missing.append(key)
                continue
            tol = abs(float(tolerance.get(key, tolerance.get(key.removesuffix(".pos"), 0.0))))
            if tol <= 0.0:
                tol = max(50.0, abs(target) * 0.08)
            err = abs(cur - float(target))
            errors[key] = round(err, 4)
            score_sum += min(3.0, err / tol)
            score_count += 1
            if err > tol:
                failed.append(key)

        gripper_ok = True
        gripper_range = ref.get("gripper_closed_range") or {}
        if self.require_gripper_closed or gripper_range:
            gmin = gripper_range.get("min", None)
            gmax = gripper_range.get("max", None)
            if gmin is not None and gmax is not None:
                gval = None
                for k, v in current.items():
                    if "gripper" in k.lower():
                        gval = v
                        break
                gripper_ok = gval is not None and float(gmin) <= float(gval) <= float(gmax)
            elif self.require_gripper_closed:
                gripper_ok = False

        normalized_score = score_sum / max(1, score_count)
        ok = bool(score_count > 0 and not failed and gripper_ok)
        return ok, {
            "verify_reason": "reference_compare",
            "normalized_pose_score": round(normalized_score, 4),
            "matched_features": score_count,
            "missing_features": missing,
            "failed_features": failed,
            "errors": errors,
            "gripper_ok": gripper_ok,
        }

    def _roi_pixels(self, rgb: np.ndarray) -> tuple[float, float, float, float]:
        h, w = rgb.shape[:2]
        x1, y1, x2, y2 = self.roi_norm
        return x1 * w, y1 * h, x2 * w, y2 * h

    @staticmethod
    def _inside(box: HandDetection, roi: tuple[float, float, float, float]) -> bool:
        x1, y1, x2, y2 = roi
        return x1 <= box.cx <= x2 and y1 <= box.cy <= y2

    def _detect_handoff_intent(self, rt: ArmRuntime, rgb: np.ndarray) -> tuple[bool, dict[str, Any]]:
        detections = self.hand_detector.detect(rgb)
        roi = self._roi_pixels(rgb)
        roi_cx = 0.5 * (roi[0] + roi[2])
        roi_cy = 0.5 * (roi[1] + roi[3])
        diag = math.hypot(rgb.shape[1], rgb.shape[0])

        candidates = [d for d in detections if self._inside(d, roi)]
        if not candidates:
            rt.stable_hand_frames = 0
            return False, {"hand_detected": False, "num_detections": len(detections)}

        # Pick hand closest to the handoff ROI center.
        hand = min(candidates, key=lambda d: math.hypot(d.cx - roi_cx, d.cy - roi_cy))
        dist = math.hypot(hand.cx - roi_cx, hand.cy - roi_cy)
        dist_norm = dist / max(1.0, diag)
        rt.hand_dist_history.append(dist_norm)

        close_enough = dist_norm <= self.max_roi_center_distance_norm
        approaching = False
        if len(rt.hand_dist_history) >= max(3, min(self.approach_window_frames, rt.hand_dist_history.maxlen)):
            hist = list(rt.hand_dist_history)[-self.approach_window_frames :]
            approaching = hist[0] - hist[-1] > 0.015

        if close_enough and (approaching or len(rt.hand_dist_history) < self.approach_window_frames):
            rt.stable_hand_frames += 1
        else:
            rt.stable_hand_frames = max(0, rt.stable_hand_frames - 1)

        ok = rt.stable_hand_frames >= self.stable_frames_required
        return ok, {
            "hand_detected": True,
            "hand_bbox": [round(hand.x1, 1), round(hand.y1, 1), round(hand.x2, 1), round(hand.y2, 1)],
            "hand_confidence": round(hand.confidence, 3),
            "dist_norm": round(dist_norm, 4),
            "approaching": approaching,
            "stable_hand_frames": rt.stable_hand_frames,
            "stable_frames_required": self.stable_frames_required,
        }

    def _request_release_and_home(self, arm: str, detail: dict[str, Any]) -> None:
        rt = self.arms[arm]
        now = time.time()
        if rt.released or now - rt.last_release_at < self.release_debounce_sec:
            return
        rt.released = True
        rt.last_release_at = now
        rt.phase = "release_requested"
        rt.phase_started_at = now

        msg = String()
        msg.data = "release_and_home"
        self.release_pubs[arm].publish(msg)
        self._publish("release_and_home_requested", "human_handoff_intent_detected", arm, detail)

    def _request_home_only(self, arm: str, reason: str, extra: dict[str, Any] | None = None) -> None:
        rt = self.arms[arm]
        rt.phase = "home_requested"
        rt.phase_started_at = time.time()
        msg = String()
        msg.data = "home"
        self.stop_pubs[arm].publish(msg)
        self._publish("home_requested", reason, arm, extra)

    def _timer_cb(self) -> None:
        now = time.time()
        for arm, rt in self.arms.items():
            if rt.phase == "verify_grasp" and now - rt.phase_started_at > self.grasp_verify_timeout_sec:
                ok, detail = self._verify_grasp(rt)
                if ok:
                    rt.phase = "wait_human_hand"
                    rt.phase_started_at = now
                    self._publish("wait_human_hand_started", "grasp_verified_on_timeout_check", arm, detail)
                else:
                    self._request_home_only(arm, "grasp_verify_timeout", detail)

            elif rt.phase == "wait_human_hand" and now - rt.phase_started_at > self.human_hand_timeout_sec:
                self._request_home_only(
                    arm,
                    "human_hand_timeout",
                    {"human_hand_timeout_sec": self.human_hand_timeout_sec},
                )



def main() -> None:
    from rclpy.executors import ExternalShutdownException

    rclpy.init()
    node = VLAHandoffSupervisor()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    except ExternalShutdownException:
        # Normal during ROS2 launch shutdown / Ctrl+C / parent context shutdown.
        pass

    except Exception as e:
        msg = str(e)
        if (
            "context is not valid" in msg
            or "rcl_shutdown already called" in msg
            or "failed to initialize wait set" in msg
        ):
            pass
        else:
            raise

    finally:
        try:
            node.destroy_node()
        except Exception:
            pass

        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
