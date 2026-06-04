# Copyright 2025 The HuggingFace Inc. team.
# SPDX-License-Identifier: Apache-2.0
"""
Ze-Ri VLA handoff client extension.

This module intentionally does NOT replace the existing
robot_client_ros_multi_gate_raw_home.py file.  It imports the existing client and
adds only handoff-specific ROS interfaces:

Subscribe:
  /zeri/vla/<client>/release            std_msgs/String or std_msgs/Bool
  /zeri/vla/<client>/release_and_home   std_msgs/String or std_msgs/Bool

Publish:
  /zeri/vla/<client>/handoff_image      sensor_msgs/Image
  /zeri/vla/<client>/state_snapshot     std_msgs/String JSON

The motor serial port stays owned by this VLA client.  Handoff supervisor nodes
must request release/home over ROS topics instead of opening /dev/follower_*.
"""

import json
import logging
import os
import re
import threading
import time
from collections.abc import Mapping
from dataclasses import asdict
from pprint import pformat
from queue import Queue
from typing import Any

import draccus

from lerobot.async_inference.robot_client_ros_multi_gate_raw_home import (  # type: ignore
    RosMultiPolicyGateRobotClient,
    _env_bool,
    _env_float,
    _extract_first_image,
    _extract_zeri_cli_args_to_env as _base_extract_zeri_cli_args_to_env,
    _to_pil_image,
)
from lerobot.async_inference.configs import RobotClientConfig
from lerobot.async_inference.helpers import visualize_action_queue_size
from lerobot.utils.import_utils import register_third_party_plugins


def _jsonable(value: Any) -> Any:
    """Best-effort conversion of tensors/arrays/scalars into JSON-safe values."""
    try:
        import torch

        if torch.is_tensor(value):
            tensor = value.detach().cpu()
            if tensor.numel() == 1:
                return float(tensor.item())
            return tensor.flatten().tolist()
    except Exception:
        pass

    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            if value.size == 1:
                return float(value.reshape(-1)[0])
            return value.reshape(-1).tolist()
        if isinstance(value, np.generic):
            return value.item()
    except Exception:
        pass

    if isinstance(value, (str, bool, int, float)) or value is None:
        return value
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]

    try:
        return float(value)
    except Exception:
        return repr(value)


class HandoffRosMultiPolicyGateRobotClient(RosMultiPolicyGateRobotClient):
    """ROS-gated VLA client with deterministic handoff release support."""

    def __init__(self, config: RobotClientConfig):
        super().__init__(config)

        self.zeri_release_topic = os.environ.get(
            "ZERI_RELEASE_TOPIC",
            f"/zeri/vla/{self.zeri_client_name}/release",
        ).strip()
        self.zeri_release_and_home_topic = os.environ.get(
            "ZERI_RELEASE_AND_HOME_TOPIC",
            f"/zeri/vla/{self.zeri_client_name}/release_and_home",
        ).strip()
        self.zeri_handoff_image_topic = os.environ.get(
            "ZERI_HANDOFF_IMAGE_TOPIC",
            f"/zeri/vla/{self.zeri_client_name}/handoff_image",
        ).strip()
        self.zeri_state_snapshot_topic = os.environ.get(
            "ZERI_STATE_SNAPSHOT_TOPIC",
            f"/zeri/vla/{self.zeri_client_name}/state_snapshot",
        ).strip()

        self.zeri_handoff_snapshot_hz = _env_float("ZERI_HANDOFF_SNAPSHOT_HZ", 10.0)
        self.zeri_publish_snapshot_only_after_handoff = _env_bool(
            "ZERI_PUBLISH_SNAPSHOT_ONLY_AFTER_HANDOFF",
            True,
        )

        # Prefer raw tick gripper control if configured.  This is safest for SO-101
        # because it bypasses policy action normalization during release.
        self.zeri_gripper_motor_name = os.environ.get("ZERI_GRIPPER_MOTOR_NAME", "gripper").strip()
        self.zeri_gripper_open_raw_tick = self._optional_int_env("ZERI_GRIPPER_OPEN_RAW_TICK")
        self.zeri_gripper_open_action_value = self._optional_float_env("ZERI_GRIPPER_OPEN_ACTION_VALUE")
        self.zeri_gripper_open_seconds = _env_float("ZERI_GRIPPER_OPEN_SECONDS", 0.45)
        self.zeri_gripper_open_fps = _env_float("ZERI_GRIPPER_OPEN_FPS", 25.0)
        self.zeri_release_home_delay_sec = _env_float("ZERI_RELEASE_HOME_DELAY_SEC", 0.5)

        self._ros_handoff_image_pub = None
        self._ros_state_snapshot_pub = None
        self._last_handoff_snapshot_at = 0.0
        self._release_lock = threading.Lock()

        self.logger.info(
            "[handoff-client] enabled | "
            f"release={self.zeri_release_topic} | "
            f"release_and_home={self.zeri_release_and_home_topic} | "
            f"handoff_image={self.zeri_handoff_image_topic} | "
            f"state_snapshot={self.zeri_state_snapshot_topic} | "
            f"snapshot_hz={self.zeri_handoff_snapshot_hz} | "
            f"gripper_motor={self.zeri_gripper_motor_name} | "
            f"open_raw_tick={self.zeri_gripper_open_raw_tick} | "
            f"open_action_value={self.zeri_gripper_open_action_value}"
        )

    @staticmethod
    def _optional_int_env(name: str) -> int | None:
        raw = os.environ.get(name, "").strip()
        if not raw:
            return None
        try:
            return int(round(float(raw)))
        except Exception:
            return None

    @staticmethod
    def _optional_float_env(name: str) -> float | None:
        raw = os.environ.get(name, "").strip()
        if not raw:
            return None
        try:
            return float(raw)
        except Exception:
            return None

    def _handoff_is_reached(self) -> bool:
        with self._active_lock:
            return bool(self._active_handoff_reached)

    def _active_metadata(self) -> dict[str, Any]:
        with self._active_lock:
            return {
                "client_name": self.zeri_client_name,
                "request_id": self._active_request_id or self._home_return_request_id,
                "policy_id": self._active_policy_id or self._home_return_policy_id,
                "task": self._active_task_text or self._home_return_task_text,
                "active_handoff_reached": self._active_handoff_reached,
                "elapsed_sec": round(time.time() - self._active_started_at, 3)
                if self._active_started_at > 0.0
                else None,
                "stamp_sec": time.time(),
            }

    def _extract_current_state_snapshot(self, raw_observation: dict[str, Any] | None = None) -> dict[str, Any]:
        if raw_observation is None:
            try:
                raw_observation = self.robot.get_observation()
                self._update_latest_raw_observation(raw_observation)
            except Exception as e:
                return {**self._active_metadata(), "error": f"get_observation_failed: {e}"}

        action_dict = None
        try:
            action_dict = self._extract_action_dict_from_observation(raw_observation)
        except Exception as e:
            self.logger.warning(f"[handoff-client] failed to extract action dict: {e}")

        payload = self._active_metadata()
        payload.update(
            {
                "action_features": list(getattr(self.robot, "action_features", [])),
                "observation_keys": sorted(list(raw_observation.keys())),
                "action_state": _jsonable(action_dict),
            }
        )

        # Include raw motor ticks when the bus exposes them.  Supervisor can use
        # these against raw-tick references if available.
        try:
            bus = getattr(self.robot, "bus", None)
            motors = list(getattr(bus, "motors", {}).keys()) if bus is not None else []
            if bus is not None and motors:
                try:
                    raw_ticks = bus.sync_read("Present_Position", motors, normalize=False)
                except TypeError:
                    raw_ticks = bus.sync_read("Present_Position", normalize=False)
                payload["raw_ticks"] = _jsonable(raw_ticks)
        except Exception as e:
            payload["raw_ticks_error"] = str(e)

        return payload

    def _publish_state_snapshot(self, raw_observation: dict[str, Any] | None = None) -> bool:
        pub = self._ros_state_snapshot_pub
        if pub is None:
            return False
        try:
            from std_msgs.msg import String

            msg = String()
            msg.data = json.dumps(
                self._extract_current_state_snapshot(raw_observation),
                ensure_ascii=False,
            )
            pub.publish(msg)
            return True
        except Exception as e:
            self.logger.warning(f"[handoff-client] failed to publish state snapshot: {e}")
            return False

    def _publish_handoff_image(self, raw_observation: dict[str, Any] | None = None) -> bool:
        pub = self._ros_handoff_image_pub
        if pub is None:
            return False

        if raw_observation is None:
            try:
                raw_observation = self.robot.get_observation()
                self._update_latest_raw_observation(raw_observation)
            except Exception as e:
                self.logger.warning(f"[handoff-client] get_observation failed for image: {e}")
                return False

        image = _extract_first_image(raw_observation)
        pil_image = _to_pil_image(image)
        if pil_image is None:
            return False

        try:
            stamp = self._ros_node.get_clock().now().to_msg() if self._ros_node is not None else None
            if stamp is None:
                return False
            msg = self._pil_rgb_to_ros_image_msg(
                pil_image,
                stamp=stamp,
                frame_id=f"{self.zeri_client_name}_handoff_image",
            )
            pub.publish(msg)
            return True
        except Exception as e:
            self.logger.warning(f"[handoff-client] failed to publish handoff image: {e}")
            return False

    def _publish_handoff_snapshot_timer(self) -> None:
        """Publish /zeri/vla/<arm>/handoff_image from the already-open wrist camera.

        Do NOT call self.robot.get_observation() here.
        That reads motor Present_Position and collides with the action loop.

        This path reads only OpenCVCamera.read_latest(), then wraps the frame
        with an image-compatible observation key so _extract_first_image() can find it.
        """
        if self.zeri_publish_snapshot_only_after_handoff and not self._handoff_is_reached():
            return

        now = time.time()
        min_period = 1.0 / max(0.1, self.zeri_handoff_snapshot_hz)
        if now - self._last_handoff_snapshot_at < min_period:
            return
        self._last_handoff_snapshot_at = now

        try:
            cameras = getattr(self.robot, "cameras", {}) or {}

            preferred_keys = []
            env_key = os.environ.get("ZERI_HANDOFF_CAMERA_KEY", "").strip()
            if env_key:
                preferred_keys.append(env_key)

            if self.zeri_client_name == "left":
                preferred_keys += ["cam_left", "left", "wrist_left", "camera_left"]
            elif self.zeri_client_name == "right":
                preferred_keys += ["cam_right", "cam_wright", "right", "wrist_right", "camera_right"]

            preferred_keys += list(cameras.keys())

            cam_key = None
            cam = None
            for key in preferred_keys:
                if key in cameras:
                    cam_key = key
                    cam = cameras[key]
                    break

            if cam is None:
                self.logger.warning(
                    f"[handoff-client] no camera found for handoff_image | "
                    f"client={self.zeri_client_name} | available={list(cameras.keys())}"
                )
                return

            frame = cam.read_latest(max_age_ms=1000)

            # IMPORTANT:
            # _extract_first_image() does NOT recognize "cam_left".
            # It recognizes "image" or keys containing "image".
            raw_observation = {
                "image": frame,
                "observation.images.image": frame,
                f"observation.images.{cam_key}": frame,
                cam_key: frame,
            }

            ok = self._publish_handoff_image(raw_observation)
            if not ok:
                self.logger.warning(
                    f"[handoff-client] camera frame read but handoff_image publish returned False | "
                    f"client={self.zeri_client_name} | camera={cam_key} | frame_shape={getattr(frame, 'shape', None)}"
                )
                return

            self.logger.debug(
                f"[handoff-client] published handoff_image | "
                f"client={self.zeri_client_name} | camera={cam_key} | frame_shape={getattr(frame, 'shape', None)}"
            )

        except Exception as e:
            self.logger.warning(f"[handoff-client] camera-only handoff_image publish failed: {e}")


    def _publish_wrist_snapshot(self, reason: str) -> bool:
        # Keep the original one-shot snapshot topic and also immediately publish
        # the new continuous handoff topics once handoff pose is reached.
        ok = super()._publish_wrist_snapshot(reason)
        raw_observation = self._get_latest_raw_observation()
        self._publish_handoff_image(raw_observation)
        self._publish_state_snapshot(raw_observation)
        return ok

    def _write_gripper_open_raw(self) -> bool:
        if self.zeri_gripper_open_raw_tick is None:
            return False

        bus = getattr(self.robot, "bus", None)
        if bus is None:
            self.logger.error("[release] robot has no bus; cannot raw-open gripper")
            return False

        motor = self.zeri_gripper_motor_name
        if motor not in getattr(bus, "motors", {}):
            self.logger.error(
                f"[release] gripper motor '{motor}' not found. "
                f"available={list(getattr(bus, 'motors', {}).keys())}"
            )
            return False

        try:
            try:
                current_payload = bus.sync_read("Present_Position", [motor], normalize=False)
                current = int(current_payload[motor])
            except TypeError:
                current_payload = bus.sync_read("Present_Position", normalize=False)
                current = int(current_payload[motor])

            target = int(self.zeri_gripper_open_raw_tick)
            calibration = getattr(bus, "calibration", {}) or {}
            cal = calibration.get(motor)
            if cal is not None:
                target = max(int(cal.range_min), min(int(cal.range_max), target))

            steps = max(1, int(self.zeri_gripper_open_seconds * self.zeri_gripper_open_fps))
            dt = 1.0 / max(1.0, self.zeri_gripper_open_fps)
            for step in range(steps):
                alpha = (step + 1) / steps
                tick = int(round(float(current) + alpha * (float(target) - float(current))))
                bus.sync_write("Goal_Position", {motor: tick}, normalize=False)
                time.sleep(dt)

            self.logger.warning(f"[release] raw gripper open finished | {motor}: {current} -> {target}")
            return True
        except Exception as e:
            self.logger.error(f"[release] raw gripper open failed: {e}")
            return False

    def _write_gripper_open_action(self) -> bool:
        if self.zeri_gripper_open_action_value is None:
            return False

        try:
            obs = self.robot.get_observation()
            action = self._extract_action_dict_from_observation(obs)
            if action is None:
                self.logger.error("[release] cannot build current action dict for gripper open")
                return False

            gripper_keys = [k for k in action.keys() if "gripper" in k.lower()]
            if not gripper_keys:
                self.logger.error(f"[release] no gripper key in action dict: {list(action.keys())}")
                return False

            key = gripper_keys[0]
            start = float(action[key])
            target = float(self.zeri_gripper_open_action_value)
            steps = max(1, int(self.zeri_gripper_open_seconds * self.zeri_gripper_open_fps))
            dt = 1.0 / max(1.0, self.zeri_gripper_open_fps)

            for step in range(steps):
                alpha = (step + 1) / steps
                action[key] = start + alpha * (target - start)
                self.robot.send_action(action)
                time.sleep(dt)

            self.logger.warning(f"[release] action gripper open finished | {key}: {start} -> {target}")
            return True
        except Exception as e:
            self.logger.error(f"[release] action gripper open failed: {e}")
            return False

    def _release_gripper_blocking(self, reason: str) -> bool:
        with self._release_lock:
            self._clear_action_queue_and_pause()
            self._publish_status("release_started", reason)

            ok = self._write_gripper_open_raw()
            if not ok:
                ok = self._write_gripper_open_action()

            self._publish_status(
                "release_finished" if ok else "release_failed",
                reason,
                extra={
                    "release_ok": ok,
                    "gripper_motor": self.zeri_gripper_motor_name,
                    "open_raw_tick": self.zeri_gripper_open_raw_tick,
                    "open_action_value": self.zeri_gripper_open_action_value,
                },
            )
            return ok

    def _request_release(self, *, home_after: bool, reason: str) -> None:
        def worker() -> None:
            ok = self._release_gripper_blocking(reason)
            if home_after:
                time.sleep(max(0.0, self.zeri_release_home_delay_sec))
                self._handle_stop_request(
                    "release_and_home_after_success" if ok else "release_failed_home_return"
                )

        threading.Thread(target=worker, name=f"{self.zeri_client_name}_release", daemon=True).start()

    def _ros_release_callback(self, msg: Any) -> None:
        raw = str(getattr(msg, "data", "")).strip().lower()
        if raw in {"", "1", "true", "yes", "open", "release", "열어", "전달"}:
            self._request_release(home_after=False, reason="ros_release_topic")

    def _ros_release_and_home_callback(self, msg: Any) -> None:
        raw = str(getattr(msg, "data", "")).strip().lower()
        if raw in {"", "1", "true", "yes", "open", "release", "release_and_home", "열어", "전달"}:
            self._request_release(home_after=True, reason="ros_release_and_home_topic")

    def _ros_stop_callback(self, msg: Any) -> None:
        raw = str(getattr(msg, "data", "")).strip().lower()
        if raw in {"open", "release"}:
            self._request_release(home_after=False, reason="ros_stop_topic_release_alias")
            return
        if raw in {"open_and_home", "release_and_home", "handoff_done"}:
            self._request_release(home_after=True, reason="ros_stop_topic_release_and_home_alias")
            return
        return super()._ros_stop_callback(msg)

    def prompt_router_loop(self):
        """ROS command loop with handoff release/snapshot topics."""
        try:
            import rclpy
            from std_msgs.msg import String
            from sensor_msgs.msg import Image as RosImage
        except Exception as e:
            self.logger.error(f"[handoff-client] rclpy/std_msgs/sensor_msgs import failed: {e}")
            self.logger.warning("[handoff-client] falling back to base prompt loop")
            return super().prompt_router_loop()

        try:
            rclpy.init(args=[])
            initialized_here = True
        except Exception:
            initialized_here = False

        node_name = re.sub(r"[^a-zA-Z0-9_]", "_", f"{self.zeri_client_name}_handoff_ros_gate")
        self._ros_node = rclpy.create_node(node_name)
        self._ros_status_pub = self._ros_node.create_publisher(String, self.zeri_status_topic, 10)
        self._ros_wrist_snapshot_pub = self._ros_node.create_publisher(
            RosImage,
            self.zeri_wrist_snapshot_topic,
            1,
        )
        self._ros_handoff_image_pub = self._ros_node.create_publisher(
            RosImage,
            self.zeri_handoff_image_topic,
            1,
        )
        self._ros_state_snapshot_pub = self._ros_node.create_publisher(
            String,
            self.zeri_state_snapshot_topic,
            10,
        )

        self._ros_node.create_subscription(String, self.zeri_command_topic, self._ros_command_callback, 10)
        self._ros_node.create_subscription(String, self.zeri_stop_topic, self._ros_stop_callback, 10)
        self._ros_node.create_subscription(String, self.zeri_release_topic, self._ros_release_callback, 10)
        self._ros_node.create_subscription(
            String,
            self.zeri_release_and_home_topic,
            self._ros_release_and_home_callback,
            10,
        )
        self._ros_node.create_timer(0.1, self._ros_supervisor_timer)
        self._ros_node.create_timer(0.02, self._publish_handoff_snapshot_timer)

        self._publish_status(
            "idle",
            "handoff_ros_gate_ready",
            extra={
                "command_topic": self.zeri_command_topic,
                "stop_topic": self.zeri_stop_topic,
                "release_topic": self.zeri_release_topic,
                "release_and_home_topic": self.zeri_release_and_home_topic,
                "status_topic": self.zeri_status_topic,
                "wrist_snapshot_topic": self.zeri_wrist_snapshot_topic,
                "handoff_image_topic": self.zeri_handoff_image_topic,
                "state_snapshot_topic": self.zeri_state_snapshot_topic,
                "available_policy_ids": sorted(list(self._valid_policy_ids())),
            },
        )

        self.logger.info(
            f"[handoff-client] spinning | command={self.zeri_command_topic} | "
            f"stop={self.zeri_stop_topic} | release={self.zeri_release_topic} | "
            f"release_and_home={self.zeri_release_and_home_topic} | "
            f"status={self.zeri_status_topic}"
        )

        try:
            while self.running:
                rclpy.spin_once(self._ros_node, timeout_sec=0.1)
        finally:
            try:
                self._publish_status("stopped", "handoff_ros_gate_thread_exit")
            except Exception:
                pass
            try:
                self._ros_node.destroy_node()
            except Exception:
                pass
            if initialized_here:
                try:
                    rclpy.shutdown()
                except Exception:
                    pass


def _extract_zeri_cli_args_to_env() -> None:
    """Extract base and handoff-specific ZERI args before draccus parsing."""
    import argparse
    import sys

    # First let the base module consume its known args.
    _base_extract_zeri_cli_args_to_env()

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--zeri_release_topic")
    parser.add_argument("--zeri_release_and_home_topic")
    parser.add_argument("--zeri_handoff_image_topic")
    parser.add_argument("--zeri_state_snapshot_topic")
    parser.add_argument("--zeri_handoff_snapshot_hz")
    parser.add_argument("--zeri_publish_snapshot_only_after_handoff")
    parser.add_argument("--zeri_gripper_motor_name")
    parser.add_argument("--zeri_gripper_open_raw_tick")
    parser.add_argument("--zeri_gripper_open_action_value")
    parser.add_argument("--zeri_gripper_open_seconds")
    parser.add_argument("--zeri_gripper_open_fps")
    parser.add_argument("--zeri_release_home_delay_sec")

    args, remaining = parser.parse_known_args()
    mapping = {
        "zeri_release_topic": "ZERI_RELEASE_TOPIC",
        "zeri_release_and_home_topic": "ZERI_RELEASE_AND_HOME_TOPIC",
        "zeri_handoff_image_topic": "ZERI_HANDOFF_IMAGE_TOPIC",
        "zeri_state_snapshot_topic": "ZERI_STATE_SNAPSHOT_TOPIC",
        "zeri_handoff_snapshot_hz": "ZERI_HANDOFF_SNAPSHOT_HZ",
        "zeri_publish_snapshot_only_after_handoff": "ZERI_PUBLISH_SNAPSHOT_ONLY_AFTER_HANDOFF",
        "zeri_gripper_motor_name": "ZERI_GRIPPER_MOTOR_NAME",
        "zeri_gripper_open_raw_tick": "ZERI_GRIPPER_OPEN_RAW_TICK",
        "zeri_gripper_open_action_value": "ZERI_GRIPPER_OPEN_ACTION_VALUE",
        "zeri_gripper_open_seconds": "ZERI_GRIPPER_OPEN_SECONDS",
        "zeri_gripper_open_fps": "ZERI_GRIPPER_OPEN_FPS",
        "zeri_release_home_delay_sec": "ZERI_RELEASE_HOME_DELAY_SEC",
    }
    for attr, env_name in mapping.items():
        value = getattr(args, attr)
        if value is not None:
            os.environ[env_name] = str(value)
    sys.argv = [sys.argv[0], *remaining]


@draccus.wrap()
def async_client(cfg: RobotClientConfig):
    logging.info(pformat(asdict(cfg)))

    client = HandoffRosMultiPolicyGateRobotClient(cfg)

    if client.start():
        client.logger.info("Starting action receiver thread...")
        action_receiver_thread = threading.Thread(target=client.receive_actions, daemon=True)
        prompt_router_thread = threading.Thread(target=client.prompt_router_loop, daemon=True)

        action_receiver_thread.start()
        prompt_router_thread.start()

        try:
            client.control_loop(task=cfg.task)
        finally:
            client.stop()
            action_receiver_thread.join()
            if cfg.debug_visualize_queue_size:
                visualize_action_queue_size(client.action_queue_size)
            client.logger.info("Client stopped")


if __name__ == "__main__":
    _extract_zeri_cli_args_to_env()
    register_third_party_plugins()
    async_client()
