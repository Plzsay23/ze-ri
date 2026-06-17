#!/usr/bin/env python3
"""Gate base velocity while Ze-Ri HRI/VLM/VLA interaction is active.

Flow:
  person_follow -> input_cmd_topic -> this gate -> output_cmd_topic -> safety_guard -> base

When a mission event or active VLM/VLA status is observed, the gate publishes zero Twist
instead of passing person-follow commands. This prevents the base from creeping forward
while the robot is speaking, deciding, or handing over an object.
"""

from __future__ import annotations

import json
import time
from typing import Any

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String


def make_reliable_qos(depth: int = 10) -> QoSProfile:
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=depth,
        reliability=ReliabilityPolicy.RELIABLE,
    )


def zero_twist() -> Twist:
    msg = Twist()
    msg.linear.x = 0.0
    msg.linear.y = 0.0
    msg.linear.z = 0.0
    msg.angular.x = 0.0
    msg.angular.y = 0.0
    msg.angular.z = 0.0
    return msg


def parse_status_text(text: str) -> dict[str, Any]:
    text = (text or "").strip()
    if not text:
        return {}
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
        return {"value": obj}
    except Exception:
        return {"raw": text}


class HriBaseStopGate(Node):
    def __init__(self) -> None:
        super().__init__("zeri_hri_base_stop_gate")

        self.declare_parameter("input_cmd_topic", "/cmd_vel_follow_raw")
        self.declare_parameter("output_cmd_topic", "/cmd_vel_raw")
        self.declare_parameter("mission_event_topic", "/zeri/mission/event")
        self.declare_parameter("vlm_status_topic", "/zeri/vlm/inference_status")
        self.declare_parameter("vla_status_topic", "/zeri/vla/status")
        self.declare_parameter("state_topic", "/zeri/hri_base_stop/state")
        self.declare_parameter("publish_hz", 20.0)
        self.declare_parameter("hri_hold_sec", 120.0)
        self.declare_parameter("min_hold_sec", 5.0)
        self.declare_parameter("release_on_terminal_status", True)

        self.input_cmd_topic = str(self.get_parameter("input_cmd_topic").value)
        self.output_cmd_topic = str(self.get_parameter("output_cmd_topic").value)
        self.mission_event_topic = str(self.get_parameter("mission_event_topic").value)
        self.vlm_status_topic = str(self.get_parameter("vlm_status_topic").value)
        self.vla_status_topic = str(self.get_parameter("vla_status_topic").value)
        self.state_topic = str(self.get_parameter("state_topic").value)
        self.publish_hz = float(self.get_parameter("publish_hz").value)
        self.hri_hold_sec = float(self.get_parameter("hri_hold_sec").value)
        self.min_hold_sec = float(self.get_parameter("min_hold_sec").value)
        self.release_on_terminal_status = bool(self.get_parameter("release_on_terminal_status").value)

        qos = make_reliable_qos(10)
        self.cmd_sub = self.create_subscription(Twist, self.input_cmd_topic, self.cmd_callback, qos)
        self.event_sub = self.create_subscription(String, self.mission_event_topic, self.mission_event_callback, qos)
        self.vlm_sub = self.create_subscription(String, self.vlm_status_topic, self.vlm_status_callback, qos)
        self.vla_sub = self.create_subscription(String, self.vla_status_topic, self.vla_status_callback, qos)

        self.cmd_pub = self.create_publisher(Twist, self.output_cmd_topic, qos)
        self.state_pub = self.create_publisher(String, self.state_topic, qos)

        self.hri_active = False
        self.active_since = 0.0
        self.last_active_update = 0.0
        self.reason = "idle"
        self.last_status_publish = 0.0

        period = 1.0 / max(1.0, self.publish_hz)
        self.timer = self.create_timer(period, self.timer_callback)

        self.get_logger().info("HRI base stop gate ready.")
        self.get_logger().info(f"  input_cmd_topic:   {self.input_cmd_topic}")
        self.get_logger().info(f"  output_cmd_topic:  {self.output_cmd_topic}")
        self.get_logger().info(f"  mission_event:     {self.mission_event_topic}")
        self.get_logger().info(f"  vlm_status_topic:  {self.vlm_status_topic}")
        self.get_logger().info(f"  vla_status_topic:  {self.vla_status_topic}")
        self.get_logger().info(f"  state_topic:       {self.state_topic}")
        self.publish_state(force=True)

    def activate(self, reason: str) -> None:
        now = time.time()
        if not self.hri_active:
            self.active_since = now
            self.get_logger().warn(f"[HRI BASE STOP] activated: {reason}")
        self.hri_active = True
        self.last_active_update = now
        self.reason = reason
        self.cmd_pub.publish(zero_twist())
        self.publish_state(force=True)

    def maybe_release(self, reason: str) -> None:
        if not self.hri_active:
            return
        now = time.time()
        if now - self.active_since < self.min_hold_sec:
            return
        self.hri_active = False
        self.reason = reason
        self.cmd_pub.publish(zero_twist())
        self.get_logger().warn(f"[HRI BASE STOP] released: {reason}")
        self.publish_state(force=True)

    def cmd_callback(self, msg: Twist) -> None:
        if self.hri_active:
            self.cmd_pub.publish(zero_twist())
            return
        self.cmd_pub.publish(msg)

    def mission_event_callback(self, msg: String) -> None:
        payload = parse_status_text(msg.data)
        event = str(payload.get("event") or payload.get("type") or payload.get("raw") or msg.data)
        self.activate(f"mission_event:{event}")

    def vlm_status_callback(self, msg: String) -> None:
        text = (msg.data or "").strip().lower()
        if not text:
            return

        active_words = (
            "hri", "mission", "infer", "inference", "processing", "thinking",
            "tts", "speaking", "vla", "handoff", "verify", "request", "running",
        )
        terminal_words = (
            "idle", "done", "complete", "completed", "finished", "finish", "failed", "error",
        )

        if any(w in text for w in active_words) and not text.startswith("idle"):
            self.activate(f"vlm_status:{msg.data}")
            return

        if self.release_on_terminal_status and any(w in text for w in terminal_words):
            self.maybe_release(f"vlm_status_terminal:{msg.data}")

    def vla_status_callback(self, msg: String) -> None:
        payload = parse_status_text(msg.data)
        status = str(payload.get("status") or payload.get("reason") or payload.get("raw") or "").strip().lower()
        if not status:
            return

        active_statuses = {
            "dispatched",
            "accepted",
            "running",
            "executing",
            "awaiting_handoff_verify",
            "handoff_ready",
            "holding",
            "home_requested",
        }
        terminal_statuses = {
            "idle",
            "succeeded",
            "success",
            "failed",
            "timeout",
            "rejected",
            "home_return_finished",
            "home_finished",
            "complete",
            "completed",
            "done",
        }

        if status in active_statuses or any(w in status for w in ("running", "handoff", "verify", "home_requested")):
            self.activate(f"vla_status:{status}")
            return

        if self.release_on_terminal_status and (
            status in terminal_statuses or any(w in status for w in ("finished", "succeeded", "failed", "timeout", "rejected", "idle"))
        ):
            self.maybe_release(f"vla_status_terminal:{status}")

    def timer_callback(self) -> None:
        now = time.time()
        if self.hri_active:
            self.cmd_pub.publish(zero_twist())
            if self.hri_hold_sec > 0 and now - self.active_since >= self.hri_hold_sec:
                self.maybe_release("max_hold_timeout")

        if now - self.last_status_publish >= 1.0:
            self.publish_state(force=True)

    def publish_state(self, *, force: bool = False) -> None:
        now = time.time()
        if not force and now - self.last_status_publish < 1.0:
            return
        self.last_status_publish = now
        payload = {
            "source": "hri_base_stop_gate",
            "hri_active": bool(self.hri_active),
            "status": "holding_base_stop" if self.hri_active else "pass_through",
            "reason": self.reason,
            "active_age_sec": max(0.0, now - self.active_since) if self.hri_active else 0.0,
            "input_cmd_topic": self.input_cmd_topic,
            "output_cmd_topic": self.output_cmd_topic,
            "stamp_sec": now,
        }
        msg = String()
        msg.data = json.dumps(payload, ensure_ascii=False)
        self.state_pub.publish(msg)


def main() -> None:
    rclpy.init()
    node = HriBaseStopGate()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.cmd_pub.publish(zero_twist())
            node.destroy_node()
        finally:
            if rclpy.ok():
                rclpy.shutdown()


if __name__ == "__main__":
    main()
