#!/usr/bin/env python3
import json
import re
import time
from typing import Any

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


def _walk_values(obj: Any):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield k, v
            yield from _walk_values(v)
    elif isinstance(obj, list):
        for item in obj:
            yield from _walk_values(item)


class PersonArrivalEventBridge(Node):
    def __init__(self):
        super().__init__("person_arrival_event_bridge")

        self.declare_parameter("person_state_topic", "/zeri/person_follow/state")
        self.declare_parameter("mission_event_topic", "/zeri/mission/event")
        self.declare_parameter("stop_distance_m", 0.75)
        self.declare_parameter("stable_sec", 1.0)
        self.declare_parameter("cooldown_sec", 30.0)
        self.declare_parameter("selected_person_id", "person_follow_target")
        self.declare_parameter("source", "person_follow_depth_lidar_drive")

        self.person_state_topic = str(self.get_parameter("person_state_topic").value)
        self.mission_event_topic = str(self.get_parameter("mission_event_topic").value)
        self.stop_distance_m = float(self.get_parameter("stop_distance_m").value)
        self.stable_sec = float(self.get_parameter("stable_sec").value)
        self.cooldown_sec = float(self.get_parameter("cooldown_sec").value)
        self.selected_person_id = str(self.get_parameter("selected_person_id").value)
        self.source = str(self.get_parameter("source").value)

        self.pub = self.create_publisher(String, self.mission_event_topic, 10)
        self.sub = self.create_subscription(String, self.person_state_topic, self.cb, 10)

        self.arrival_started_at = None
        self.last_publish_at = 0.0
        self.latest_reason = ""
        self.latest_distance = None
        self.latest_payload = {}

        self.get_logger().info(
            f"person_state_topic={self.person_state_topic}, "
            f"mission_event_topic={self.mission_event_topic}, "
            f"stop_distance_m={self.stop_distance_m}, "
            f"stable_sec={self.stable_sec}, cooldown_sec={self.cooldown_sec}"
        )

    def parse_state(self, text: str):
        text = text.strip()
        if not text:
            return {}

        try:
            if text.startswith("{"):
                return json.loads(text)
        except Exception:
            pass

        # fallback: raw string 상태도 처리
        return {"raw_text": text}

    def extract_distance(self, payload: dict):
        keys = {
            "distance_m",
            "target_distance_m",
            "person_distance_m",
            "depth_m",
            "z_m",
            "range_m",
            "distance",
            "target_z",
        }

        for k, v in _walk_values(payload):
            if str(k) in keys:
                try:
                    x = float(v)
                    if 0.05 <= x <= 10.0:
                        return x
                except Exception:
                    pass

        raw = json.dumps(payload, ensure_ascii=False)
        m = re.search(r"(?:distance|depth|range|z)[^0-9\-]*([0-9]+(?:\.[0-9]+)?)", raw, re.I)
        if m:
            try:
                x = float(m.group(1))
                if 0.05 <= x <= 10.0:
                    return x
            except Exception:
                pass

        return None

    def state_says_arrived(self, payload: dict):
        raw = json.dumps(payload, ensure_ascii=False).lower()

        negative_tokens = [
            "lost",
            "no_person",
            "no target",
            "search",
            "searching",
            "not_found",
            "none",
        ]
        if any(t in raw for t in negative_tokens):
            return False, "negative_state"

        positive_tokens = [
            "arrived",
            "reached",
            "hold_person",
            "too_close",
            "stop_for_person",
            "stopped_for_person",
            "hold_position",
            "target_reached",
            "near_target",
        ]
        for t in positive_tokens:
            if t in raw:
                return True, f"state_token:{t}"

        distance = self.extract_distance(payload)
        if distance is not None and distance <= self.stop_distance_m:
            return True, f"distance_le_stop:{distance:.3f}<={self.stop_distance_m:.3f}"

        return False, "not_arrived"

    def publish_event(self):
        now = time.time()
        if now - self.last_publish_at < self.cooldown_sec:
            return

        payload = {
            "event": "arrived_at_person",
            "selected_person_id": self.selected_person_id,
            "target_context": {
                "distance_m": self.latest_distance,
                "source": self.source,
                "arrival_reason": self.latest_reason,
                "person_state_topic": self.person_state_topic,
            },
            "stamp_sec": now,
        }

        msg = String()
        msg.data = json.dumps(payload, ensure_ascii=False)
        self.pub.publish(msg)
        self.last_publish_at = now

        self.get_logger().warning(f"[ARRIVED_AT_PERSON] {msg.data}")

    def cb(self, msg: String):
        now = time.time()
        payload = self.parse_state(msg.data)
        self.latest_payload = payload
        self.latest_distance = self.extract_distance(payload)

        arrived, reason = self.state_says_arrived(payload)
        self.latest_reason = reason

        if not arrived:
            self.arrival_started_at = None
            return

        if self.arrival_started_at is None:
            self.arrival_started_at = now
            self.get_logger().info(f"arrival candidate: {reason}")
            return

        if now - self.arrival_started_at >= self.stable_sec:
            self.publish_event()


def main():
    rclpy.init()
    node = PersonArrivalEventBridge()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
