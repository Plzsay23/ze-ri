#!/usr/bin/env python3
"""Route Ze-Ri VLA task requests to one bimanual ROS-gated robot client.

Input:
  /zeri/vla/task_request      std_msgs/String JSON

Output:
  /zeri/vla/bimanual/command  std_msgs/String JSON
  /zeri/vla/status            std_msgs/String JSON

The bimanual client opens shared cameras such as /dev/top only once. Each route
still carries an arm field so single-arm policy fallback can map actions to the
left or right arm when possible.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


@dataclass(frozen=True)
class RouteSpec:
    selected_task: str
    arm: str
    policy_id: str
    task: str
    duration_sec: float
    timeout_sec: float


DEFAULT_ROUTES: dict[str, RouteSpec] = {
    "water_delivery": RouteSpec(
        selected_task="water_delivery",
        arm="right",
        policy_id="pick_water_act",
        task="Deliver the water bottle to the person.",
        duration_sec=20.0,
        timeout_sec=60.0,
    ),
    "oxygen_mask_delivery": RouteSpec(
        selected_task="oxygen_mask_delivery",
        arm="left",
        policy_id="black_act",
        task="Deliver the oxygen mask to the person.",
        duration_sec=20.0,
        timeout_sec=60.0,
    ),
    "radio_delivery": RouteSpec(
        selected_task="radio_delivery",
        arm="right",
        policy_id="blue_act",
        task="Deliver the radio device to the person.",
        duration_sec=20.0,
        timeout_sec=60.0,
    ),
    "black_act": RouteSpec(
        selected_task="black_act",
        arm="left",
        policy_id="black_act",
        task="Execute black_act.",
        duration_sec=20.0,
        timeout_sec=60.0,
    ),
    "blue_act": RouteSpec(
        selected_task="blue_act",
        arm="right",
        policy_id="blue_act",
        task="Execute blue_act.",
        duration_sec=20.0,
        timeout_sec=60.0,
    ),
}


def _normalise_arm(value: str) -> str:
    value = str(value or "").strip().lower()
    if value in {"left", "l", "left_arm", "왼팔"}:
        return "left"
    if value in {"right", "r", "right_arm", "오른팔"}:
        return "right"
    if value in {"both", "bimanual", "dual"}:
        return "both"
    raise ValueError(f"Invalid arm: {value}. Use left/right/both.")


def _load_routes(path: str) -> dict[str, RouteSpec]:
    path = str(path or "").strip()
    if not path:
        return dict(DEFAULT_ROUTES)

    path = os.path.expanduser(path)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    routes: dict[str, RouteSpec] = {}
    for selected_task, item in data.get("routes", {}).items():
        arm = _normalise_arm(str(item.get("arm", "left")))
        policy_id = str(item.get("policy_id") or selected_task).strip()
        task = str(item.get("task") or item.get("instruction") or selected_task).strip()
        duration_sec = float(item.get("duration_sec", item.get("task_duration_sec", 20.0)))
        timeout_sec = float(item.get("timeout_sec", 60.0))

        routes[selected_task] = RouteSpec(
            selected_task=selected_task,
            arm=arm,
            policy_id=policy_id,
            task=task,
            duration_sec=duration_sec,
            timeout_sec=timeout_sec,
        )

    return routes or dict(DEFAULT_ROUTES)


class VlaTaskRouterBimanualNode(Node):
    def __init__(self) -> None:
        super().__init__("zeri_vla_task_router_bimanual_node")

        self.declare_parameter("task_request_topic", "/zeri/vla/task_request")
        self.declare_parameter("bimanual_command_topic", "/zeri/vla/bimanual/command")
        self.declare_parameter("bimanual_status_topic", "/zeri/vla/bimanual/status")
        self.declare_parameter("global_status_topic", "/zeri/vla/status")
        self.declare_parameter("route_manifest_path", "")
        self.declare_parameter("reject_unknown_task", True)

        self.task_request_topic = str(self.get_parameter("task_request_topic").value)
        self.bimanual_command_topic = str(self.get_parameter("bimanual_command_topic").value)
        self.bimanual_status_topic = str(self.get_parameter("bimanual_status_topic").value)
        self.global_status_topic = str(self.get_parameter("global_status_topic").value)
        self.route_manifest_path = str(self.get_parameter("route_manifest_path").value)
        self.reject_unknown_task = bool(self.get_parameter("reject_unknown_task").value)

        self.routes = _load_routes(self.route_manifest_path)

        self.task_sub = self.create_subscription(String, self.task_request_topic, self.task_request_callback, 10)
        self.client_status_sub = self.create_subscription(
            String,
            self.bimanual_status_topic,
            self.client_status_callback,
            10,
        )
        self.command_pub = self.create_publisher(String, self.bimanual_command_topic, 10)
        self.status_pub = self.create_publisher(String, self.global_status_topic, 10)

        self.publish_status("idle", "router_ready", extra={"routes": self.describe_routes()})

        self.get_logger().info("VLA bimanual task router started.")
        self.get_logger().info(f"  request:  {self.task_request_topic}")
        self.get_logger().info(f"  command:  {self.bimanual_command_topic}")
        self.get_logger().info(f"  client status: {self.bimanual_status_topic}")
        self.get_logger().info(f"  global status: {self.global_status_topic}")
        self.get_logger().info(f"  routes:   {self.describe_routes()}")

    def describe_routes(self) -> dict[str, dict[str, Any]]:
        return {
            name: {
                "arm": spec.arm,
                "policy_id": spec.policy_id,
                "task": spec.task,
                "duration_sec": spec.duration_sec,
                "timeout_sec": spec.timeout_sec,
            }
            for name, spec in self.routes.items()
        }

    def publish_status(
        self,
        status: str,
        reason: str,
        *,
        selected_task: str | None = None,
        arm: str | None = None,
        policy_id: str | None = None,
        request_id: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        payload: dict[str, Any] = {
            "source": "vla_task_router_bimanual_node",
            "status": status,
            "reason": reason,
            "selected_task": selected_task,
            "arm": arm,
            "policy_id": policy_id,
            "request_id": request_id,
            "stamp_sec": time.time(),
        }
        if extra:
            payload.update(extra)

        msg = String()
        msg.data = json.dumps(payload, ensure_ascii=False)
        self.status_pub.publish(msg)
        self.get_logger().info(f"[VLA ROUTER STATUS] {msg.data}")

    @staticmethod
    def parse_task_request(text: str) -> dict[str, Any]:
        text = str(text or "").strip()
        if not text:
            return {}
        if text.startswith("{"):
            return json.loads(text)
        return {"selected_task": text, "instruction": text}

    def task_request_callback(self, msg: String) -> None:
        try:
            req = self.parse_task_request(msg.data)
        except Exception as e:
            self.publish_status("rejected", f"invalid_task_request_json: {e}")
            return

        selected_task = str(req.get("selected_task") or req.get("task") or "").strip()
        request_id = str(req.get("request_id") or req.get("task_id") or f"{selected_task}_{int(time.time() * 1000)}")

        if not selected_task:
            self.publish_status("rejected", "missing_selected_task", request_id=request_id)
            return

        spec = self.routes.get(selected_task)
        if spec is None:
            if self.reject_unknown_task:
                self.publish_status(
                    "rejected",
                    "unknown_selected_task",
                    selected_task=selected_task,
                    request_id=request_id,
                    extra={"available_tasks": sorted(self.routes)},
                )
                return

            try:
                arm = _normalise_arm(str(req.get("arm") or "left"))
            except Exception as e:
                self.publish_status("rejected", str(e), selected_task=selected_task, request_id=request_id)
                return

            spec = RouteSpec(
                selected_task=selected_task,
                arm=arm,
                policy_id=str(req.get("policy_id") or selected_task).strip(),
                task=str(req.get("instruction") or req.get("task_for_policy") or selected_task).strip(),
                duration_sec=float(req.get("duration_sec", req.get("task_duration_sec", 20.0))),
                timeout_sec=float(req.get("timeout_sec", 60.0)),
            )

        task_for_policy = str(
            req.get("task_for_policy")
            or req.get("vla_instruction")
            or req.get("instruction")
            or spec.task
        ).strip()

        duration_sec = float(req.get("duration_sec", req.get("task_duration_sec", spec.duration_sec)))
        timeout_sec = float(req.get("timeout_sec", spec.timeout_sec))
        policy_id = str(req.get("policy_id") or spec.policy_id).strip()
        arm = str(req.get("arm") or spec.arm).strip().lower()

        command = {
            "request_id": request_id,
            "selected_task": selected_task,
            "arm": arm,
            "policy_id": policy_id,
            "task_for_policy": task_for_policy,
            "duration_sec": duration_sec,
            "timeout_sec": timeout_sec,
            "source": "vla_task_router_bimanual_node",
        }

        out = String()
        out.data = json.dumps(command, ensure_ascii=False)
        self.command_pub.publish(out)

        self.publish_status(
            "dispatched",
            "command_published_to_bimanual_client",
            selected_task=selected_task,
            arm=arm,
            policy_id=policy_id,
            request_id=request_id,
            extra={"command": command},
        )

    def client_status_callback(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
        except Exception:
            payload = {"raw_status": msg.data}

        payload["source"] = "vla_task_router_bimanual_node"
        payload["stamp_sec_router"] = time.time()

        out = String()
        out.data = json.dumps(payload, ensure_ascii=False)
        self.status_pub.publish(out)
        self.get_logger().info(f"[VLA BIMANUAL CLIENT STATUS] {out.data}")


def main() -> None:
    rclpy.init()
    node = VlaTaskRouterBimanualNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
