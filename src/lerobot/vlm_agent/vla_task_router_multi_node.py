#!/usr/bin/env python3
"""Route Ze-Ri VLA task requests to left/right ROS-gated robot clients.

Input:
  /zeri/vla/task_request  std_msgs/String JSON

Output:
  /zeri/vla/left/command  std_msgs/String JSON
  /zeri/vla/right/command std_msgs/String JSON
  /zeri/vla/status        std_msgs/String JSON

Task request JSON examples:
  {"selected_task":"oxygen_mask_delivery","instruction":"Deliver the oxygen mask to the person.","task_duration_sec":20.0,"timeout_sec":60.0}
  {"selected_task":"radio_delivery","instruction":"Deliver the radio device to the person."}

Optional route manifest JSON:
{
  "routes": {
    "oxygen_mask_delivery": {
      "arm": "left",
      "policy_id": "oxygen_mask_delivery",
      "task": "Deliver the oxygen mask to the person.",
      "duration_sec": 20.0,
      "timeout_sec": 60.0
    },
    "radio_delivery": {
      "arm": "right",
      "policy_id": "radio_delivery",
      "task": "Deliver the radio device to the person.",
      "duration_sec": 20.0,
      "timeout_sec": 60.0
    }
  }
}
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
    "oxygen_mask_delivery": RouteSpec(
        selected_task="oxygen_mask_delivery",
        arm="left",
        policy_id="oxygen_mask_delivery",
        task="Deliver the oxygen mask to the person.",
        duration_sec=20.0,
        timeout_sec=60.0,
    ),
    "radio_delivery": RouteSpec(
        selected_task="radio_delivery",
        arm="right",
        policy_id="radio_delivery",
        task="Deliver the radio device to the person.",
        duration_sec=20.0,
        timeout_sec=60.0,
    ),
}


def _load_routes(path: str) -> dict[str, RouteSpec]:
    path = path.strip()
    if not path:
        return dict(DEFAULT_ROUTES)

    path = os.path.expanduser(path)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    routes: dict[str, RouteSpec] = {}
    for selected_task, item in data.get("routes", {}).items():
        arm = str(item.get("arm", "")).strip().lower()
        if arm not in {"left", "right"}:
            raise ValueError(f"Invalid arm for {selected_task}: {arm}")

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


class VlaTaskRouterMultiNode(Node):
    def __init__(self) -> None:
        super().__init__("zeri_vla_task_router_multi_node")

        self.declare_parameter("task_request_topic", "/zeri/vla/task_request")
        self.declare_parameter("global_status_topic", "/zeri/vla/status")
        self.declare_parameter("left_command_topic", "/zeri/vla/left/command")
        self.declare_parameter("right_command_topic", "/zeri/vla/right/command")
        self.declare_parameter("left_status_topic", "/zeri/vla/left/status")
        self.declare_parameter("right_status_topic", "/zeri/vla/right/status")
        self.declare_parameter("route_manifest_path", "")
        self.declare_parameter("reject_unknown_task", True)

        self.task_request_topic = self.get_parameter("task_request_topic").value
        self.global_status_topic = self.get_parameter("global_status_topic").value
        self.left_command_topic = self.get_parameter("left_command_topic").value
        self.right_command_topic = self.get_parameter("right_command_topic").value
        self.left_status_topic = self.get_parameter("left_status_topic").value
        self.right_status_topic = self.get_parameter("right_status_topic").value
        self.route_manifest_path = self.get_parameter("route_manifest_path").value
        self.reject_unknown_task = bool(self.get_parameter("reject_unknown_task").value)

        self.routes = _load_routes(str(self.route_manifest_path))

        self.task_sub = self.create_subscription(String, self.task_request_topic, self.task_request_callback, 10)
        self.left_status_sub = self.create_subscription(
            String, self.left_status_topic, lambda msg: self.client_status_callback("left", msg), 10
        )
        self.right_status_sub = self.create_subscription(
            String, self.right_status_topic, lambda msg: self.client_status_callback("right", msg), 10
        )

        self.left_pub = self.create_publisher(String, self.left_command_topic, 10)
        self.right_pub = self.create_publisher(String, self.right_command_topic, 10)
        self.status_pub = self.create_publisher(String, self.global_status_topic, 10)

        self.publish_status("idle", "router_ready", extra={"routes": self.describe_routes()})

        self.get_logger().info("VLA multi-task router started.")
        self.get_logger().info(f"  request:      {self.task_request_topic}")
        self.get_logger().info(f"  left command: {self.left_command_topic}")
        self.get_logger().info(f"  right command:{self.right_command_topic}")
        self.get_logger().info(f"  status:       {self.global_status_topic}")
        self.get_logger().info(f"  routes:       {self.describe_routes()}")

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
            "source": "vla_task_router_multi_node",
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
        text = text.strip()
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

            spec = RouteSpec(
                selected_task=selected_task,
                arm=str(req.get("arm") or "left").strip().lower(),
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

        command = {
            "request_id": request_id,
            "selected_task": selected_task,
            "policy_id": policy_id,
            "task_for_policy": task_for_policy,
            "duration_sec": duration_sec,
            "timeout_sec": timeout_sec,
            "source": "vla_task_router_multi_node",
        }

        out = String()
        out.data = json.dumps(command, ensure_ascii=False)

        if spec.arm == "left":
            self.left_pub.publish(out)
        elif spec.arm == "right":
            self.right_pub.publish(out)
        else:
            self.publish_status(
                "rejected",
                "invalid_route_arm",
                selected_task=selected_task,
                policy_id=policy_id,
                request_id=request_id,
                extra={"arm": spec.arm},
            )
            return

        self.publish_status(
            "dispatched",
            "command_published_to_arm_client",
            selected_task=selected_task,
            arm=spec.arm,
            policy_id=policy_id,
            request_id=request_id,
            extra={"command": command},
        )

    def client_status_callback(self, arm: str, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
        except Exception:
            payload = {"raw_status": msg.data}

        payload["source"] = "vla_task_router_multi_node"
        payload["arm"] = arm
        payload["stamp_sec_router"] = time.time()

        out = String()
        out.data = json.dumps(payload, ensure_ascii=False)
        self.status_pub.publish(out)
        self.get_logger().info(f"[VLA CLIENT STATUS][{arm}] {out.data}")


def main() -> None:
    rclpy.init()
    node = VlaTaskRouterMultiNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
