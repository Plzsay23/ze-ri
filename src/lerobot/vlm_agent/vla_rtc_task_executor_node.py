#!/usr/bin/env python3
# src/lerobot/vlm_agent/vla_rtc_task_executor_node.py

import json
import os
import shlex
import signal
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String


SUPPORTED_TASKS = {
    "water_delivery",
    "oxygen_mask_delivery",
    "radio_delivery",
}


def make_reliable_qos(depth: int = 10) -> QoSProfile:
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=depth,
        reliability=ReliabilityPolicy.RELIABLE,
    )


def safe_float(value: Any, default: float, min_value: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = default
    return max(min_value, parsed)


def bool_to_cli(value: bool) -> str:
    return "True" if value else "False"


def split_extra_args(raw: str) -> list[str]:
    raw = str(raw or "").strip()
    if not raw:
        return []

    args = shlex.split(raw)

    # This LeRobot robot_client variant does not accept --display_data.
    # Drop it here so stale launch commands do not crash the task executor.
    filtered: list[str] = []
    for arg in args:
        if arg == "--display_data" or arg.startswith("--display_data="):
            continue
        filtered.append(arg)
    return filtered


@dataclass
class VlaTask:
    task_id: str
    selected_task: str
    instruction: str
    robot_port: str
    robot_id: str
    robot_cameras: str
    policy_path: str
    task_duration_sec: float
    timeout_sec: float


class VlaActTaskExecutorNode(Node):
    """
    VLM -> LeRobot ACT robot_client executor.

    Subscribe:
      /zeri/vla/task_request : std_msgs/String JSON

    Publish:
      /zeri/vla/status : std_msgs/String JSON

    Supported selected_task:
      - water_delivery
      - oxygen_mask_delivery
      - radio_delivery

    Expected robot_client command form:
      python -m lerobot.async_inference.robot_client \
        --robot.type=so101_follower \
        --robot.port=/dev/follower_left \
        --robot.id=follower_left \
        --robot.cameras='{top: {type: opencv, index_or_path: /dev/cam_left, width: 640, height: 480, fps: 25}}' \
        --task=oxygen_mask_delivery \
        --server_address=127.0.0.1:8080 \
        --policy_type=act \
        --pretrained_name_or_path=plzsay/pap_black \
        --policy_device=cuda \
        --actions_per_chunk=100 \
        --chunk_size_threshold=0.1 \
        --aggregate_fn_name=weighted_average \
        --debug_visualize_queue_size=True
    """

    def __init__(self) -> None:
        super().__init__("zeri_vla_act_task_executor_node")

        # ROS topics
        self.declare_parameter("task_request_topic", "/zeri/vla/task_request")
        self.declare_parameter("vla_status_topic", "/zeri/vla/status")

        # Process / module
        self.declare_parameter("python_executable", "python")
        self.declare_parameter("rtc_client_module", "lerobot.async_inference.robot_client")

        # LeRobot common options
        self.declare_parameter("server_address", "127.0.0.1:8080")
        self.declare_parameter("robot_type", "so101_follower")
        self.declare_parameter("policy_type", "act")
        self.declare_parameter("policy_device", "cuda")

        # Backward-compatible single-arm fallback parameters.
        self.declare_parameter("robot_port", "/dev/follower")
        self.declare_parameter("robot_id", "follower")
        self.declare_parameter("robot_cameras", "")

        # Task-specific robot parameters.
        self.declare_parameter("oxygen_robot_port", "/dev/follower_left")
        self.declare_parameter("oxygen_robot_id", "follower_left")
        self.declare_parameter(
            "oxygen_robot_cameras",
            "{top: {type: opencv, index_or_path: /dev/cam_left, width: 640, height: 480, fps: 25}}",
        )

        self.declare_parameter("radio_robot_port", "/dev/follower_right")
        self.declare_parameter("radio_robot_id", "follower_right")
        self.declare_parameter(
            "radio_robot_cameras",
            "{top: {type: opencv, index_or_path: /dev/cam_right, width: 640, height: 480, fps: 25}}",
        )

        # Hugging Face repo id or local path. For now both may point to pap_black.
        self.declare_parameter("oxygen_act_policy_path", "")
        self.declare_parameter("radio_act_policy_path", "")

        # Request/task text.
        self.declare_parameter("oxygen_instruction", "oxygen_mask_delivery")
        self.declare_parameter("radio_instruction", "radio_delivery")
        self.declare_parameter("use_selected_task_as_task_arg", True)

        # Async inference options matching the user's existing pipeline.
        self.declare_parameter("actions_per_chunk", 100)
        self.declare_parameter("chunk_size_threshold", 0.1)
        self.declare_parameter("aggregate_fn_name", "weighted_average")
        self.declare_parameter("debug_visualize_queue_size", True)

        # Execution policy.
        self.declare_parameter("completion_mode", "timed")  # timed | process_exit
        self.declare_parameter("default_task_duration_sec", 20.0)
        self.declare_parameter("default_timeout_sec", 60.0)
        self.declare_parameter("shutdown_grace_sec", 5.0)
        self.declare_parameter("allow_concurrent_tasks", False)
        self.declare_parameter("extra_robot_client_args", "")

        self.task_request_topic = str(self.get_parameter("task_request_topic").value)
        self.vla_status_topic = str(self.get_parameter("vla_status_topic").value)

        self.python_executable = str(self.get_parameter("python_executable").value)
        self.rtc_client_module = str(self.get_parameter("rtc_client_module").value)

        self.server_address = str(self.get_parameter("server_address").value)
        self.robot_type = str(self.get_parameter("robot_type").value)
        self.policy_type = str(self.get_parameter("policy_type").value)
        self.policy_device = str(self.get_parameter("policy_device").value)

        self.robot_port = str(self.get_parameter("robot_port").value).strip()
        self.robot_id = str(self.get_parameter("robot_id").value).strip()
        self.robot_cameras = str(self.get_parameter("robot_cameras").value).strip()

        self.oxygen_robot_port = str(self.get_parameter("oxygen_robot_port").value).strip()
        self.oxygen_robot_id = str(self.get_parameter("oxygen_robot_id").value).strip()
        self.oxygen_robot_cameras = str(self.get_parameter("oxygen_robot_cameras").value).strip()

        self.radio_robot_port = str(self.get_parameter("radio_robot_port").value).strip()
        self.radio_robot_id = str(self.get_parameter("radio_robot_id").value).strip()
        self.radio_robot_cameras = str(self.get_parameter("radio_robot_cameras").value).strip()

        self.oxygen_act_policy_path = str(self.get_parameter("oxygen_act_policy_path").value).strip()
        self.radio_act_policy_path = str(self.get_parameter("radio_act_policy_path").value).strip()

        self.oxygen_instruction = str(self.get_parameter("oxygen_instruction").value).strip()
        self.radio_instruction = str(self.get_parameter("radio_instruction").value).strip()
        self.use_selected_task_as_task_arg = bool(self.get_parameter("use_selected_task_as_task_arg").value)

        self.actions_per_chunk = str(self.get_parameter("actions_per_chunk").value)
        self.chunk_size_threshold = str(self.get_parameter("chunk_size_threshold").value)
        self.aggregate_fn_name = str(self.get_parameter("aggregate_fn_name").value)
        self.debug_visualize_queue_size = bool(
            self.get_parameter("debug_visualize_queue_size").value
        )

        self.completion_mode = str(self.get_parameter("completion_mode").value).strip()
        if self.completion_mode not in {"timed", "process_exit"}:
            self.get_logger().warn(
                f"Unknown completion_mode={self.completion_mode}. Falling back to timed."
            )
            self.completion_mode = "timed"

        self.default_task_duration_sec = float(
            self.get_parameter("default_task_duration_sec").value
        )
        self.default_timeout_sec = float(self.get_parameter("default_timeout_sec").value)
        self.shutdown_grace_sec = float(self.get_parameter("shutdown_grace_sec").value)
        self.allow_concurrent_tasks = bool(
            self.get_parameter("allow_concurrent_tasks").value
        )
        self.extra_robot_client_args = str(
            self.get_parameter("extra_robot_client_args").value
        ).strip()

        self.lock = threading.Lock()
        self.active_task_id: Optional[str] = None
        self.active_process: Optional[subprocess.Popen] = None

        qos = make_reliable_qos(depth=10)

        self.task_sub = self.create_subscription(
            String,
            self.task_request_topic,
            self.task_request_callback,
            qos,
        )
        self.status_pub = self.create_publisher(String, self.vla_status_topic, qos)

        self.log_startup_config()
        self.publish_status(
            task_id="none",
            selected_task="none",
            status="idle",
            reason="executor_ready",
        )

    def log_startup_config(self) -> None:
        self.get_logger().info("VLA ACT task executor started.")
        self.get_logger().info(f"  task_request_topic:          {self.task_request_topic}")
        self.get_logger().info(f"  vla_status_topic:            {self.vla_status_topic}")
        self.get_logger().info(f"  rtc_client_module:           {self.rtc_client_module}")
        self.get_logger().info(f"  server_address:              {self.server_address}")
        self.get_logger().info(f"  robot_type:                  {self.robot_type}")
        self.get_logger().info(f"  policy_type:                 {self.policy_type}")
        self.get_logger().info(f"  policy_device:               {self.policy_device}")
        self.get_logger().info("  task routing:")
        self.get_logger().info(f"    oxygen_robot_port:         {self.oxygen_robot_port}")
        self.get_logger().info(f"    oxygen_robot_id:           {self.oxygen_robot_id}")
        self.get_logger().info(f"    oxygen_robot_cameras:      {self.oxygen_robot_cameras}")
        self.get_logger().info(f"    oxygen_act_policy_path:    {self.oxygen_act_policy_path}")
        self.get_logger().info(f"    radio_robot_port:          {self.radio_robot_port}")
        self.get_logger().info(f"    radio_robot_id:            {self.radio_robot_id}")
        self.get_logger().info(f"    radio_robot_cameras:       {self.radio_robot_cameras}")
        self.get_logger().info(f"    radio_act_policy_path:     {self.radio_act_policy_path}")
        self.get_logger().info(f"  completion_mode:             {self.completion_mode}")
        self.get_logger().info(f"  default_task_duration_sec:   {self.default_task_duration_sec}")
        self.get_logger().info(f"  default_timeout_sec:         {self.default_timeout_sec}")
        self.get_logger().info(f"  actions_per_chunk:           {self.actions_per_chunk}")
        self.get_logger().info(f"  chunk_size_threshold:        {self.chunk_size_threshold}")
        self.get_logger().info(f"  aggregate_fn_name:           {self.aggregate_fn_name}")
        self.get_logger().info(
            f"  debug_visualize_queue_size:  {self.debug_visualize_queue_size}"
        )
        self.get_logger().info(f"  extra_robot_client_args:     {self.extra_robot_client_args}")

    def publish_status(
        self,
        task_id: str,
        selected_task: str,
        status: str,
        reason: str = "",
        robot_port: str = "",
        robot_id: str = "",
        robot_cameras: str = "",
        policy_path: str = "",
        return_code: Optional[int] = None,
        elapsed_sec: Optional[float] = None,
    ) -> None:
        payload = {
            "task_id": task_id,
            "selected_task": selected_task,
            "status": status,
            "reason": reason,
            "robot_type": self.robot_type,
            "robot_port": robot_port,
            "robot_id": robot_id,
            "robot_cameras": robot_cameras,
            "policy_type": self.policy_type,
            "policy_path": policy_path,
            "return_code": return_code,
            "elapsed_sec": elapsed_sec,
            "stamp_sec": time.time(),
        }

        msg = String()
        msg.data = json.dumps(payload, ensure_ascii=False)
        self.status_pub.publish(msg)
        self.get_logger().info(f"[VLA STATUS] {msg.data}")

    def default_instruction_for_task(self, selected_task: str) -> str:
        if self.use_selected_task_as_task_arg:
            return selected_task

        if selected_task == "oxygen_mask_delivery":
            return self.oxygen_instruction or selected_task

        if selected_task == "water_delivery":
            return selected_task

        if selected_task == "radio_delivery":
            return self.radio_instruction or selected_task

        return selected_task

    def route_for_task(self, selected_task: str) -> tuple[str, str, str, str]:
        if selected_task == "oxygen_mask_delivery":
            return (
                self.oxygen_robot_port or self.robot_port,
                self.oxygen_robot_id or self.robot_id,
                self.oxygen_robot_cameras or self.robot_cameras,
                self.oxygen_act_policy_path,
            )

        if selected_task == "water_delivery":
            return (
                self.radio_robot_port or self.robot_port,
                self.radio_robot_id or self.robot_id,
                self.radio_robot_cameras or self.robot_cameras,
                self.radio_act_policy_path,
            )

        if selected_task == "radio_delivery":
            return (
                self.radio_robot_port or self.robot_port,
                self.radio_robot_id or self.robot_id,
                self.radio_robot_cameras or self.robot_cameras,
                self.radio_act_policy_path,
            )

        return (self.robot_port, self.robot_id, self.robot_cameras, "")

    def parse_task_request(self, raw: str) -> Optional[VlaTask]:
        try:
            data: Dict[str, Any] = json.loads(raw)
        except json.JSONDecodeError:
            self.get_logger().error(f"Invalid task_request JSON: {raw}")
            self.publish_status(
                task_id="unknown",
                selected_task="unknown",
                status="rejected",
                reason="invalid_json",
            )
            return None

        selected_task = str(data.get("selected_task", "")).strip()
        task_id = str(data.get("task_id", "")).strip()

        if not task_id:
            task_id = f"{selected_task}_{int(time.time() * 1000)}"

        if selected_task not in SUPPORTED_TASKS:
            self.publish_status(
                task_id=task_id,
                selected_task=selected_task,
                status="rejected",
                reason="unsupported_task",
            )
            return None

        robot_port, robot_id, robot_cameras, policy_path = self.route_for_task(selected_task)

        if not robot_port:
            self.publish_status(
                task_id=task_id,
                selected_task=selected_task,
                status="rejected",
                reason="empty_robot_port",
            )
            return None

        if not robot_id:
            self.publish_status(
                task_id=task_id,
                selected_task=selected_task,
                status="rejected",
                reason="empty_robot_id",
                robot_port=robot_port,
            )
            return None

        if not policy_path:
            self.publish_status(
                task_id=task_id,
                selected_task=selected_task,
                status="rejected",
                reason="empty_policy_path",
                robot_port=robot_port,
                robot_id=robot_id,
                robot_cameras=robot_cameras,
            )
            return None

        instruction = str(data.get("instruction", "")).strip()
        if not instruction:
            instruction = self.default_instruction_for_task(selected_task)
        elif self.use_selected_task_as_task_arg:
            # The user's existing LeRobot pipeline uses TASK_NAME as --task.
            instruction = selected_task

        task_duration_sec = safe_float(
            data.get("task_duration_sec", self.default_task_duration_sec),
            default=self.default_task_duration_sec,
            min_value=1.0,
        )
        timeout_sec = safe_float(
            data.get("timeout_sec", self.default_timeout_sec),
            default=self.default_timeout_sec,
            min_value=task_duration_sec + 1.0,
        )

        return VlaTask(
            task_id=task_id,
            selected_task=selected_task,
            instruction=instruction,
            robot_port=robot_port,
            robot_id=robot_id,
            robot_cameras=robot_cameras,
            policy_path=policy_path,
            task_duration_sec=task_duration_sec,
            timeout_sec=timeout_sec,
        )

    def build_robot_client_command(self, task: VlaTask) -> list[str]:
        cmd = [
            self.python_executable,
            "-m",
            self.rtc_client_module,
            f"--robot.type={self.robot_type}",
            f"--robot.port={task.robot_port}",
            f"--robot.id={task.robot_id}",
        ]

        if task.robot_cameras:
            cmd.append(f"--robot.cameras={task.robot_cameras}")

        cmd.extend(
            [
                f"--task={task.instruction}",
                f"--server_address={self.server_address}",
                f"--policy_type={self.policy_type}",
                f"--pretrained_name_or_path={task.policy_path}",
                f"--policy_device={self.policy_device}",
                f"--actions_per_chunk={self.actions_per_chunk}",
                f"--chunk_size_threshold={self.chunk_size_threshold}",
                f"--aggregate_fn_name={self.aggregate_fn_name}",
                f"--debug_visualize_queue_size={bool_to_cli(self.debug_visualize_queue_size)}",
            ]
        )

        cmd.extend(split_extra_args(self.extra_robot_client_args))
        return cmd

    def task_request_callback(self, msg: String) -> None:
        task = self.parse_task_request(msg.data)
        if task is None:
            return

        with self.lock:
            if self.active_task_id is not None and not self.allow_concurrent_tasks:
                self.publish_status(
                    task_id=task.task_id,
                    selected_task=task.selected_task,
                    status="rejected",
                    reason=f"executor_busy_active_task={self.active_task_id}",
                    robot_port=task.robot_port,
                    robot_id=task.robot_id,
                    robot_cameras=task.robot_cameras,
                    policy_path=task.policy_path,
                )
                return
            self.active_task_id = task.task_id

        self.publish_status(
            task_id=task.task_id,
            selected_task=task.selected_task,
            status="accepted",
            reason="task_accepted",
            robot_port=task.robot_port,
            robot_id=task.robot_id,
            robot_cameras=task.robot_cameras,
            policy_path=task.policy_path,
        )

        worker = threading.Thread(target=self.run_task, args=(task,), daemon=True)
        worker.start()

    def log_process_output(self, proc: subprocess.Popen, task_id: str) -> None:
        if proc.stdout is None:
            return

        try:
            for line in proc.stdout:
                text = line.rstrip()
                if text:
                    self.get_logger().info(f"[VLA RTC OUT][{task_id}] {text}")
        except Exception as exc:
            self.get_logger().warn(f"stdout reader stopped: {exc}")

    def terminate_process(self, proc: subprocess.Popen) -> None:
        if proc.poll() is not None:
            return

        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except Exception:
            try:
                proc.terminate()
            except Exception:
                pass

        deadline = time.time() + self.shutdown_grace_sec
        while time.time() < deadline:
            if proc.poll() is not None:
                return
            time.sleep(0.1)

        if proc.poll() is None:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass

    def run_task(self, task: VlaTask) -> None:
        start_time = time.time()
        proc: Optional[subprocess.Popen] = None

        try:
            cmd = self.build_robot_client_command(task)
            command_text = " ".join(shlex.quote(part) for part in cmd)
            self.get_logger().info(f"[VLA ACT] command: {command_text}")

            self.publish_status(
                task_id=task.task_id,
                selected_task=task.selected_task,
                status="running",
                reason="robot_client_started",
                robot_port=task.robot_port,
                robot_id=task.robot_id,
                robot_cameras=task.robot_cameras,
                policy_path=task.policy_path,
            )

            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                preexec_fn=os.setsid,
            )

            with self.lock:
                self.active_process = proc

            stdout_thread = threading.Thread(
                target=self.log_process_output,
                args=(proc, task.task_id),
                daemon=True,
            )
            stdout_thread.start()

            task_deadline = start_time + task.task_duration_sec
            timeout_deadline = start_time + task.timeout_sec

            while True:
                return_code = proc.poll()

                if return_code is not None:
                    elapsed = round(time.time() - start_time, 3)
                    if return_code == 0:
                        self.publish_status(
                            task_id=task.task_id,
                            selected_task=task.selected_task,
                            status="succeeded",
                            reason="robot_client_returned_zero",
                            robot_port=task.robot_port,
                            robot_id=task.robot_id,
                            robot_cameras=task.robot_cameras,
                            policy_path=task.policy_path,
                            return_code=return_code,
                            elapsed_sec=elapsed,
                        )
                    else:
                        self.publish_status(
                            task_id=task.task_id,
                            selected_task=task.selected_task,
                            status="failed",
                            reason="robot_client_nonzero_return",
                            robot_port=task.robot_port,
                            robot_id=task.robot_id,
                            robot_cameras=task.robot_cameras,
                            policy_path=task.policy_path,
                            return_code=return_code,
                            elapsed_sec=elapsed,
                        )
                    return

                now = time.time()

                if self.completion_mode == "timed" and now >= task_deadline:
                    self.get_logger().info(
                        f"[VLA ACT] timed duration completed. "
                        f"task_id={task.task_id}, duration={task.task_duration_sec:.3f}s"
                    )
                    self.terminate_process(proc)
                    elapsed = round(time.time() - start_time, 3)
                    self.publish_status(
                        task_id=task.task_id,
                        selected_task=task.selected_task,
                        status="succeeded",
                        reason="timed_execution_completed",
                        robot_port=task.robot_port,
                        robot_id=task.robot_id,
                        robot_cameras=task.robot_cameras,
                        policy_path=task.policy_path,
                        return_code=proc.poll(),
                        elapsed_sec=elapsed,
                    )
                    return

                if now >= timeout_deadline:
                    self.get_logger().error(
                        f"[VLA ACT] task timeout. "
                        f"task_id={task.task_id}, timeout={task.timeout_sec:.3f}s"
                    )
                    self.terminate_process(proc)
                    elapsed = round(time.time() - start_time, 3)
                    self.publish_status(
                        task_id=task.task_id,
                        selected_task=task.selected_task,
                        status="timeout",
                        reason="robot_client_timeout",
                        robot_port=task.robot_port,
                        robot_id=task.robot_id,
                        robot_cameras=task.robot_cameras,
                        policy_path=task.policy_path,
                        return_code=proc.poll(),
                        elapsed_sec=elapsed,
                    )
                    return

                time.sleep(0.1)

        except Exception as exc:
            elapsed = round(time.time() - start_time, 3)
            self.get_logger().error(f"[VLA ACT] task error: {exc}")

            if proc is not None and proc.poll() is None:
                self.terminate_process(proc)

            self.publish_status(
                task_id=task.task_id,
                selected_task=task.selected_task,
                status="failed",
                reason=f"exception: {exc}",
                robot_port=task.robot_port,
                robot_id=task.robot_id,
                robot_cameras=task.robot_cameras,
                policy_path=task.policy_path,
                return_code=None,
                elapsed_sec=elapsed,
            )

        finally:
            with self.lock:
                if self.active_task_id == task.task_id:
                    self.active_task_id = None
                    self.active_process = None

            self.publish_status(
                task_id=task.task_id,
                selected_task=task.selected_task,
                status="idle",
                reason="executor_returned_to_idle",
                robot_port=task.robot_port,
                robot_id=task.robot_id,
                robot_cameras=task.robot_cameras,
                policy_path=task.policy_path,
            )

    def destroy_node(self) -> None:
        self.get_logger().info("Stopping VLA ACT task executor.")

        with self.lock:
            proc = self.active_process

        if proc is not None and proc.poll() is None:
            self.get_logger().warn("Terminating active robot_client process.")
            self.terminate_process(proc)

        self.publish_status(
            task_id="none",
            selected_task="none",
            status="idle",
            reason="executor_shutdown",
        )

        super().destroy_node()


def main() -> None:
    rclpy.init()
    node: Optional[VlaActTaskExecutorNode] = None

    try:
        node = VlaActTaskExecutorNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
