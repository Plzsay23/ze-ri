#!/usr/bin/env python3
import math
import time
from typing import Optional

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Float32, String


def wrap_deg(angle: float) -> float:
    """Return angle in [-180, 180)."""
    return (angle + 180.0) % 360.0 - 180.0


class VoiceDoaTurnNode(Node):
    """
    Turn the mobile base toward the detected human voice direction.

    Inputs:
      /zeri/audio/vad         std_msgs/Bool
      /zeri/audio/doa_deg     std_msgs/Float32

    Outputs:
      /auto_cmd               std_msgs/String: A / D / W / X
      /mode_select            std_msgs/String: AUTO / STOP

    Existing pipeline:
      /auto_cmd + /mode_select
        -> voice_mode_manager
        -> /mode_cmd
        -> lidar_guard_node
        -> /safe_cmd
        -> cmd_serial_bridge
        -> Arduino
    """

    def __init__(self):
        super().__init__("voice_doa_turn_node")

        self.declare_parameter("speech_vad_topic", "/zeri/audio/vad")
        self.declare_parameter("doa_topic", "/zeri/audio/doa_deg")

        self.declare_parameter("output_cmd_topic", "/auto_cmd")
        self.declare_parameter("mode_select_topic", "/mode_select")

        # ReSpeaker DOA 기준에서 로봇 정면에 해당하는 각도.
        # 정면에서 말했을 때 doa_deg가 0 근처면 0.0 유지.
        self.declare_parameter("front_deg", 0.0)

        # 정면으로 인정할 오차 범위.
        self.declare_parameter("deadband_deg", 12.0)

        # speech_vad가 false가 된 뒤 이 시간까지는 잠깐 유지.
        self.declare_parameter("speech_hold_sec", 0.45)

        # DOA 값이 이 시간 이상 안 들어오면 정지.
        self.declare_parameter("doa_timeout_sec", 0.75)

        self.declare_parameter("publish_rate_hz", 8.0)

        # 현재 코드 기준:
        # A/D는 회전, Q/E는 strafe로 보는 게 맞음.
        self.declare_parameter("left_turn_cmd", "A")
        self.declare_parameter("right_turn_cmd", "D")
        self.declare_parameter("stop_cmd", "X")
        self.declare_parameter("forward_cmd", "W")
        self.declare_parameter("turn_start_deadband_deg", 18.0)
        self.declare_parameter("turn_stop_deadband_deg", 8.0)
        self.declare_parameter("forward_hold_sec", 4.0)
        self.declare_parameter("smoothing_alpha", 0.35)

        # DOA 각도 증가 방향이 로봇 기준 좌우와 반대면 true.
        self.declare_parameter("invert_direction", False)

        # true면 speech 시작 시 AUTO, 종료 시 STOP을 보냄.
        self.declare_parameter("manage_mode", True)

        self.speech_vad_topic = str(self.get_parameter("speech_vad_topic").value)
        self.doa_topic = str(self.get_parameter("doa_topic").value)
        self.output_cmd_topic = str(self.get_parameter("output_cmd_topic").value)
        self.mode_select_topic = str(self.get_parameter("mode_select_topic").value)

        self.front_deg = float(self.get_parameter("front_deg").value)
        self.deadband_deg = float(self.get_parameter("deadband_deg").value)
        self.speech_hold_sec = float(self.get_parameter("speech_hold_sec").value)
        self.doa_timeout_sec = float(self.get_parameter("doa_timeout_sec").value)
        self.publish_rate_hz = float(self.get_parameter("publish_rate_hz").value)

        self.left_turn_cmd = str(self.get_parameter("left_turn_cmd").value).strip().upper()
        self.right_turn_cmd = str(self.get_parameter("right_turn_cmd").value).strip().upper()
        self.stop_cmd = str(self.get_parameter("stop_cmd").value).strip().upper()
        self.forward_cmd = str(self.get_parameter("forward_cmd").value).strip().upper()
        self.turn_start_deadband_deg = float(self.get_parameter("turn_start_deadband_deg").value)
        self.turn_stop_deadband_deg = float(self.get_parameter("turn_stop_deadband_deg").value)
        self.forward_hold_sec = float(self.get_parameter("forward_hold_sec").value)
        self.smoothing_alpha = float(self.get_parameter("smoothing_alpha").value)

        self.invert_direction = bool(self.get_parameter("invert_direction").value)
        self.manage_mode = bool(self.get_parameter("manage_mode").value)

        if self.publish_rate_hz <= 0.0:
            self.publish_rate_hz = 8.0

        self.turn_start_deadband_deg = max(self.turn_start_deadband_deg, self.deadband_deg)
        self.turn_stop_deadband_deg = min(self.turn_stop_deadband_deg, self.deadband_deg)
        self.turn_stop_deadband_deg = max(1.0, self.turn_stop_deadband_deg)
        self.forward_hold_sec = max(0.0, self.forward_hold_sec)
        self.smoothing_alpha = min(1.0, max(0.0, self.smoothing_alpha))

        self.latest_speech_vad = False
        self.latest_doa: Optional[float] = None

        now = time.monotonic()
        self.last_speech_true_time = 0.0
        self.last_doa_time = 0.0
        self.last_cmd: Optional[str] = None
        self.auto_mode_active = False
        self.smoothed_error: Optional[float] = None
        self.turn_cmd: Optional[str] = None
        self.last_aligned_time = 0.0
        self.last_log_time = now

        self.cmd_pub = self.create_publisher(String, self.output_cmd_topic, 10)
        self.mode_pub = self.create_publisher(String, self.mode_select_topic, 10)

        self.debug_cmd_pub = self.create_publisher(String, "/zeri/audio/voice_turn_cmd", 10)
        self.debug_error_pub = self.create_publisher(Float32, "/zeri/audio/voice_turn_error_deg", 10)
        self.debug_active_pub = self.create_publisher(Bool, "/zeri/audio/voice_turn_active", 10)

        self.speech_sub = self.create_subscription(
            Bool,
            self.speech_vad_topic,
            self.speech_callback,
            10,
        )

        self.doa_sub = self.create_subscription(
            Float32,
            self.doa_topic,
            self.doa_callback,
            10,
        )

        self.timer = self.create_timer(1.0 / self.publish_rate_hz, self.timer_callback)

        self.get_logger().info("VoiceDoaTurnNode started")
        self.get_logger().info(f"speech_vad_topic={self.speech_vad_topic}")
        self.get_logger().info(f"doa_topic={self.doa_topic}")
        self.get_logger().info(f"output_cmd_topic={self.output_cmd_topic}")
        self.get_logger().info(f"mode_select_topic={self.mode_select_topic}")
        self.get_logger().info(
            f"front_deg={self.front_deg:.1f}, deadband_deg={self.deadband_deg:.1f}, "
            f"invert_direction={self.invert_direction}, manage_mode={self.manage_mode}"
        )
        self.get_logger().info(
            f"left_turn_cmd={self.left_turn_cmd}, "
            f"right_turn_cmd={self.right_turn_cmd}, forward_cmd={self.forward_cmd}, "
            f"stop_cmd={self.stop_cmd}, forward_hold_sec={self.forward_hold_sec:.1f}"
        )

    def speech_callback(self, msg: Bool):
        self.latest_speech_vad = bool(msg.data)
        if self.latest_speech_vad:
            self.last_speech_true_time = time.monotonic()

    def doa_callback(self, msg: Float32):
        self.latest_doa = float(msg.data) % 360.0
        self.last_doa_time = time.monotonic()

    def publish_string(self, pub, text: str):
        msg = String()
        msg.data = text
        pub.publish(msg)

    def publish_mode(self, mode: str):
        if not self.manage_mode:
            return
        self.publish_string(self.mode_pub, mode)

    def publish_cmd(self, cmd: str, force: bool = False):
        if force or cmd != self.last_cmd:
            self.publish_string(self.cmd_pub, cmd)
            self.publish_string(self.debug_cmd_pub, cmd)
            self.last_cmd = cmd
            self.get_logger().info(f"voice_turn_cmd={cmd}")

    def stop(self):
        self.publish_cmd(self.stop_cmd)
        self.turn_cmd = None
        self.smoothed_error = None
        self.last_aligned_time = 0.0
        if self.auto_mode_active:
            self.publish_mode("STOP")
            self.auto_mode_active = False

    def speech_is_active(self, now: float) -> bool:
        return (now - self.last_speech_true_time) <= self.speech_hold_sec

    def doa_is_fresh(self, now: float) -> bool:
        return self.latest_doa is not None and (now - self.last_doa_time) <= self.doa_timeout_sec

    def timer_callback(self):
        now = time.monotonic()

        tracking_active = self.speech_is_active(now) and self.doa_is_fresh(now)
        forward_hold_active = (now - self.last_aligned_time) <= self.forward_hold_sec
        command_active = tracking_active or forward_hold_active

        active_msg = Bool()
        active_msg.data = bool(command_active)
        self.debug_active_pub.publish(active_msg)

        if not command_active:
            self.stop()
            return

        if not self.auto_mode_active:
            self.publish_mode("AUTO")
            self.auto_mode_active = True

        cmd = self.forward_cmd
        error_for_log = None

        if tracking_active:
            assert self.latest_doa is not None

            raw_error = wrap_deg(self.latest_doa - self.front_deg)
            if self.smoothed_error is None:
                self.smoothed_error = raw_error
            else:
                delta = wrap_deg(raw_error - self.smoothed_error)
                self.smoothed_error = wrap_deg(self.smoothed_error + self.smoothing_alpha * delta)

            error = self.smoothed_error
            error_for_log = error

            error_msg = Float32()
            error_msg.data = float(error)
            self.debug_error_pub.publish(error_msg)

            if abs(error) <= self.turn_stop_deadband_deg:
                cmd = self.forward_cmd
                self.turn_cmd = None
                self.last_aligned_time = now
            elif self.turn_cmd is not None and abs(error) < self.turn_start_deadband_deg:
                self.last_aligned_time = 0.0
                cmd = self.turn_cmd
            else:
                if error > 0.0:
                    cmd = self.right_turn_cmd
                else:
                    cmd = self.left_turn_cmd

                if self.invert_direction:
                    if cmd == self.right_turn_cmd:
                        cmd = self.left_turn_cmd
                    elif cmd == self.left_turn_cmd:
                        cmd = self.right_turn_cmd

                self.last_aligned_time = 0.0
                self.turn_cmd = cmd
        else:
            cmd = self.forward_cmd

        self.publish_cmd(cmd)

        if now - self.last_log_time > 0.8:
            self.last_log_time = now
            if error_for_log is None:
                self.get_logger().info(
                    f"voice target hold -> cmd={cmd}, active={command_active}"
                )
            else:
                self.get_logger().info(
                    f"doa={self.latest_doa:.1f}, front={self.front_deg:.1f}, "
                    f"error={error_for_log:.1f}, cmd={cmd}, active={command_active}"
                )

    def destroy_node(self):
        try:
            self.publish_cmd(self.stop_cmd, force=True)
            if self.manage_mode:
                self.publish_mode("STOP")
        except Exception:
            pass
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = VoiceDoaTurnNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
