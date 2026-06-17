#!/usr/bin/env python3

import math
import time
from collections import deque

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from std_msgs.msg import Bool, Float32, String


def wrap_deg(angle: float) -> float:
    return (angle + 180.0) % 360.0 - 180.0


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def circular_mean_deg(values: list[float]) -> float | None:
    if not values:
        return None

    sx = 0.0
    sy = 0.0
    for deg in values:
        rad = math.radians(deg)
        sx += math.cos(rad)
        sy += math.sin(rad)

    if abs(sx) < 1e-9 and abs(sy) < 1e-9:
        return None

    return math.degrees(math.atan2(sy, sx)) % 360.0


class VoiceFollowCmdNode(Node):
    """
    Continuous voice-direction follower.

    The node samples ReSpeaker VAD/DOA while voice is active, latches the
    averaged direction, and publishes smooth /cmd_vel_raw commands at a fixed
    rate. It intentionally does not create lateral commands.
    """

    def __init__(self) -> None:
        super().__init__("voice_follow_cmd_node")

        self.declare_parameter("vad_topic", "/zeri/audio/vad")
        self.declare_parameter("doa_topic", "/zeri/audio/doa_deg")
        self.declare_parameter("cmd_topic", "/cmd_vel_raw")
        self.declare_parameter("state_topic", "/zeri/audio/follow_state")

        self.declare_parameter("front_angle_deg", 245.0)
        self.declare_parameter("invert_turn", False)

        self.declare_parameter("sample_window_sec", 0.8)
        self.declare_parameter("min_voice_samples", 3)
        self.declare_parameter("target_hold_sec", 4.0)

        self.declare_parameter("turn_deadband_deg", 10.0)
        self.declare_parameter("turn_in_place_deg", 55.0)
        self.declare_parameter("allow_arc_motion", False)

        self.declare_parameter("forward_speed", 0.16)
        self.declare_parameter("min_forward_speed", 0.06)
        self.declare_parameter("turn_kp", 0.012)
        self.declare_parameter("min_turn_speed", 0.08)
        self.declare_parameter("max_turn_speed", 0.38)

        self.declare_parameter("cmd_hz", 20.0)
        self.declare_parameter("smooth_alpha", 0.35)
        self.declare_parameter("log_state", True)

        self.vad_topic = str(self.get_parameter("vad_topic").value)
        self.doa_topic = str(self.get_parameter("doa_topic").value)
        self.cmd_topic = str(self.get_parameter("cmd_topic").value)
        self.state_topic = str(self.get_parameter("state_topic").value)

        self.front_angle_deg = float(self.get_parameter("front_angle_deg").value)
        self.invert_turn = bool(self.get_parameter("invert_turn").value)

        self.sample_window_sec = float(self.get_parameter("sample_window_sec").value)
        self.min_voice_samples = int(self.get_parameter("min_voice_samples").value)
        self.target_hold_sec = float(self.get_parameter("target_hold_sec").value)

        self.turn_deadband_deg = float(self.get_parameter("turn_deadband_deg").value)
        self.turn_in_place_deg = float(self.get_parameter("turn_in_place_deg").value)
        self.allow_arc_motion = bool(self.get_parameter("allow_arc_motion").value)

        self.forward_speed = float(self.get_parameter("forward_speed").value)
        self.min_forward_speed = float(self.get_parameter("min_forward_speed").value)
        self.turn_kp = float(self.get_parameter("turn_kp").value)
        self.min_turn_speed = float(self.get_parameter("min_turn_speed").value)
        self.max_turn_speed = float(self.get_parameter("max_turn_speed").value)

        self.cmd_hz = float(self.get_parameter("cmd_hz").value)
        self.smooth_alpha = clamp(float(self.get_parameter("smooth_alpha").value), 0.0, 1.0)
        self.log_state = bool(self.get_parameter("log_state").value)

        self.vad = False
        self.last_doa_deg: float | None = None
        self.samples: deque[tuple[float, float]] = deque(maxlen=200)

        self.target_doa_deg: float | None = None
        self.target_err_deg = 0.0
        self.target_update_time = 0.0

        self.active_cmd = Twist()

        self.last_state_log_time = 0.0
        self.last_state_mode: str | None = None

        self.cmd_pub = self.create_publisher(Twist, self.cmd_topic, 10)
        self.state_pub = self.create_publisher(String, self.state_topic, 10)

        self.create_subscription(Bool, self.vad_topic, self.on_vad, 10)
        self.create_subscription(Float32, self.doa_topic, self.on_doa, 10)

        self.timer = self.create_timer(1.0 / max(self.cmd_hz, 1.0), self.on_timer)

        self.get_logger().info(
            "continuous voice follower started: "
            f"front={self.front_angle_deg}, invert_turn={self.invert_turn}, "
            f"hold={self.target_hold_sec}, cmd_hz={self.cmd_hz}"
        )

    def on_vad(self, msg: Bool) -> None:
        self.vad = bool(msg.data)

    def on_doa(self, msg: Float32) -> None:
        doa = float(msg.data) % 360.0
        self.last_doa_deg = doa

        if self.vad:
            self.samples.append((time.time(), doa))

    def trim_samples(self, now: float) -> None:
        while self.samples and now - self.samples[0][0] > self.sample_window_sec:
            self.samples.popleft()

    def update_target_from_voice(self, now: float) -> None:
        self.trim_samples(now)
        if len(self.samples) < self.min_voice_samples:
            return

        values = [doa for _, doa in self.samples]
        doa = circular_mean_deg(values)
        if doa is None:
            return

        err = wrap_deg(doa - self.front_angle_deg)
        if self.invert_turn:
            err = -err

        self.target_doa_deg = doa
        self.target_err_deg = err
        self.target_update_time = now

    def publish_state(self, text: str) -> None:
        msg = String()
        msg.data = text
        self.state_pub.publish(msg)

        if not self.log_state:
            return

        now = time.time()
        mode = text.split(" ", 1)[0]
        if mode != self.last_state_mode or now - self.last_state_log_time > 1.0:
            self.get_logger().info(text)
            self.last_state_mode = mode
            self.last_state_log_time = now

    def stop(self, reason: str) -> None:
        self.active_cmd = Twist()
        self.cmd_pub.publish(self.active_cmd)
        self.publish_state(reason)

    def desired_cmd_from_error(self, err_deg: float) -> tuple[str, Twist]:
        cmd = Twist()
        abs_err = abs(err_deg)

        if abs_err > self.turn_deadband_deg:
            turn = abs(err_deg) * self.turn_kp
            turn = clamp(turn, self.min_turn_speed, self.max_turn_speed)
            cmd.angular.z = math.copysign(turn, err_deg)

        if abs_err <= self.turn_deadband_deg:
            cmd.linear.x = self.forward_speed
            mode = "FORWARD"
        elif self.allow_arc_motion and abs_err < self.turn_in_place_deg:
            span = max(self.turn_in_place_deg - self.turn_deadband_deg, 1e-6)
            turn_ratio = clamp((abs_err - self.turn_deadband_deg) / span, 0.0, 1.0)
            cmd.linear.x = self.forward_speed - (
                self.forward_speed - self.min_forward_speed
            ) * turn_ratio
            mode = "ARC"
        else:
            cmd.linear.x = 0.0
            mode = "TURN"

        cmd.linear.y = 0.0
        return mode, cmd

    def smooth_cmd(self, desired: Twist) -> Twist:
        alpha = self.smooth_alpha

        out = Twist()
        out.linear.x = alpha * desired.linear.x + (1.0 - alpha) * self.active_cmd.linear.x
        out.linear.y = 0.0
        out.angular.z = alpha * desired.angular.z + (1.0 - alpha) * self.active_cmd.angular.z

        self.active_cmd = out
        return out

    def on_timer(self) -> None:
        now = time.time()

        if self.vad:
            self.update_target_from_voice(now)
        else:
            self.trim_samples(now)

        has_target = (
            self.target_doa_deg is not None
            and now - self.target_update_time <= self.target_hold_sec
        )

        if not has_target:
            self.samples.clear()
            self.target_doa_deg = None
            self.target_err_deg = 0.0
            self.stop(
                "WAIT_VOICE "
                f"vad={int(self.vad)} "
                f"samples={len(self.samples)}/{self.min_voice_samples} "
                f"last_doa={-1.0 if self.last_doa_deg is None else self.last_doa_deg:.1f}"
            )
            return

        mode, desired = self.desired_cmd_from_error(self.target_err_deg)
        cmd = self.smooth_cmd(desired)
        self.cmd_pub.publish(cmd)

        age = now - self.target_update_time
        self.publish_state(
            f"{mode} vad={int(self.vad)} "
            f"target_doa={self.target_doa_deg:.1f} "
            f"front={self.front_angle_deg:.1f} "
            f"err={self.target_err_deg:.1f} "
            f"vx={cmd.linear.x:.3f} wz={cmd.angular.z:.3f} "
            f"age={age:.2f}/{self.target_hold_sec:.2f}"
        )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = VoiceFollowCmdNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.cmd_pub.publish(Twist())
        except Exception:
            pass
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
