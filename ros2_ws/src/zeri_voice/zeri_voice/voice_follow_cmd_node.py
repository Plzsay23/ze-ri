#!/usr/bin/env python3

import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Float32, String
from geometry_msgs.msg import Twist


def wrap_deg(angle: float) -> float:
    """
    Wrap angle to [-180, 180).
    """
    return (angle + 180.0) % 360.0 - 180.0


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


class VoiceFollowCmdNode(Node):
    """
    Converts VAD + DOA into /cmd_vel_raw.

    Pipeline:
      /zeri/audio/vad
      /zeri/audio/doa_deg
        -> voice_follow_cmd_node
        -> /cmd_vel_raw
        -> lidar_depth_guard_node
        -> /cmd_vel
        -> base_key_odom_serial_node
        -> Arduino

    This node must publish to /cmd_vel_raw, not /cmd_vel.
    """

    def __init__(self):
        super().__init__("voice_follow_cmd_node")

        self.declare_parameter("vad_topic", "/zeri/audio/vad")
        self.declare_parameter("doa_topic", "/zeri/audio/doa_deg")
        self.declare_parameter("cmd_topic", "/cmd_vel_raw")
        self.declare_parameter("state_topic", "/zeri/audio/follow_state")

        # ReSpeaker에서 로봇 정면이 몇 도로 나오는지.
        # 정면에서 말해보고 그 DOA 값을 넣는다.
        self.declare_parameter("front_angle_deg", 0.0)

        # 좌우 회전이 반대로 나오면 true.
        self.declare_parameter("invert_turn", False)

        # VAD가 순간적으로 끊겨도 마지막 음성 방향을 유지하는 시간.
        self.declare_parameter("voice_hold_sec", 0.8)

        # 이 각도 이하이면 정면이라고 보고 전진.
        self.declare_parameter("drive_deadband_deg", 20.0)

        # 전진 속도. 기존 Arduino key 방식에서는 실제 속도는 Arduino 코드가 결정.
        self.declare_parameter("forward_speed", 0.16)

        # 각도 오차에 따른 회전 속도.
        self.declare_parameter("turn_kp", 0.018)
        self.declare_parameter("max_turn_speed", 0.45)

        self.declare_parameter("cmd_hz", 10.0)
        self.declare_parameter("publish_stop_when_no_voice", True)
        self.declare_parameter("log_state", True)

        self.vad_topic = str(self.get_parameter("vad_topic").value)
        self.doa_topic = str(self.get_parameter("doa_topic").value)
        self.cmd_topic = str(self.get_parameter("cmd_topic").value)
        self.state_topic = str(self.get_parameter("state_topic").value)

        self.front_angle_deg = float(self.get_parameter("front_angle_deg").value)
        self.invert_turn = bool(self.get_parameter("invert_turn").value)
        self.voice_hold_sec = float(self.get_parameter("voice_hold_sec").value)
        self.drive_deadband_deg = float(self.get_parameter("drive_deadband_deg").value)
        self.forward_speed = float(self.get_parameter("forward_speed").value)
        self.turn_kp = float(self.get_parameter("turn_kp").value)
        self.max_turn_speed = float(self.get_parameter("max_turn_speed").value)
        self.cmd_hz = float(self.get_parameter("cmd_hz").value)
        self.publish_stop_when_no_voice = bool(
            self.get_parameter("publish_stop_when_no_voice").value
        )
        self.log_state = bool(self.get_parameter("log_state").value)

        self.last_vad = False
        self.last_doa_deg = None
        self.last_voice_time = 0.0
        self.last_state_log_time = 0.0
        self.last_mode = None

        self.cmd_pub = self.create_publisher(Twist, self.cmd_topic, 10)
        self.state_pub = self.create_publisher(String, self.state_topic, 10)

        self.create_subscription(Bool, self.vad_topic, self.on_vad, 10)
        self.create_subscription(Float32, self.doa_topic, self.on_doa, 10)

        period = 1.0 / max(self.cmd_hz, 1.0)
        self.timer = self.create_timer(period, self.on_timer)

        self.get_logger().info(
            "voice follow node started: "
            f"vad_topic={self.vad_topic}, doa_topic={self.doa_topic}, "
            f"cmd_topic={self.cmd_topic}, front_angle_deg={self.front_angle_deg}, "
            f"invert_turn={self.invert_turn}"
        )

    def on_vad(self, msg: Bool):
        self.last_vad = bool(msg.data)

        if self.last_vad:
            self.last_voice_time = time.time()

    def on_doa(self, msg: Float32):
        self.last_doa_deg = float(msg.data) % 360.0

        if self.last_vad:
            self.last_voice_time = time.time()

    def publish_state(self, text: str):
        msg = String()
        msg.data = text
        self.state_pub.publish(msg)

        if not self.log_state:
            return

        now = time.time()
        mode = text.split(" ", 1)[0]

        if mode != self.last_mode or now - self.last_state_log_time > 1.0:
            self.get_logger().info(text)
            self.last_state_log_time = now
            self.last_mode = mode

    def publish_stop(self, reason: str):
        if self.publish_stop_when_no_voice:
            self.cmd_pub.publish(Twist())

        self.publish_state(reason)

    def on_timer(self):
        now = time.time()

        if self.last_doa_deg is None:
            self.publish_stop("NO_DOA")
            return

        voice_active = (now - self.last_voice_time) <= self.voice_hold_sec

        if not voice_active:
            self.publish_stop("NO_VOICE")
            return

        err_deg = wrap_deg(self.last_doa_deg - self.front_angle_deg)

        # 좌우가 반대로 나오면 부호 반전.
        if self.invert_turn:
            err_deg = -err_deg

        abs_err = abs(err_deg)

        cmd = Twist()

        if abs_err <= self.drive_deadband_deg:
            cmd.linear.x = self.forward_speed
            cmd.linear.y = 0.0
            cmd.angular.z = 0.0
            mode = "FORWARD_TO_VOICE"
        else:
            cmd.linear.x = 0.0
            cmd.linear.y = 0.0
            cmd.angular.z = clamp(
                self.turn_kp * err_deg,
                -self.max_turn_speed,
                self.max_turn_speed,
            )
            mode = "TURN_TO_VOICE"

        self.cmd_pub.publish(cmd)

        self.publish_state(
            f"{mode} "
            f"vad=1 "
            f"doa={self.last_doa_deg:.1f} "
            f"front={self.front_angle_deg:.1f} "
            f"err={err_deg:.1f} "
            f"vx={cmd.linear.x:.3f} "
            f"vy={cmd.linear.y:.3f} "
            f"wz={cmd.angular.z:.3f}"
        )


def main(args=None):
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
