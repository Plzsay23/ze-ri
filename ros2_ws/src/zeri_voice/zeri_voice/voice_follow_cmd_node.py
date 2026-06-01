#!/usr/bin/env python3

import math
import time
from collections import deque

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Float32, String
from geometry_msgs.msg import Twist


def wrap_deg(angle: float) -> float:
    return (angle + 180.0) % 360.0 - 180.0


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def circular_mean_deg(values):
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
    Listen-Move-Listen voice follower.

    핵심:
      - 모터 정지 상태에서만 DOA 수집
      - 움직이는 동안 DOA 무시
      - 모터 소리로 인한 DOA 오염 방지
      - /cmd_vel_raw만 발행
      - lateral q/e 없음
    """

    def __init__(self):
        super().__init__("voice_follow_cmd_node")

        self.declare_parameter("vad_topic", "/zeri/audio/vad")
        self.declare_parameter("doa_topic", "/zeri/audio/doa_deg")
        self.declare_parameter("cmd_topic", "/cmd_vel_raw")
        self.declare_parameter("state_topic", "/zeri/audio/follow_state")

        self.declare_parameter("front_angle_deg", 245.0)
        self.declare_parameter("invert_turn", False)

        # LISTEN 단계
        self.declare_parameter("listen_sec", 0.8)
        self.declare_parameter("min_voice_samples", 5)
        self.declare_parameter("max_sample_age_sec", 1.2)

        # MOVE 단계
        self.declare_parameter("move_burst_sec", 0.45)
        self.declare_parameter("post_move_stop_sec", 0.25)

        # 판단 각도
        self.declare_parameter("align_enter_deg", 50.0)

        # 명령
        self.declare_parameter("forward_speed", 0.16)
        self.declare_parameter("min_turn_speed", 0.34)
        self.declare_parameter("max_turn_speed", 0.42)
        self.declare_parameter("turn_kp", 0.012)

        self.declare_parameter("cmd_hz", 10.0)
        self.declare_parameter("log_state", True)

        self.vad_topic = str(self.get_parameter("vad_topic").value)
        self.doa_topic = str(self.get_parameter("doa_topic").value)
        self.cmd_topic = str(self.get_parameter("cmd_topic").value)
        self.state_topic = str(self.get_parameter("state_topic").value)

        self.front_angle_deg = float(self.get_parameter("front_angle_deg").value)
        self.invert_turn = bool(self.get_parameter("invert_turn").value)

        self.listen_sec = float(self.get_parameter("listen_sec").value)
        self.min_voice_samples = int(self.get_parameter("min_voice_samples").value)
        self.max_sample_age_sec = float(self.get_parameter("max_sample_age_sec").value)

        self.move_burst_sec = float(self.get_parameter("move_burst_sec").value)
        self.post_move_stop_sec = float(self.get_parameter("post_move_stop_sec").value)

        self.align_enter_deg = float(self.get_parameter("align_enter_deg").value)

        self.forward_speed = float(self.get_parameter("forward_speed").value)
        self.min_turn_speed = float(self.get_parameter("min_turn_speed").value)
        self.max_turn_speed = float(self.get_parameter("max_turn_speed").value)
        self.turn_kp = float(self.get_parameter("turn_kp").value)

        self.cmd_hz = float(self.get_parameter("cmd_hz").value)
        self.log_state = bool(self.get_parameter("log_state").value)

        self.vad = False
        self.last_doa_deg = None

        self.mode = "LISTEN"
        self.mode_start_time = time.time()

        self.samples = deque(maxlen=100)

        self.move_cmd = Twist()
        self.target_doa_deg = None
        self.target_err_deg = 0.0

        self.last_state_log_time = 0.0
        self.last_state_mode = None

        self.cmd_pub = self.create_publisher(Twist, self.cmd_topic, 10)
        self.state_pub = self.create_publisher(String, self.state_topic, 10)

        self.create_subscription(Bool, self.vad_topic, self.on_vad, 10)
        self.create_subscription(Float32, self.doa_topic, self.on_doa, 10)

        period = 1.0 / max(self.cmd_hz, 1.0)
        self.timer = self.create_timer(period, self.on_timer)

        self.get_logger().info(
            "listen-move-listen voice follower started: "
            f"front={self.front_angle_deg}, invert_turn={self.invert_turn}, "
            f"listen_sec={self.listen_sec}, move_burst_sec={self.move_burst_sec}"
        )

    def on_vad(self, msg: Bool):
        self.vad = bool(msg.data)

    def on_doa(self, msg: Float32):
        self.last_doa_deg = float(msg.data) % 360.0

        # 핵심: 움직이는 동안 DOA는 저장하지 않음
        if self.mode == "LISTEN" and self.vad:
            self.samples.append((time.time(), self.last_doa_deg))

    def publish_stop(self):
        self.cmd_pub.publish(Twist())

    def publish_state(self, text: str):
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

    def set_mode(self, mode: str):
        self.mode = mode
        self.mode_start_time = time.time()

    def fresh_samples(self):
        now = time.time()
        return [
            doa for ts, doa in self.samples
            if now - ts <= self.max_sample_age_sec
        ]

    def make_move_cmd_from_error(self, err_deg: float) -> Twist:
        cmd = Twist()

        abs_err = abs(err_deg)

        if abs_err <= self.align_enter_deg:
            cmd.linear.x = self.forward_speed
            cmd.angular.z = 0.0
            return cmd

        sign = 1.0 if err_deg > 0.0 else -1.0

        speed = abs(self.turn_kp * err_deg)
        speed = clamp(speed, self.min_turn_speed, self.max_turn_speed)

        cmd.linear.x = 0.0
        cmd.angular.z = sign * speed
        return cmd

    def decide_move(self):
        samples = self.fresh_samples()

        if len(samples) < self.min_voice_samples:
            self.move_cmd = Twist()
            self.target_doa_deg = None
            self.target_err_deg = 0.0
            return False

        doa = circular_mean_deg(samples)

        if doa is None:
            self.move_cmd = Twist()
            self.target_doa_deg = None
            self.target_err_deg = 0.0
            return False

        err = wrap_deg(doa - self.front_angle_deg)

        if self.invert_turn:
            err = -err

        self.target_doa_deg = doa
        self.target_err_deg = err
        self.move_cmd = self.make_move_cmd_from_error(err)
        return True

    def on_timer(self):
        now = time.time()
        elapsed = now - self.mode_start_time

        if self.mode == "LISTEN":
            self.publish_stop()

            samples = self.fresh_samples()

            self.publish_state(
                f"LISTEN vad={int(self.vad)} "
                f"last_doa={-1.0 if self.last_doa_deg is None else self.last_doa_deg:.1f} "
                f"samples={len(samples)}/{self.min_voice_samples} "
                f"elapsed={elapsed:.2f}/{self.listen_sec:.2f}"
            )

            if elapsed >= self.listen_sec:
                ok = self.decide_move()

                if ok:
                    self.set_mode("MOVE")
                else:
                    self.samples.clear()
                    self.set_mode("LISTEN")

            return

        if self.mode == "MOVE":
            # 핵심: MOVE 중에는 새 DOA를 믿지 않고 기존 결정만 사용
            self.cmd_pub.publish(self.move_cmd)

            self.publish_state(
                f"MOVE target_doa={-1.0 if self.target_doa_deg is None else self.target_doa_deg:.1f} "
                f"front={self.front_angle_deg:.1f} "
                f"err={self.target_err_deg:.1f} "
                f"vx={self.move_cmd.linear.x:.3f} "
                f"wz={self.move_cmd.angular.z:.3f} "
                f"elapsed={elapsed:.2f}/{self.move_burst_sec:.2f}"
            )

            if elapsed >= self.move_burst_sec:
                self.publish_stop()
                self.samples.clear()
                self.set_mode("PAUSE")

            return

        if self.mode == "PAUSE":
            self.publish_stop()

            self.publish_state(
                f"PAUSE elapsed={elapsed:.2f}/{self.post_move_stop_sec:.2f}"
            )

            if elapsed >= self.post_move_stop_sec:
                self.samples.clear()
                self.set_mode("LISTEN")

            return

        self.publish_stop()
        self.publish_state(f"UNKNOWN_MODE {self.mode}")
        self.set_mode("LISTEN")


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
