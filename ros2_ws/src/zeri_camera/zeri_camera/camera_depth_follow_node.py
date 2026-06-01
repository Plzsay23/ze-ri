#!/usr/bin/env python3

import time
import math
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import Image
from std_msgs.msg import Bool, String
from geometry_msgs.msg import Twist


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


class CameraDepthFollowNode(Node):
    """
    Depth-camera based person/object direction follower.

    역할:
      - DOA 사용 안 함
      - VAD는 enable trigger로만 사용
      - depth image에서 가장 가까운 foreground target의 x 위치를 추정
      - target이 왼쪽/오른쪽이면 회전
      - target이 중앙이면 전진
      - target이 너무 가까우면 정지
      - /cmd_vel_raw만 발행
      - q/e 횡이동 없음
    """

    def __init__(self):
        super().__init__("camera_depth_follow_node")

        self.declare_parameter("depth_topic", "/zeri/vlm/input_depth")
        self.declare_parameter("vad_topic", "/zeri/audio/vad")
        self.declare_parameter("cmd_topic", "/cmd_vel_raw")
        self.declare_parameter("state_topic", "/zeri/camera_follow/state")

        self.declare_parameter("use_vad_gate", True)
        self.declare_parameter("voice_hold_sec", 2.0)

        self.declare_parameter("cmd_hz", 10.0)
        self.declare_parameter("depth_timeout_sec", 1.0)

        # depth ROI. 바닥 영향을 줄이기 위해 하단 일부 제외.
        self.declare_parameter("roi_x_min", 0.05)
        self.declare_parameter("roi_x_max", 0.95)
        self.declare_parameter("roi_y_min", 0.20)
        self.declare_parameter("roi_y_max", 0.82)

        # 추적 대상 depth 범위
        self.declare_parameter("depth_min_m", 0.35)
        self.declare_parameter("depth_max_m", 2.80)

        # target 추출 조건
        self.declare_parameter("min_valid_pixels", 250)
        self.declare_parameter("target_window_px", 100)

        # 주행 조건
        self.declare_parameter("target_distance_m", 1.00)
        self.declare_parameter("distance_deadband_m", 0.18)
        self.declare_parameter("stop_distance_m", 0.55)

        # 화면 중앙 정렬
        self.declare_parameter("center_deadband_norm", 0.16)

        self.declare_parameter("forward_speed", 0.16)
        self.declare_parameter("turn_kp", 0.65)
        self.declare_parameter("min_turn_speed", 0.32)
        self.declare_parameter("max_turn_speed", 0.45)

        self.declare_parameter("invert_turn", False)
        self.declare_parameter("smooth_alpha", 0.45)

        self.depth_topic = str(self.get_parameter("depth_topic").value)
        self.vad_topic = str(self.get_parameter("vad_topic").value)
        self.cmd_topic = str(self.get_parameter("cmd_topic").value)
        self.state_topic = str(self.get_parameter("state_topic").value)

        self.use_vad_gate = bool(self.get_parameter("use_vad_gate").value)
        self.voice_hold_sec = float(self.get_parameter("voice_hold_sec").value)

        self.cmd_hz = float(self.get_parameter("cmd_hz").value)
        self.depth_timeout_sec = float(self.get_parameter("depth_timeout_sec").value)

        self.roi_x_min = float(self.get_parameter("roi_x_min").value)
        self.roi_x_max = float(self.get_parameter("roi_x_max").value)
        self.roi_y_min = float(self.get_parameter("roi_y_min").value)
        self.roi_y_max = float(self.get_parameter("roi_y_max").value)

        self.depth_min_m = float(self.get_parameter("depth_min_m").value)
        self.depth_max_m = float(self.get_parameter("depth_max_m").value)

        self.min_valid_pixels = int(self.get_parameter("min_valid_pixels").value)
        self.target_window_px = int(self.get_parameter("target_window_px").value)

        self.target_distance_m = float(self.get_parameter("target_distance_m").value)
        self.distance_deadband_m = float(self.get_parameter("distance_deadband_m").value)
        self.stop_distance_m = float(self.get_parameter("stop_distance_m").value)

        self.center_deadband_norm = float(self.get_parameter("center_deadband_norm").value)

        self.forward_speed = float(self.get_parameter("forward_speed").value)
        self.turn_kp = float(self.get_parameter("turn_kp").value)
        self.min_turn_speed = float(self.get_parameter("min_turn_speed").value)
        self.max_turn_speed = float(self.get_parameter("max_turn_speed").value)

        self.invert_turn = bool(self.get_parameter("invert_turn").value)
        self.smooth_alpha = float(self.get_parameter("smooth_alpha").value)

        self.last_depth = None
        self.last_depth_stamp = 0.0
        self.last_depth_encoding = ""

        self.vad = False
        self.last_voice_time = 0.0

        self.smooth_x = None
        self.smooth_z = None

        self.last_state_log_time = 0.0
        self.last_mode = None

        self.cmd_pub = self.create_publisher(Twist, self.cmd_topic, 10)
        self.state_pub = self.create_publisher(String, self.state_topic, 10)

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )

        self.create_subscription(Image, self.depth_topic, self.on_depth, sensor_qos)
        self.create_subscription(Bool, self.vad_topic, self.on_vad, 10)

        self.timer = self.create_timer(1.0 / max(self.cmd_hz, 1.0), self.on_timer)

        self.get_logger().info(
            "camera depth follow started: "
            f"depth={self.depth_topic}, vad={self.vad_topic}, cmd={self.cmd_topic}, "
            f"use_vad_gate={self.use_vad_gate}"
        )

    def on_vad(self, msg: Bool):
        self.vad = bool(msg.data)
        if self.vad:
            self.last_voice_time = time.time()

    def on_depth(self, msg: Image):
        try:
            depth = self.decode_depth(msg)
        except Exception as e:
            self.get_logger().warn(f"failed to decode depth: {repr(e)}")
            return

        self.last_depth = depth
        self.last_depth_stamp = time.time()
        self.last_depth_encoding = msg.encoding

    def decode_depth(self, msg: Image):
        h = int(msg.height)
        w = int(msg.width)
        step = int(msg.step)
        enc = msg.encoding

        if enc in ("16UC1", "mono16"):
            dtype = np.dtype(">u2" if msg.is_bigendian else "<u2")
            row_bytes = w * 2
            raw = np.frombuffer(msg.data, dtype=np.uint8).reshape(h, step)
            raw = raw[:, :row_bytes].copy()
            arr = raw.view(dtype).reshape(h, w)
            depth_m = arr.astype(np.float32) / 1000.0
            return depth_m

        if enc == "32FC1":
            dtype = np.dtype(">f4" if msg.is_bigendian else "<f4")
            row_bytes = w * 4
            raw = np.frombuffer(msg.data, dtype=np.uint8).reshape(h, step)
            raw = raw[:, :row_bytes].copy()
            arr = raw.view(dtype).reshape(h, w)
            return arr.astype(np.float32)

        raise RuntimeError(f"unsupported depth encoding: {enc}")

    def publish_stop(self, reason):
        self.cmd_pub.publish(Twist())
        self.publish_state(reason)

    def publish_state(self, text):
        msg = String()
        msg.data = text
        self.state_pub.publish(msg)

        now = time.time()
        mode = text.split(" ", 1)[0]

        if mode != self.last_mode or now - self.last_state_log_time > 1.0:
            self.get_logger().info(text)
            self.last_mode = mode
            self.last_state_log_time = now

    def find_target(self, depth):
        h, w = depth.shape

        x0 = int(clamp(self.roi_x_min, 0.0, 1.0) * w)
        x1 = int(clamp(self.roi_x_max, 0.0, 1.0) * w)
        y0 = int(clamp(self.roi_y_min, 0.0, 1.0) * h)
        y1 = int(clamp(self.roi_y_max, 0.0, 1.0) * h)

        if x1 <= x0 or y1 <= y0:
            return None

        roi = depth[y0:y1, x0:x1]

        valid = np.isfinite(roi)
        valid &= roi >= self.depth_min_m
        valid &= roi <= self.depth_max_m

        valid_count = int(np.count_nonzero(valid))
        if valid_count < self.min_valid_pixels:
            return None

        # 가까운 물체를 강하게 가중.
        # 사람/장애물이 배경보다 가까우면 그쪽 column score가 커짐.
        weights = np.zeros_like(roi, dtype=np.float32)
        weights[valid] = 1.0 / np.maximum(roi[valid] ** 2, 1e-6)

        col_score = weights.sum(axis=0)

        if col_score.size < 3 or float(col_score.max()) <= 0.0:
            return None

        # column score smoothing
        kernel_size = max(9, min(41, col_score.size // 12))
        if kernel_size % 2 == 0:
            kernel_size += 1

        kernel = np.ones(kernel_size, dtype=np.float32) / float(kernel_size)
        smooth_score = np.convolve(col_score, kernel, mode="same")

        peak = int(np.argmax(smooth_score))

        win = int(self.target_window_px)
        lx = max(0, peak - win)
        rx = min(roi.shape[1], peak + win + 1)

        local_weights = weights[:, lx:rx]
        local_valid = valid[:, lx:rx]

        if int(np.count_nonzero(local_valid)) < self.min_valid_pixels:
            return None

        xs = np.arange(lx, rx, dtype=np.float32)
        col_w = local_weights.sum(axis=0)
        total_w = float(col_w.sum())

        if total_w <= 1e-6:
            return None

        target_x_roi = float((xs * col_w).sum() / total_w)
        target_x = x0 + target_x_roi

        local_depth_values = roi[:, lx:rx][local_valid]
        if local_depth_values.size == 0:
            return None

        target_z = float(np.median(local_depth_values))

        confidence = min(1.0, valid_count / float(self.min_valid_pixels * 8))

        return {
            "x": target_x,
            "z": target_z,
            "valid_count": valid_count,
            "confidence": confidence,
            "image_w": w,
            "image_h": h,
        }

    def smooth_target(self, x, z):
        a = clamp(self.smooth_alpha, 0.0, 1.0)

        if self.smooth_x is None:
            self.smooth_x = x
            self.smooth_z = z
        else:
            self.smooth_x = a * x + (1.0 - a) * self.smooth_x
            self.smooth_z = a * z + (1.0 - a) * self.smooth_z

        return self.smooth_x, self.smooth_z

    def make_turn_cmd(self, err_norm):
        cmd = Twist()

        # 화면 기준:
        # target이 오른쪽이면 err_norm > 0
        # 로봇은 오른쪽으로 돌아야 하므로 angular.z는 음수
        signed_err = err_norm
        if self.invert_turn:
            signed_err = -signed_err

        wz = -self.turn_kp * signed_err

        if abs(wz) > 1e-6:
            wz_abs = clamp(abs(wz), self.min_turn_speed, self.max_turn_speed)
            wz = math.copysign(wz_abs, wz)

        cmd.angular.z = wz
        return cmd

    def on_timer(self):
        now = time.time()

        if self.use_vad_gate:
            voice_active = (now - self.last_voice_time) <= self.voice_hold_sec
            if not voice_active:
                self.smooth_x = None
                self.smooth_z = None
                self.publish_stop("WAIT_VOICE")
                return

        if self.last_depth is None:
            self.publish_stop("NO_DEPTH")
            return

        if now - self.last_depth_stamp > self.depth_timeout_sec:
            self.publish_stop("DEPTH_TIMEOUT")
            return

        target = self.find_target(self.last_depth)

        if target is None:
            self.smooth_x = None
            self.smooth_z = None
            self.publish_stop("NO_TARGET")
            return

        x, z = self.smooth_target(target["x"], target["z"])

        w = float(target["image_w"])
        cx = w * 0.5

        err_norm = (x - cx) / max(cx, 1.0)
        abs_err = abs(err_norm)

        cmd = Twist()

        if z <= self.stop_distance_m:
            self.cmd_pub.publish(Twist())
            self.publish_state(
                f"TOO_CLOSE x={x:.1f} z={z:.2f} err={err_norm:.2f} "
                f"valid={target['valid_count']} enc={self.last_depth_encoding}"
            )
            return

        if abs_err > self.center_deadband_norm:
            cmd = self.make_turn_cmd(err_norm)
            mode = "TURN_TO_CAMERA"
        else:
            if z > self.target_distance_m + self.distance_deadband_m:
                cmd.linear.x = self.forward_speed
                mode = "FORWARD_TO_CAMERA"
            else:
                mode = "HOLD_DISTANCE"
                cmd = Twist()

        # q/e 방지
        cmd.linear.y = 0.0

        self.cmd_pub.publish(cmd)

        self.publish_state(
            f"{mode} x={x:.1f} z={z:.2f} "
            f"err={err_norm:.2f} "
            f"vx={cmd.linear.x:.3f} wz={cmd.angular.z:.3f} "
            f"valid={target['valid_count']} conf={target['confidence']:.2f} "
            f"enc={self.last_depth_encoding}"
        )


def main(args=None):
    rclpy.init(args=args)
    node = CameraDepthFollowNode()

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
