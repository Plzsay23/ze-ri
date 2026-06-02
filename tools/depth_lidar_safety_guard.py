#!/usr/bin/env python3

import json
import math
import re
import time

import numpy as np

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan, Image
from std_msgs.msg import String


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def sign_changed(a, b, eps):
    return abs(a) > eps and abs(b) > eps and (a * b) < 0.0


def copy_twist(src):
    t = Twist()
    t.linear.x = float(src.linear.x)
    t.linear.y = float(src.linear.y)
    t.linear.z = float(src.linear.z)
    t.angular.x = float(src.angular.x)
    t.angular.y = float(src.angular.y)
    t.angular.z = float(src.angular.z)
    return t


def zero_twist():
    return Twist()


def is_zero_cmd(t):
    return (
        abs(t.linear.x) < 1e-4
        and abs(t.linear.y) < 1e-4
        and abs(t.angular.z) < 1e-4
    )


def approach(current, target, max_delta):
    if target > current + max_delta:
        return current + max_delta
    if target < current - max_delta:
        return current - max_delta
    return target


class DepthLidarSafetyGuard(Node):
    def __init__(self):
        super().__init__("depth_lidar_safety_guard")

        self.declare_parameter("cmd_raw_topic", "/cmd_vel_raw")
        self.declare_parameter("cmd_out_topic", "/cmd_vel")
        self.declare_parameter("scan_topic", "/scan_front")
        self.declare_parameter("depth_topic", "/camera/camera/aligned_depth_to_color/image_raw")
        self.declare_parameter("person_state_topic", "/zeri/person_follow/state")
        self.declare_parameter("state_topic", "/zeri/safety_guard/state")

        self.declare_parameter("publish_hz", 20.0)
        self.declare_parameter("cmd_timeout_sec", 0.45)

        # 사람 추종 명령이 너무 빨리 좌우/전후 전환되는 것 방지
        self.declare_parameter("raw_hold_sec", 0.22)

        # 출력 속도 제한. 틱틱거림 방지용.
        self.declare_parameter("linear_accel_mps2", 0.28)
        self.declare_parameter("strafe_accel_mps2", 0.24)
        self.declare_parameter("angular_accel_rps2", 0.90)

        self.declare_parameter("max_linear_x", 0.18)
        self.declare_parameter("max_linear_y", 0.14)
        self.declare_parameter("max_angular_z", 0.45)

        # LiDAR/depth 공통 전방 안전 거리
        self.declare_parameter("front_stop_m", 0.55)
        self.declare_parameter("front_slow_m", 0.95)
        self.declare_parameter("front_emergency_m", 0.38)

        # 횡이동 회피
        self.declare_parameter("enable_strafe_avoidance", True)
        self.declare_parameter("strafe_speed", 0.10)
        self.declare_parameter("strafe_clear_m", 0.75)
        self.declare_parameter("side_stop_m", 0.45)

        # 횡이동 못 할 때 회전 회피
        self.declare_parameter("avoid_turn_speed", 0.25)

        # 사람과 가까워지면 translation 정지
        self.declare_parameter("person_stop_distance_m", 0.75)
        self.declare_parameter("person_resume_distance_m", 0.95)
        self.declare_parameter("person_emergency_distance_m", 0.45)

        # depth 처리
        self.declare_parameter("depth_min_m", 0.25)
        self.declare_parameter("depth_max_m", 4.0)
        self.declare_parameter("depth_min_pixels", 80)
        self.declare_parameter("depth_percentile", 20.0)

        self.cmd_raw_topic = str(self.get_parameter("cmd_raw_topic").value)
        self.cmd_out_topic = str(self.get_parameter("cmd_out_topic").value)
        self.scan_topic = str(self.get_parameter("scan_topic").value)
        self.depth_topic = str(self.get_parameter("depth_topic").value)
        self.person_state_topic = str(self.get_parameter("person_state_topic").value)
        self.state_topic = str(self.get_parameter("state_topic").value)

        self.publish_hz = float(self.get_parameter("publish_hz").value)
        self.cmd_timeout_sec = float(self.get_parameter("cmd_timeout_sec").value)
        self.raw_hold_sec = float(self.get_parameter("raw_hold_sec").value)

        self.linear_accel = float(self.get_parameter("linear_accel_mps2").value)
        self.strafe_accel = float(self.get_parameter("strafe_accel_mps2").value)
        self.angular_accel = float(self.get_parameter("angular_accel_rps2").value)

        self.max_linear_x = float(self.get_parameter("max_linear_x").value)
        self.max_linear_y = float(self.get_parameter("max_linear_y").value)
        self.max_angular_z = float(self.get_parameter("max_angular_z").value)

        self.front_stop_m = float(self.get_parameter("front_stop_m").value)
        self.front_slow_m = float(self.get_parameter("front_slow_m").value)
        self.front_emergency_m = float(self.get_parameter("front_emergency_m").value)

        self.enable_strafe_avoidance = bool(self.get_parameter("enable_strafe_avoidance").value)
        self.strafe_speed = float(self.get_parameter("strafe_speed").value)
        self.strafe_clear_m = float(self.get_parameter("strafe_clear_m").value)
        self.side_stop_m = float(self.get_parameter("side_stop_m").value)
        self.avoid_turn_speed = float(self.get_parameter("avoid_turn_speed").value)

        self.person_stop_distance_m = float(self.get_parameter("person_stop_distance_m").value)
        self.person_resume_distance_m = float(self.get_parameter("person_resume_distance_m").value)
        self.person_emergency_distance_m = float(self.get_parameter("person_emergency_distance_m").value)

        self.depth_min_m = float(self.get_parameter("depth_min_m").value)
        self.depth_max_m = float(self.get_parameter("depth_max_m").value)
        self.depth_min_pixels = int(self.get_parameter("depth_min_pixels").value)
        self.depth_percentile = float(self.get_parameter("depth_percentile").value)

        self.raw_cmd = zero_twist()
        self.raw_cmd_time = 0.0

        self.held_raw = zero_twist()
        self.held_raw_time = 0.0

        self.out_cmd = zero_twist()
        self.last_pub_time = time.time()

        self.lidar_front = math.inf
        self.lidar_left = math.inf
        self.lidar_right = math.inf
        self.last_scan_time = 0.0

        self.depth_front = math.inf
        self.depth_left = math.inf
        self.depth_right = math.inf
        self.depth_lower = math.inf
        self.last_depth_time = 0.0

        self.person_distance = math.inf
        self.person_close_latched = False
        self.last_person_state_time = 0.0

        self.action = "IDLE"

        self.create_subscription(Twist, self.cmd_raw_topic, self.on_cmd_raw, 10)
        self.create_subscription(LaserScan, self.scan_topic, self.on_scan, 10)
        self.create_subscription(Image, self.depth_topic, self.on_depth, 10)
        self.create_subscription(String, self.person_state_topic, self.on_person_state, 10)

        self.cmd_pub = self.create_publisher(Twist, self.cmd_out_topic, 10)
        self.state_pub = self.create_publisher(String, self.state_topic, 10)

        self.timer = self.create_timer(1.0 / max(self.publish_hz, 1.0), self.on_timer)

        self.get_logger().info(
            "depth_lidar_safety_guard started: "
            f"{self.cmd_raw_topic} -> {self.cmd_out_topic}, "
            f"scan={self.scan_topic}, depth={self.depth_topic}, "
            f"strafe={self.enable_strafe_avoidance}"
        )

    def now_sec(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def on_cmd_raw(self, msg):
        self.raw_cmd = copy_twist(msg)
        self.raw_cmd_time = self.now_sec()

    def sector_distance(self, msg, deg_min, deg_max):
        vals = []
        a = msg.angle_min
        for r in msg.ranges:
            deg = math.degrees(a)
            a += msg.angle_increment

            if deg < deg_min or deg > deg_max:
                continue

            if not math.isfinite(r):
                continue

            if r < max(float(msg.range_min), 0.05):
                continue

            if r > min(float(msg.range_max), 10.0):
                continue

            vals.append(float(r))

        if not vals:
            return math.inf

        return float(np.percentile(np.array(vals, dtype=np.float32), 10.0))

    def on_scan(self, msg):
        # scan_front 기준: 정면 0도, 좌측 +, 우측 -
        self.lidar_front = self.sector_distance(msg, -22.0, 22.0)
        self.lidar_left = self.sector_distance(msg, 25.0, 85.0)
        self.lidar_right = self.sector_distance(msg, -85.0, -25.0)
        self.last_scan_time = self.now_sec()

    def decode_depth(self, msg):
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
            return arr.astype(np.float32) / 1000.0

        if enc == "32FC1":
            dtype = np.dtype(">f4" if msg.is_bigendian else "<f4")
            row_bytes = w * 4
            raw = np.frombuffer(msg.data, dtype=np.uint8).reshape(h, step)
            raw = raw[:, :row_bytes].copy()
            arr = raw.view(dtype).reshape(h, w)
            return arr.astype(np.float32)

        raise RuntimeError(f"unsupported depth encoding: {enc}")

    def roi_distance(self, depth, x1, x2, y1, y2):
        h, w = depth.shape

        ix1 = int(clamp(x1 * w, 0, w - 1))
        ix2 = int(clamp(x2 * w, 0, w - 1))
        iy1 = int(clamp(y1 * h, 0, h - 1))
        iy2 = int(clamp(y2 * h, 0, h - 1))

        if ix2 <= ix1 or iy2 <= iy1:
            return math.inf

        roi = depth[iy1:iy2, ix1:ix2]
        valid = np.isfinite(roi)
        valid &= roi >= self.depth_min_m
        valid &= roi <= self.depth_max_m

        if int(np.count_nonzero(valid)) < self.depth_min_pixels:
            return math.inf

        return float(np.percentile(roi[valid], self.depth_percentile))

    def on_depth(self, msg):
        try:
            d = self.decode_depth(msg)
        except Exception as e:
            self.get_logger().warn(f"depth decode failed: {repr(e)}")
            return

        # 좌/중/우 + 하단 전방. 하단은 낮은 장애물/바닥 근처 물체 확인용.
        self.depth_left = self.roi_distance(d, 0.12, 0.42, 0.35, 0.82)
        self.depth_front = self.roi_distance(d, 0.38, 0.62, 0.30, 0.78)
        self.depth_right = self.roi_distance(d, 0.58, 0.88, 0.35, 0.82)
        self.depth_lower = self.roi_distance(d, 0.32, 0.68, 0.68, 0.95)

        self.last_depth_time = self.now_sec()

    def on_person_state(self, msg):
        # camera_person_follow_node의 state 형식이 조금 달라도 최대한 거리값을 뽑는다.
        txt = msg.data
        dist = None

        try:
            data = json.loads(txt)
            for k in (
                "person_distance_m",
                "distance_m",
                "target_distance_m",
                "depth_m",
                "person_depth_m",
            ):
                if k in data and data[k] is not None:
                    dist = float(data[k])
                    break
        except Exception:
            pass

        if dist is None:
            m = re.search(r"(?:distance|depth)[^0-9]*([0-9]+(?:\.[0-9]+)?)", txt, re.I)
            if m:
                dist = float(m.group(1))

        if dist is not None and math.isfinite(dist):
            self.person_distance = dist
            self.last_person_state_time = self.now_sec()

    def get_held_raw(self, now):
        fresh = (now - self.raw_cmd_time) <= self.cmd_timeout_sec
        if not fresh:
            self.held_raw = zero_twist()
            return zero_twist()

        proposed = copy_twist(self.raw_cmd)

        # 정지 명령은 즉시 반영
        if is_zero_cmd(proposed):
            self.held_raw = proposed
            self.held_raw_time = now
            return proposed

        # 기존 명령이 없으면 바로 수용
        if is_zero_cmd(self.held_raw):
            self.held_raw = proposed
            self.held_raw_time = now
            return proposed

        changed = False
        changed |= sign_changed(self.held_raw.linear.x, proposed.linear.x, 0.03)
        changed |= sign_changed(self.held_raw.linear.y, proposed.linear.y, 0.03)
        changed |= sign_changed(self.held_raw.angular.z, proposed.angular.z, 0.05)

        # 방향 전환이 너무 빠르면 기존 명령을 조금 더 유지
        if changed and (now - self.held_raw_time) < self.raw_hold_sec:
            return copy_twist(self.held_raw)

        self.held_raw = proposed
        self.held_raw_time = now
        return proposed

    def update_person_latch(self, now):
        # person state가 오래되면 latch 해제
        if now - self.last_person_state_time > 1.0:
            self.person_close_latched = False
            self.person_distance = math.inf
            return

        if self.person_distance <= self.person_stop_distance_m:
            self.person_close_latched = True

        if self.person_distance >= self.person_resume_distance_m:
            self.person_close_latched = False

    def apply_safety(self, cmd, now):
        target = copy_twist(cmd)
        self.action = "PASS"

        self.update_person_latch(now)

        # 속도 상한
        target.linear.x = clamp(target.linear.x, -self.max_linear_x, self.max_linear_x)
        target.linear.y = clamp(target.linear.y, -self.max_linear_y, self.max_linear_y)
        target.angular.z = clamp(target.angular.z, -self.max_angular_z, self.max_angular_z)

        scan_fresh = (now - self.last_scan_time) < 0.8
        depth_fresh = (now - self.last_depth_time) < 0.8

        lidar_front = self.lidar_front if scan_fresh else math.inf
        lidar_left = self.lidar_left if scan_fresh else math.inf
        lidar_right = self.lidar_right if scan_fresh else math.inf

        depth_front = min(self.depth_front, self.depth_lower) if depth_fresh else math.inf
        depth_left = self.depth_left if depth_fresh else math.inf
        depth_right = self.depth_right if depth_fresh else math.inf

        front = min(lidar_front, depth_front)
        left_clear = min(lidar_left, depth_left)
        right_clear = min(lidar_right, depth_right)

        # 사람이 가까우면 이동 정지. 회전은 아주 약하게만 허용.
        if self.person_close_latched:
            target.linear.x = 0.0
            target.linear.y = 0.0
            target.angular.z = clamp(target.angular.z, -0.20, 0.20)
            self.action = "PERSON_CLOSE_STOP"

            if self.person_distance <= self.person_emergency_distance_m:
                target.angular.z = 0.0
                self.action = "PERSON_EMERGENCY_STOP"

            return target, front, left_clear, right_clear

        # 긴급 전방 정지
        if target.linear.x > 0.0 and front <= self.front_emergency_m:
            target.linear.x = 0.0
            target.linear.y = 0.0
            target.angular.z = 0.0
            self.action = "EMERGENCY_FRONT_STOP"
            return target, front, left_clear, right_clear

        # 전방 장애물: 전진 차단 + 가능하면 횡이동
        if target.linear.x > 0.0 and front <= self.front_stop_m:
            target.linear.x = 0.0

            if self.enable_strafe_avoidance:
                if left_clear >= self.strafe_clear_m or right_clear >= self.strafe_clear_m:
                    if left_clear >= right_clear:
                        target.linear.y = abs(self.strafe_speed)
                        self.action = "FRONT_BLOCK_STRAFE_LEFT"
                    else:
                        target.linear.y = -abs(self.strafe_speed)
                        self.action = "FRONT_BLOCK_STRAFE_RIGHT"

                    # 회전까지 같이 하면 틱틱거릴 수 있어서 횡이동 중에는 회전 줄임
                    target.angular.z *= 0.35
                else:
                    target.linear.y = 0.0
                    target.angular.z = self.avoid_turn_speed if left_clear >= right_clear else -self.avoid_turn_speed
                    self.action = "FRONT_BLOCK_TURN"
            else:
                target.linear.y = 0.0
                target.angular.z = self.avoid_turn_speed if left_clear >= right_clear else -self.avoid_turn_speed
                self.action = "FRONT_BLOCK_TURN"

        # 전방 감속 구간
        elif target.linear.x > 0.0 and front <= self.front_slow_m:
            ratio = (front - self.front_stop_m) / max(self.front_slow_m - self.front_stop_m, 1e-6)
            ratio = clamp(ratio, 0.20, 1.0)
            target.linear.x *= ratio
            self.action = "FRONT_SLOW"

        # 좌우 너무 가까우면 그 방향 횡이동 금지
        if target.linear.y > 0.0 and left_clear <= self.side_stop_m:
            target.linear.y = 0.0
            if self.action == "PASS":
                self.action = "LEFT_SIDE_STOP"

        if target.linear.y < 0.0 and right_clear <= self.side_stop_m:
            target.linear.y = 0.0
            if self.action == "PASS":
                self.action = "RIGHT_SIDE_STOP"

        return target, front, left_clear, right_clear

    def smooth_output(self, target, dt):
        out = copy_twist(self.out_cmd)

        # 부호가 반대로 바뀔 때는 한 번에 반전하지 않고 0을 거쳐감
        tx = target.linear.x
        ty = target.linear.y
        tz = target.angular.z

        if sign_changed(out.linear.x, tx, 0.02):
            tx = 0.0
        if sign_changed(out.linear.y, ty, 0.02):
            ty = 0.0
        if sign_changed(out.angular.z, tz, 0.04):
            tz = 0.0

        out.linear.x = approach(out.linear.x, tx, self.linear_accel * dt)
        out.linear.y = approach(out.linear.y, ty, self.strafe_accel * dt)
        out.angular.z = approach(out.angular.z, tz, self.angular_accel * dt)

        # 아주 작은 값 제거
        if abs(out.linear.x) < 0.005:
            out.linear.x = 0.0
        if abs(out.linear.y) < 0.005:
            out.linear.y = 0.0
        if abs(out.angular.z) < 0.01:
            out.angular.z = 0.0

        self.out_cmd = out
        return out

    def publish_state(self, front, left_clear, right_clear):
        payload = {
            "action": self.action,
            "lidar_front_m": None if not math.isfinite(self.lidar_front) else round(self.lidar_front, 3),
            "depth_front_m": None if not math.isfinite(self.depth_front) else round(self.depth_front, 3),
            "depth_lower_m": None if not math.isfinite(self.depth_lower) else round(self.depth_lower, 3),
            "front_m": None if not math.isfinite(front) else round(front, 3),
            "left_clear_m": None if not math.isfinite(left_clear) else round(left_clear, 3),
            "right_clear_m": None if not math.isfinite(right_clear) else round(right_clear, 3),
            "person_distance_m": None if not math.isfinite(self.person_distance) else round(self.person_distance, 3),
            "person_close_latched": self.person_close_latched,
            "out": {
                "x": round(self.out_cmd.linear.x, 3),
                "y": round(self.out_cmd.linear.y, 3),
                "wz": round(self.out_cmd.angular.z, 3),
            },
        }
        msg = String()
        msg.data = json.dumps(payload, ensure_ascii=False)
        self.state_pub.publish(msg)

    def on_timer(self):
        now = self.now_sec()

        raw = self.get_held_raw(now)
        safe_target, front, left_clear, right_clear = self.apply_safety(raw, now)

        wall_now = time.time()
        dt = wall_now - self.last_pub_time
        self.last_pub_time = wall_now
        dt = clamp(dt, 0.001, 0.20)

        out = self.smooth_output(safe_target, dt)
        self.cmd_pub.publish(out)
        self.publish_state(front, left_clear, right_clear)


def main(args=None):
    rclpy.init(args=args)
    node = DepthLidarSafetyGuard()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.cmd_pub.publish(zero_twist())
        except Exception:
            pass
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
