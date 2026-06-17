#!/usr/bin/env python3

import math
import time

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist, TransformStamped
from nav_msgs.msg import Odometry

from tf2_ros import TransformBroadcaster

try:
    import serial
except ImportError as e:
    serial = None


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def yaw_to_quat(yaw):
    half = yaw * 0.5
    return {
        "x": 0.0,
        "y": 0.0,
        "z": math.sin(half),
        "w": math.cos(half),
    }


class BaseKeyOdomSerialNode(Node):
    """
    /cmd_vel -> Arduino keyboard protocol + encoder odom.

    Arduino protocol:
      w/s : forward/backward
      a/d : rotate left/right
      q/e : strafe left/right
      r/t : diagonal front-left/front-right
      f/g : diagonal back-left/back-right
      x   : stop

    이 버전의 핵심:
      - Arduino 코드는 그대로 둠
      - linear.x + angular.z 동시 명령을 w/a/d 시간분할 키 스트림으로 변환
      - YOLO person follow의 FORWARD_STEER_PERSON을 실제로 더 부드럽게 반영
      - encoder line: ENC LF RF LR RR
    """

    def __init__(self):
        super().__init__("base_key_odom_serial_node")

        self.declare_parameter("port", "/dev/arduino")
        self.declare_parameter("baudrate", 115200)

        self.declare_parameter("cmd_topic", "/cmd_vel")
        self.declare_parameter("odom_topic", "/odom")

        self.declare_parameter("odom_frame_id", "odom")
        self.declare_parameter("base_frame_id", "base_link")
        self.declare_parameter("publish_tf", True)

        self.declare_parameter("ticks_per_rev", 3464.0)
        self.declare_parameter("wheel_radius", 0.075)
        self.declare_parameter("lx", 0.1575)
        self.declare_parameter("ly", 0.2125)

        self.declare_parameter("cmd_timeout_sec", 0.50)
        self.declare_parameter("key_hz", 20.0)
        self.declare_parameter("odom_publish_hz", 20.0)
        self.declare_parameter("publish_open_loop_odom", True)
        self.declare_parameter("encoder_timeout_sec", 0.30)

        self.declare_parameter("linear_deadband", 0.03)
        self.declare_parameter("angular_deadband", 0.05)
        self.declare_parameter("strafe_deadband", 0.03)

        self.declare_parameter("enable_strafe", False)
        self.declare_parameter("enable_diagonal_keys", False)

        # vx+wz 동시 명령에서 회전 키를 섞는 비율
        self.declare_parameter("mixed_forward_turn", True)
        self.declare_parameter("turn_mix_gain", 1.25)
        self.declare_parameter("turn_mix_min_duty", 0.22)
        self.declare_parameter("turn_mix_max_duty", 0.55)
        self.declare_parameter("angular_ref", 0.45)

        self.declare_parameter("log_sent_key", True)
        self.declare_parameter("log_all_keys", False)
        self.declare_parameter("log_encoder_line", False)

        self.port = str(self.get_parameter("port").value)
        self.baudrate = int(self.get_parameter("baudrate").value)

        self.cmd_topic = str(self.get_parameter("cmd_topic").value)
        self.odom_topic = str(self.get_parameter("odom_topic").value)

        self.odom_frame_id = str(self.get_parameter("odom_frame_id").value)
        self.base_frame_id = str(self.get_parameter("base_frame_id").value)
        self.publish_tf = bool(self.get_parameter("publish_tf").value)

        self.ticks_per_rev = float(self.get_parameter("ticks_per_rev").value)
        self.wheel_radius = float(self.get_parameter("wheel_radius").value)
        self.lx = float(self.get_parameter("lx").value)
        self.ly = float(self.get_parameter("ly").value)

        self.cmd_timeout_sec = float(self.get_parameter("cmd_timeout_sec").value)
        self.key_hz = float(self.get_parameter("key_hz").value)
        self.odom_publish_hz = float(self.get_parameter("odom_publish_hz").value)
        self.publish_open_loop_odom = bool(self.get_parameter("publish_open_loop_odom").value)
        self.encoder_timeout_sec = float(self.get_parameter("encoder_timeout_sec").value)

        self.linear_deadband = float(self.get_parameter("linear_deadband").value)
        self.angular_deadband = float(self.get_parameter("angular_deadband").value)
        self.strafe_deadband = float(self.get_parameter("strafe_deadband").value)

        self.enable_strafe = bool(self.get_parameter("enable_strafe").value)
        self.enable_diagonal_keys = bool(self.get_parameter("enable_diagonal_keys").value)

        self.mixed_forward_turn = bool(self.get_parameter("mixed_forward_turn").value)
        self.turn_mix_gain = float(self.get_parameter("turn_mix_gain").value)
        self.turn_mix_min_duty = float(self.get_parameter("turn_mix_min_duty").value)
        self.turn_mix_max_duty = float(self.get_parameter("turn_mix_max_duty").value)
        self.angular_ref = float(self.get_parameter("angular_ref").value)

        self.log_sent_key = bool(self.get_parameter("log_sent_key").value)
        self.log_all_keys = bool(self.get_parameter("log_all_keys").value)
        self.log_encoder_line = bool(self.get_parameter("log_encoder_line").value)

        if serial is None:
            raise RuntimeError("pyserial is not installed. Install with: python -m pip install pyserial")

        self.ser = serial.Serial(
            self.port,
            self.baudrate,
            timeout=0.005,
            write_timeout=0.05,
        )

        time.sleep(1.5)
        self.ser.reset_input_buffer()
        self.ser.reset_output_buffer()

        self.last_cmd = Twist()
        self.last_cmd_time = 0.0

        self.last_key = None
        self.stop_key_last_sent = 0.0
        self.turn_mix_acc = 0.0

        self.prev_enc = None
        self.last_encoder_time = 0.0
        self.last_open_loop_stamp = self.get_clock().now()

        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0

        self.last_odom_stamp = self.get_clock().now()

        self.cmd_sub = self.create_subscription(
            Twist,
            self.cmd_topic,
            self.on_cmd,
            10,
        )

        self.odom_pub = self.create_publisher(Odometry, self.odom_topic, 10)
        self.tf_broadcaster = TransformBroadcaster(self)

        self.read_timer = self.create_timer(0.01, self.read_serial_once)
        self.key_timer = self.create_timer(1.0 / max(self.key_hz, 1.0), self.send_key_once)
        self.odom_timer = self.create_timer(
            1.0 / max(self.odom_publish_hz, 1.0),
            self.publish_open_loop_odom_once,
        )

        self.get_logger().info(
            "base key odom serial node started: "
            f"port={self.port}, baudrate={self.baudrate}, "
            f"cmd={self.cmd_topic}, odom={self.odom_topic}, "
            f"key_hz={self.key_hz}, mixed_forward_turn={self.mixed_forward_turn}, "
            f"open_loop_odom={self.publish_open_loop_odom}"
        )

    def now_sec(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def on_cmd(self, msg: Twist):
        self.last_cmd = msg
        self.last_cmd_time = self.now_sec()

    def write_key(self, key):
        if not key:
            key = "x"

        try:
            self.ser.write(key.encode("ascii"))
            self.ser.flush()
        except Exception as e:
            self.get_logger().error(f"serial write failed: {repr(e)}")
            return

        if self.log_sent_key:
            if self.log_all_keys or key != self.last_key:
                self.get_logger().info(f"sent key: {key}")

        self.last_key = key

    def select_key_from_cmd(self, cmd):
        vx = float(cmd.linear.x)
        vy = float(cmd.linear.y)
        wz = float(cmd.angular.z)

        has_vx = abs(vx) >= self.linear_deadband
        has_vy = abs(vy) >= self.strafe_deadband
        has_wz = abs(wz) >= self.angular_deadband

        if not has_vx and not has_vy and not has_wz:
            return "x"

        forward_key = "w" if vx > 0.0 else "s"
        turn_key = "a" if wz > 0.0 else "d"
        strafe_key = "q" if vy > 0.0 else "e"

        # strafe는 지금 사람 추종에서는 꺼둠.
        if has_vy and self.enable_strafe and not has_vx and not has_wz:
            return strafe_key

        # 전진/후진 + 회전 동시 명령
        if has_vx and has_wz and self.mixed_forward_turn:
            ref = max(abs(self.angular_ref), 1e-6)
            duty = clamp(
                abs(wz) / ref * self.turn_mix_gain,
                self.turn_mix_min_duty,
                self.turn_mix_max_duty,
            )

            self.turn_mix_acc += duty

            if self.turn_mix_acc >= 1.0:
                self.turn_mix_acc -= 1.0
                return turn_key

            return forward_key

        # 대각선 키를 쓰고 싶을 때만 사용. 기본은 비활성.
        if has_vx and has_vy and self.enable_diagonal_keys:
            if vx > 0.0 and vy > 0.0:
                return "r"  # front-left
            if vx > 0.0 and vy < 0.0:
                return "t"  # front-right
            if vx < 0.0 and vy > 0.0:
                return "f"  # back-left
            if vx < 0.0 and vy < 0.0:
                return "g"  # back-right

        if has_wz and not has_vx:
            return turn_key

        if has_vx:
            return forward_key

        if has_vy and self.enable_strafe:
            return strafe_key

        return "x"

    def send_key_once(self):
        now = self.now_sec()

        cmd_fresh = (now - self.last_cmd_time) <= self.cmd_timeout_sec

        if not cmd_fresh:
            # timeout 때는 x를 너무 많이 찍지 않도록 주기 제한
            if self.last_key != "x" or now - self.stop_key_last_sent > 0.5:
                self.write_key("x")
                self.stop_key_last_sent = now
            return

        key = self.select_key_from_cmd(self.last_cmd)
        self.write_key(key)

    def read_serial_once(self):
        try:
            while self.ser.in_waiting > 0:
                raw = self.ser.readline()
                if not raw:
                    return

                line = raw.decode("utf-8", errors="ignore").strip()
                if not line:
                    continue

                if self.log_encoder_line:
                    self.get_logger().info(f"serial: {line}")

                if line.startswith("ENC "):
                    self.handle_encoder_line(line)
                else:
                    # READY 등 일반 메시지
                    if self.log_encoder_line:
                        self.get_logger().info(f"arduino: {line}")

        except Exception as e:
            self.get_logger().warn(f"serial read failed: {repr(e)}")

    def handle_encoder_line(self, line):
        parts = line.split()

        if len(parts) != 5:
            return

        try:
            lf = int(parts[1])
            rf = int(parts[2])
            lr = int(parts[3])
            rr = int(parts[4])
        except ValueError:
            return

        enc = (lf, rf, lr, rr)
        self.last_encoder_time = self.now_sec()

        if self.prev_enc is None:
            self.prev_enc = enc
            self.publish_odom(0.0, 0.0, 0.0)
            return

        dlf = lf - self.prev_enc[0]
        drf = rf - self.prev_enc[1]
        dlr = lr - self.prev_enc[2]
        drr = rr - self.prev_enc[3]

        self.prev_enc = enc

        meters_per_tick = (2.0 * math.pi * self.wheel_radius) / max(self.ticks_per_rev, 1e-6)

        fl = dlf * meters_per_tick
        fr = drf * meters_per_tick
        rl = dlr * meters_per_tick
        rr_m = drr * meters_per_tick

        # Mecanum forward/right/yaw body delta.
        # encoder line은 LF RF LR RR 순서.
        dx_body = (fl + fr + rl + rr_m) * 0.25
        dy_body = (-fl + fr + rl - rr_m) * 0.25

        base_sum = max(self.lx + self.ly, 1e-6)
        dyaw = (-fl + fr - rl + rr_m) / (4.0 * base_sum)

        self.integrate_odom(dx_body, dy_body, dyaw)

    def publish_open_loop_odom_once(self):
        if not self.publish_open_loop_odom:
            return

        now = self.get_clock().now()
        now_sec = now.nanoseconds * 1e-9

        encoder_alive = (
            self.last_encoder_time > 0.0
            and now_sec - self.last_encoder_time <= self.encoder_timeout_sec
        )
        if encoder_alive:
            self.last_open_loop_stamp = now
            return

        dt = (now.nanoseconds - self.last_open_loop_stamp.nanoseconds) * 1e-9
        if dt <= 0.0:
            return

        self.last_open_loop_stamp = now

        cmd_fresh = (now_sec - self.last_cmd_time) <= self.cmd_timeout_sec
        if not cmd_fresh:
            self.publish_odom(0.0, 0.0, 0.0)
            return

        vx = float(self.last_cmd.linear.x)
        vy = float(self.last_cmd.linear.y) if self.enable_strafe else 0.0
        wz = float(self.last_cmd.angular.z)

        dx_body = vx * dt
        dy_body = vy * dt
        dyaw = wz * dt

        self.integrate_odom(dx_body, dy_body, dyaw)

    def integrate_odom(self, dx_body, dy_body, dyaw):
        yaw_mid = self.yaw + 0.5 * dyaw

        dx_world = math.cos(yaw_mid) * dx_body - math.sin(yaw_mid) * dy_body
        dy_world = math.sin(yaw_mid) * dx_body + math.cos(yaw_mid) * dy_body

        self.x += dx_world
        self.y += dy_world
        self.yaw = math.atan2(math.sin(self.yaw + dyaw), math.cos(self.yaw + dyaw))

        self.publish_odom(dx_body, dy_body, dyaw)

    def publish_odom(self, dx_body, dy_body, dyaw):
        stamp = self.get_clock().now()

        dt = (stamp.nanoseconds - self.last_odom_stamp.nanoseconds) * 1e-9
        if dt <= 1e-6:
            dt = 1e-6

        self.last_odom_stamp = stamp

        q = yaw_to_quat(self.yaw)

        odom = Odometry()
        odom.header.stamp = stamp.to_msg()
        odom.header.frame_id = self.odom_frame_id
        odom.child_frame_id = self.base_frame_id

        odom.pose.pose.position.x = self.x
        odom.pose.pose.position.y = self.y
        odom.pose.pose.position.z = 0.0

        odom.pose.pose.orientation.x = q["x"]
        odom.pose.pose.orientation.y = q["y"]
        odom.pose.pose.orientation.z = q["z"]
        odom.pose.pose.orientation.w = q["w"]

        odom.twist.twist.linear.x = dx_body / dt
        odom.twist.twist.linear.y = dy_body / dt
        odom.twist.twist.angular.z = dyaw / dt

        # 대충 안정적인 covariance
        odom.pose.covariance[0] = 0.05
        odom.pose.covariance[7] = 0.05
        odom.pose.covariance[35] = 0.10
        odom.twist.covariance[0] = 0.10
        odom.twist.covariance[7] = 0.10
        odom.twist.covariance[35] = 0.20

        self.odom_pub.publish(odom)

        if self.publish_tf:
            tf = TransformStamped()
            tf.header.stamp = stamp.to_msg()
            tf.header.frame_id = self.odom_frame_id
            tf.child_frame_id = self.base_frame_id

            tf.transform.translation.x = self.x
            tf.transform.translation.y = self.y
            tf.transform.translation.z = 0.0

            tf.transform.rotation.x = q["x"]
            tf.transform.rotation.y = q["y"]
            tf.transform.rotation.z = q["z"]
            tf.transform.rotation.w = q["w"]

            self.tf_broadcaster.sendTransform(tf)

    def destroy_node(self):
        try:
            self.write_key("x")
        except Exception:
            pass

        try:
            if hasattr(self, "ser") and self.ser is not None:
                self.ser.close()
        except Exception:
            pass

        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)

    node = BaseKeyOdomSerialNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
