#!/usr/bin/env python3

import math
import time
from typing import Optional, Tuple

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, TransformStamped
from nav_msgs.msg import Odometry
from tf2_ros import TransformBroadcaster

try:
    import serial
except ImportError as exc:
    serial = None
    SERIAL_IMPORT_ERROR = exc
else:
    SERIAL_IMPORT_ERROR = None


class BaseKeyOdomSerialNode(Node):
    """
    기존 Arduino test.ino 프로토콜을 그대로 사용한다.

    Arduino command:
        w : forward
        s : backward
        a : turn left
        d : turn right
        q : strafe left
        e : strafe right
        x : stop

    Arduino encoder output:
        ENC LF RF LR RR

    ROS:
        subscribe: /cmd_vel
        publish  : /odom
        tf       : odom -> base_link
    """

    def __init__(self) -> None:
        super().__init__("base_key_odom_serial_node")

        self.declare_parameter("port", "/dev/arduino")
        self.declare_parameter("baudrate", 115200)

        self.declare_parameter("cmd_topic", "/cmd_vel")
        self.declare_parameter("odom_topic", "/odom")

        self.declare_parameter("odom_frame", "odom")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("publish_tf", True)

        self.declare_parameter("send_rate_hz", 20.0)
        self.declare_parameter("command_timeout_sec", 0.5)

        self.declare_parameter("linear_threshold", 0.03)
        self.declare_parameter("angular_threshold", 0.10)

        self.declare_parameter("ticks_per_rev", 3464.0)
        self.declare_parameter("wheel_radius", 0.075)
        self.declare_parameter("lx", 0.1575)
        self.declare_parameter("ly", 0.2125)

        self.declare_parameter("open_retry_sec", 1.0)
        self.declare_parameter("serial_write_timeout_sec", 0.05)
        self.declare_parameter("arduino_reset_wait_sec", 2.0)

        self.declare_parameter("log_sent_key", True)
        self.declare_parameter("log_encoder_line", False)

        self.port = str(self.get_parameter("port").value)
        self.baudrate = int(self.get_parameter("baudrate").value)

        self.cmd_topic = str(self.get_parameter("cmd_topic").value)
        self.odom_topic = str(self.get_parameter("odom_topic").value)

        self.odom_frame = str(self.get_parameter("odom_frame").value)
        self.base_frame = str(self.get_parameter("base_frame").value)
        self.publish_tf = bool(self.get_parameter("publish_tf").value)

        self.send_rate_hz = float(self.get_parameter("send_rate_hz").value)
        self.command_timeout_sec = float(self.get_parameter("command_timeout_sec").value)

        self.linear_threshold = float(self.get_parameter("linear_threshold").value)
        self.angular_threshold = float(self.get_parameter("angular_threshold").value)

        self.ticks_per_rev = float(self.get_parameter("ticks_per_rev").value)
        self.wheel_radius = float(self.get_parameter("wheel_radius").value)
        self.lx = float(self.get_parameter("lx").value)
        self.ly = float(self.get_parameter("ly").value)

        self.open_retry_sec = float(self.get_parameter("open_retry_sec").value)
        self.serial_write_timeout_sec = float(
            self.get_parameter("serial_write_timeout_sec").value
        )
        self.arduino_reset_wait_sec = float(
            self.get_parameter("arduino_reset_wait_sec").value
        )

        self.log_sent_key = bool(self.get_parameter("log_sent_key").value)
        self.log_encoder_line = bool(self.get_parameter("log_encoder_line").value)

        if serial is None:
            raise RuntimeError(
                f"pyserial import failed: {SERIAL_IMPORT_ERROR}. "
                "Install with: sudo apt install python3-serial"
            )

        self.ser: Optional["serial.Serial"] = None
        self.last_open_attempt = 0.0
        self.opened_time = 0.0

        self.latest_cmd = Twist()
        self.last_cmd_time = 0.0
        self.last_sent_key: Optional[str] = None

        self.prev_ticks: Optional[Tuple[int, int, int, int]] = None
        self.prev_odom_time: Optional[float] = None

        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0

        self.last_vx = 0.0
        self.last_vy = 0.0
        self.last_wz = 0.0

        self.cmd_sub = self.create_subscription(
            Twist,
            self.cmd_topic,
            self.on_cmd,
            10,
        )

        self.odom_pub = self.create_publisher(
            Odometry,
            self.odom_topic,
            10,
        )

        self.tf_broadcaster = TransformBroadcaster(self)

        period = 1.0 / max(self.send_rate_hz, 1.0)
        self.timer = self.create_timer(period, self.on_timer)

        self.get_logger().info("base_key_odom_serial_node started")
        self.get_logger().info(f"  port={self.port}")
        self.get_logger().info(f"  baudrate={self.baudrate}")
        self.get_logger().info(f"  cmd_topic={self.cmd_topic}")
        self.get_logger().info(f"  odom_topic={self.odom_topic}")
        self.get_logger().info(f"  ticks_per_rev={self.ticks_per_rev}")
        self.get_logger().info(f"  wheel_radius={self.wheel_radius}")
        self.get_logger().info(f"  lx={self.lx}")
        self.get_logger().info(f"  ly={self.ly}")

    def on_cmd(self, msg: Twist) -> None:
        self.latest_cmd = msg
        self.last_cmd_time = time.monotonic()

    def on_timer(self) -> None:
        self._ensure_serial_open()

        if self.ser is None or not self.ser.is_open:
            return

        now = time.monotonic()

        if now - self.opened_time < self.arduino_reset_wait_sec:
            self._read_serial_lines()
            return

        self._send_current_key(now)
        self._read_serial_lines()

    def _send_current_key(self, now: float) -> None:
        if now - self.last_cmd_time > self.command_timeout_sec:
            key = "x"
        else:
            key = self._twist_to_key(self.latest_cmd)

        try:
            self.ser.write(key.encode("ascii"))
            self.ser.flush()
        except Exception as exc:
            self.get_logger().warn(f"serial write failed: {exc}")
            self._close_serial()
            return

        if self.log_sent_key and key != self.last_sent_key:
            self.get_logger().info(f"sent key: {key}")
            self.last_sent_key = key

    def _read_serial_lines(self) -> None:
        if self.ser is None:
            return

        try:
            while self.ser.in_waiting > 0:
                raw = self.ser.readline()
                if not raw:
                    break

                line = raw.decode("utf-8", errors="ignore").strip()

                if not line:
                    continue

                if self.log_encoder_line:
                    self.get_logger().info(f"arduino: {line}")

                if line.startswith("ENC "):
                    self._handle_encoder_line(line)

        except Exception as exc:
            self.get_logger().warn(f"serial read failed: {exc}")
            self._close_serial()

    def _handle_encoder_line(self, line: str) -> None:
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

        ticks = (lf, rf, lr, rr)
        now = time.monotonic()

        if self.prev_ticks is None:
            self.prev_ticks = ticks
            self.prev_odom_time = now
            self._publish_odom()
            return

        if self.prev_odom_time is None:
            self.prev_ticks = ticks
            self.prev_odom_time = now
            return

        dt = now - self.prev_odom_time
        if dt <= 0.0:
            return

        d_lf_ticks = ticks[0] - self.prev_ticks[0]
        d_rf_ticks = ticks[1] - self.prev_ticks[1]
        d_lr_ticks = ticks[2] - self.prev_ticks[2]
        d_rr_ticks = ticks[3] - self.prev_ticks[3]

        self.prev_ticks = ticks
        self.prev_odom_time = now

        meters_per_tick = (2.0 * math.pi * self.wheel_radius) / self.ticks_per_rev

        d_lf = d_lf_ticks * meters_per_tick
        d_rf = d_rf_ticks * meters_per_tick
        d_lr = d_lr_ticks * meters_per_tick
        d_rr = d_rr_ticks * meters_per_tick

        # Mecanum inverse kinematics.
        #
        # Arduino drive convention:
        #   LF = x + y + wz
        #   RF = x - y - wz
        #   LR = x - y + wz
        #   RR = x + y - wz
        #
        # Physical odom:
        #   dx     = (LF + RF + LR + RR) / 4
        #   dy     = (LF - RF - LR + RR) / 4
        #   dtheta = (LF - RF + LR - RR) / (4 * (lx + ly))

        dx_body = (d_lf + d_rf + d_lr + d_rr) / 4.0
        dy_body = (d_lf - d_rf - d_lr + d_rr) / 4.0

        radius_sum = self.lx + self.ly
        if radius_sum <= 0.0:
            dtheta = 0.0
        else:
            dtheta = (d_lf - d_rf + d_lr - d_rr) / (4.0 * radius_sum)

        mid_yaw = self.yaw + 0.5 * dtheta

        cos_yaw = math.cos(mid_yaw)
        sin_yaw = math.sin(mid_yaw)

        self.x += dx_body * cos_yaw - dy_body * sin_yaw
        self.y += dx_body * sin_yaw + dy_body * cos_yaw
        self.yaw = self._normalize_angle(self.yaw + dtheta)

        self.last_vx = dx_body / dt
        self.last_vy = dy_body / dt
        self.last_wz = dtheta / dt

        self._publish_odom()

    def _publish_odom(self) -> None:
        stamp = self.get_clock().now().to_msg()

        qz = math.sin(self.yaw * 0.5)
        qw = math.cos(self.yaw * 0.5)

        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = self.odom_frame
        odom.child_frame_id = self.base_frame

        odom.pose.pose.position.x = self.x
        odom.pose.pose.position.y = self.y
        odom.pose.pose.position.z = 0.0

        odom.pose.pose.orientation.x = 0.0
        odom.pose.pose.orientation.y = 0.0
        odom.pose.pose.orientation.z = qz
        odom.pose.pose.orientation.w = qw

        odom.twist.twist.linear.x = self.last_vx
        odom.twist.twist.linear.y = self.last_vy
        odom.twist.twist.angular.z = self.last_wz

        odom.pose.covariance[0] = 0.05
        odom.pose.covariance[7] = 0.05
        odom.pose.covariance[35] = 0.10

        odom.twist.covariance[0] = 0.10
        odom.twist.covariance[7] = 0.10
        odom.twist.covariance[35] = 0.20

        self.odom_pub.publish(odom)

        if self.publish_tf:
            tf = TransformStamped()
            tf.header.stamp = stamp
            tf.header.frame_id = self.odom_frame
            tf.child_frame_id = self.base_frame

            tf.transform.translation.x = self.x
            tf.transform.translation.y = self.y
            tf.transform.translation.z = 0.0

            tf.transform.rotation.x = 0.0
            tf.transform.rotation.y = 0.0
            tf.transform.rotation.z = qz
            tf.transform.rotation.w = qw

            self.tf_broadcaster.sendTransform(tf)

    def _twist_to_key(self, msg: Twist) -> str:
        x = self._finite(msg.linear.x)
        y = self._finite(msg.linear.y)
        wz = self._finite(msg.angular.z)

        ax = abs(x)
        ay = abs(y)
        aw = abs(wz)

        if (
            ax < self.linear_threshold
            and ay < self.linear_threshold
            and aw < self.angular_threshold
        ):
            return "x"

        if aw >= ax and aw >= ay:
            if wz > 0.0:
                return "a"
            return "d"

        if ay >= ax and ay >= aw:
            if y > 0.0:
                return "q"
            return "e"

        if x > 0.0:
            return "w"

        return "s"

    @staticmethod
    def _finite(value: float) -> float:
        value = float(value)
        if not math.isfinite(value):
            return 0.0
        return value

    @staticmethod
    def _normalize_angle(angle: float) -> float:
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle

    def _ensure_serial_open(self) -> None:
        if self.ser is not None and self.ser.is_open:
            return

        now = time.monotonic()
        if now - self.last_open_attempt < self.open_retry_sec:
            return

        self.last_open_attempt = now

        try:
            self.ser = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=0.001,
                write_timeout=self.serial_write_timeout_sec,
            )
        except Exception as exc:
            self.get_logger().warn(f"failed to open serial port {self.port}: {exc}")
            self.ser = None
            return

        self.opened_time = time.monotonic()
        self.get_logger().info(f"opened serial port: {self.port}")
        self.get_logger().info(
            f"waiting {self.arduino_reset_wait_sec:.1f}s for Arduino reset"
        )

    def _close_serial(self) -> None:
        if self.ser is None:
            return

        try:
            if self.ser.is_open:
                self.ser.close()
        except Exception:
            pass

        self.ser = None

    def destroy_node(self) -> bool:
        try:
            if self.ser is not None and self.ser.is_open:
                self.ser.write(b"x")
                self.ser.flush()
                self.ser.close()
        except Exception:
            pass

        return super().destroy_node()


def main(args=None) -> None:
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
