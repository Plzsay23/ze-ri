#!/usr/bin/env python3

import math
import time
from typing import Optional

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

try:
    import serial
except ImportError as exc:
    serial = None
    SERIAL_IMPORT_ERROR = exc
else:
    SERIAL_IMPORT_ERROR = None


class CmdVelSerialNode(Node):
    """
    Subscribe geometry_msgs/Twist and send velocity command to Arduino over serial.

    Serial protocol, one line per command:

        V <linear_x> <linear_y> <angular_z>\n

    Example:

        V 0.150 0.000 0.000

    Units:
        linear_x  : m/s
        linear_y  : m/s
        angular_z : rad/s

    Arduino side must parse this line format.
    """

    def __init__(self) -> None:
        super().__init__("cmd_vel_serial_node")

        self.declare_parameter("port", "/dev/arduino")
        self.declare_parameter("baudrate", 115200)
        self.declare_parameter("cmd_topic", "/cmd_vel")
        self.declare_parameter("send_rate_hz", 20.0)
        self.declare_parameter("command_timeout_sec", 0.5)

        self.declare_parameter("max_linear_x", 0.30)
        self.declare_parameter("max_linear_y", 0.30)
        self.declare_parameter("max_angular_z", 1.50)

        self.declare_parameter("serial_write_timeout_sec", 0.05)
        self.declare_parameter("open_retry_sec", 1.0)
        self.declare_parameter("log_sent_command", False)

        self.port = str(self.get_parameter("port").value)
        self.baudrate = int(self.get_parameter("baudrate").value)
        self.cmd_topic = str(self.get_parameter("cmd_topic").value)
        self.send_rate_hz = float(self.get_parameter("send_rate_hz").value)
        self.command_timeout_sec = float(self.get_parameter("command_timeout_sec").value)

        self.max_linear_x = float(self.get_parameter("max_linear_x").value)
        self.max_linear_y = float(self.get_parameter("max_linear_y").value)
        self.max_angular_z = float(self.get_parameter("max_angular_z").value)

        self.serial_write_timeout_sec = float(
            self.get_parameter("serial_write_timeout_sec").value
        )
        self.open_retry_sec = float(self.get_parameter("open_retry_sec").value)
        self.log_sent_command = bool(self.get_parameter("log_sent_command").value)

        self.ser: Optional["serial.Serial"] = None
        self.last_open_attempt = 0.0

        self.last_cmd_time = 0.0
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_wz = 0.0

        if serial is None:
            raise RuntimeError(
                f"pyserial import failed: {SERIAL_IMPORT_ERROR}. "
                "Install with: sudo apt install python3-serial"
            )

        self.sub = self.create_subscription(
            Twist,
            self.cmd_topic,
            self.on_cmd_vel,
            10,
        )

        period = 1.0 / max(self.send_rate_hz, 1.0)
        self.timer = self.create_timer(period, self.on_timer)

        self.get_logger().info("cmd_vel_serial_node started")
        self.get_logger().info(f"  port={self.port}")
        self.get_logger().info(f"  baudrate={self.baudrate}")
        self.get_logger().info(f"  cmd_topic={self.cmd_topic}")
        self.get_logger().info(f"  send_rate_hz={self.send_rate_hz}")
        self.get_logger().info(f"  command_timeout_sec={self.command_timeout_sec}")

    def on_cmd_vel(self, msg: Twist) -> None:
        self.current_x = self._clamp_finite(
            msg.linear.x,
            -self.max_linear_x,
            self.max_linear_x,
        )
        self.current_y = self._clamp_finite(
            msg.linear.y,
            -self.max_linear_y,
            self.max_linear_y,
        )
        self.current_wz = self._clamp_finite(
            msg.angular.z,
            -self.max_angular_z,
            self.max_angular_z,
        )
        self.last_cmd_time = time.monotonic()

    def on_timer(self) -> None:
        self._ensure_serial_open()

        if self.ser is None or not self.ser.is_open:
            return

        now = time.monotonic()

        if now - self.last_cmd_time > self.command_timeout_sec:
            x = 0.0
            y = 0.0
            wz = 0.0
        else:
            x = self.current_x
            y = self.current_y
            wz = self.current_wz

        line = f"V {x:.3f} {y:.3f} {wz:.3f}\n"

        try:
            self.ser.write(line.encode("ascii"))
        except Exception as exc:
            self.get_logger().warn(f"serial write failed: {exc}")
            self._close_serial()
            return

        if self.log_sent_command:
            self.get_logger().info(f"sent: {line.strip()}")

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
                timeout=0.0,
                write_timeout=self.serial_write_timeout_sec,
            )
        except Exception as exc:
            self.get_logger().warn(f"failed to open serial port {self.port}: {exc}")
            self.ser = None
            return

        self.get_logger().info(f"opened serial port: {self.port}")

    def _close_serial(self) -> None:
        if self.ser is None:
            return

        try:
            if self.ser.is_open:
                self.ser.close()
        except Exception:
            pass

        self.ser = None

    @staticmethod
    def _clamp_finite(value: float, low: float, high: float) -> float:
        value = float(value)

        if not math.isfinite(value):
            return 0.0

        if value < low:
            return low

        if value > high:
            return high

        return value

    def destroy_node(self) -> bool:
        try:
            if self.ser is not None and self.ser.is_open:
                self.ser.write(b"V 0.000 0.000 0.000\n")
                self.ser.close()
        except Exception:
            pass

        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)

    node = CmdVelSerialNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
