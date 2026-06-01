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


class CmdVelKeySerialNode(Node):
    """
    geometry_msgs/Twist를 기존 Arduino 키보드 명령으로 변환해서 보낸다.

    Arduino test.ino 프로토콜:
        w : forward
        s : backward
        a : turn left
        d : turn right
        q : strafe left
        e : strafe right
        x : stop

    입력:
        /cmd_vel

    출력:
        serial char command to Arduino
    """

    def __init__(self) -> None:
        super().__init__("cmd_vel_key_serial_node")

        self.declare_parameter("port", "/dev/arduino")
        self.declare_parameter("baudrate", 115200)
        self.declare_parameter("cmd_topic", "/cmd_vel")

        self.declare_parameter("send_rate_hz", 20.0)
        self.declare_parameter("command_timeout_sec", 0.5)

        self.declare_parameter("linear_threshold", 0.03)
        self.declare_parameter("angular_threshold", 0.10)

        self.declare_parameter("open_retry_sec", 1.0)
        self.declare_parameter("serial_write_timeout_sec", 0.05)
        self.declare_parameter("arduino_reset_wait_sec", 2.0)

        self.declare_parameter("send_newline", False)
        self.declare_parameter("log_sent_command", True)

        self.port = str(self.get_parameter("port").value)
        self.baudrate = int(self.get_parameter("baudrate").value)
        self.cmd_topic = str(self.get_parameter("cmd_topic").value)

        self.send_rate_hz = float(self.get_parameter("send_rate_hz").value)
        self.command_timeout_sec = float(self.get_parameter("command_timeout_sec").value)

        self.linear_threshold = float(self.get_parameter("linear_threshold").value)
        self.angular_threshold = float(self.get_parameter("angular_threshold").value)

        self.open_retry_sec = float(self.get_parameter("open_retry_sec").value)
        self.serial_write_timeout_sec = float(
            self.get_parameter("serial_write_timeout_sec").value
        )
        self.arduino_reset_wait_sec = float(
            self.get_parameter("arduino_reset_wait_sec").value
        )

        self.send_newline = bool(self.get_parameter("send_newline").value)
        self.log_sent_command = bool(self.get_parameter("log_sent_command").value)

        if serial is None:
            raise RuntimeError(
                f"pyserial import failed: {SERIAL_IMPORT_ERROR}. "
                "Install with: sudo apt install python3-serial"
            )

        self.ser: Optional["serial.Serial"] = None
        self.last_open_attempt = 0.0
        self.opened_time = 0.0

        self.last_cmd_time = 0.0
        self.latest_cmd = Twist()
        self.last_sent_key = None

        self.sub = self.create_subscription(
            Twist,
            self.cmd_topic,
            self.on_cmd,
            10,
        )

        period = 1.0 / max(self.send_rate_hz, 1.0)
        self.timer = self.create_timer(period, self.on_timer)

        self.get_logger().info("cmd_vel_key_serial_node started")
        self.get_logger().info(f"  port={self.port}")
        self.get_logger().info(f"  baudrate={self.baudrate}")
        self.get_logger().info(f"  cmd_topic={self.cmd_topic}")
        self.get_logger().info("  protocol=w/s/a/d/q/e/x")

    def on_cmd(self, msg: Twist) -> None:
        self.latest_cmd = msg
        self.last_cmd_time = time.monotonic()

    def on_timer(self) -> None:
        self._ensure_serial_open()

        if self.ser is None or not self.ser.is_open:
            return

        now = time.monotonic()

        # Arduino는 serial open 때 reset될 수 있으므로 처음 몇 초 대기
        if now - self.opened_time < self.arduino_reset_wait_sec:
            return

        if now - self.last_cmd_time > self.command_timeout_sec:
            key = "x"
        else:
            key = self._twist_to_key(self.latest_cmd)

        payload = key.encode("ascii")
        if self.send_newline:
            payload += b"\n"

        try:
            self.ser.write(payload)
            self.ser.flush()
        except Exception as exc:
            self.get_logger().warn(f"serial write failed: {exc}")
            self._close_serial()
            return

        if self.log_sent_command and key != self.last_sent_key:
            self.get_logger().info(f"sent key: {key}")
            self.last_sent_key = key

    def _twist_to_key(self, msg: Twist) -> str:
        x = self._finite(msg.linear.x)
        y = self._finite(msg.linear.y)
        wz = self._finite(msg.angular.z)

        # 가장 큰 축 하나만 기존 키 명령으로 변환
        ax = abs(x)
        ay = abs(y)
        aw = abs(wz)

        if ax < self.linear_threshold and ay < self.linear_threshold and aw < self.angular_threshold:
            return "x"

        # 회전이 가장 크면 a/d
        if aw >= ax and aw >= ay:
            if wz > 0.0:
                return "a"   # 좌회전
            else:
                return "d"   # 우회전

        # 평행이동이 가장 크면 q/e
        if ay >= ax and ay >= aw:
            if y > 0.0:
                return "q"   # 좌측 평행이동
            else:
                return "e"   # 우측 평행이동

        # 전후진
        if x > 0.0:
            return "w"
        else:
            return "s"

    @staticmethod
    def _finite(value: float) -> float:
        value = float(value)
        if not math.isfinite(value):
            return 0.0
        return value

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

    node = CmdVelKeySerialNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
