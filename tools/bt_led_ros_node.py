#!/usr/bin/env python3

import os
import errno
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32


class BluetoothLedBridge(Node):
    """
    Subscribe:
      /zeri/led/cmd : std_msgs/msg/Int32

    Valid values:
      0 = OFF
      1 = RED
      2 = GREEN
      3 = BLUE
      4 = YELLOW
      5 = MAGENTA
      6 = CYAN
      7 = WHITE

    Output:
      writes one ASCII digit to /dev/rfcomm0
    """

    def __init__(self):
        super().__init__("bt_led_bridge")

        self.declare_parameter("port", "/dev/rfcomm0")
        self.declare_parameter("topic", "/zeri/led/cmd")
        self.declare_parameter("reconnect_period_sec", 1.0)

        self.port = self.get_parameter("port").value
        self.topic = self.get_parameter("topic").value
        self.reconnect_period_sec = float(
            self.get_parameter("reconnect_period_sec").value
        )

        self.fd = None
        self.last_cmd = None

        self.color_names = {
            0: "OFF",
            1: "RED",
            2: "GREEN",
            3: "BLUE",
            4: "YELLOW",
            5: "MAGENTA",
            6: "CYAN",
            7: "WHITE",
        }

        self.sub = self.create_subscription(
            Int32,
            self.topic,
            self.on_led_cmd,
            10,
        )

        self.timer = self.create_timer(
            self.reconnect_period_sec,
            self.ensure_port_open,
        )

        self.get_logger().info(f"BT LED bridge started")
        self.get_logger().info(f"Subscribing topic: {self.topic}")
        self.get_logger().info(f"Bluetooth port: {self.port}")
        self.get_logger().info("Command map: 0=OFF, 1=RED, 2=GREEN, 3=BLUE, 4=YELLOW, 5=MAGENTA, 6=CYAN, 7=WHITE")

    def ensure_port_open(self):
        if self.fd is not None:
            return

        try:
            self.fd = os.open(
                self.port,
                os.O_WRONLY | os.O_NOCTTY | os.O_NONBLOCK,
            )
            self.get_logger().info(f"Opened Bluetooth port: {self.port}")

        except FileNotFoundError:
            self.get_logger().warn(
                f"{self.port} not found. Run rfcomm connect first.",
                throttle_duration_sec=3.0,
            )

        except PermissionError:
            self.get_logger().warn(
                f"Permission denied: {self.port}. Try: sudo chmod 666 {self.port}",
                throttle_duration_sec=3.0,
            )

        except OSError as e:
            self.get_logger().warn(
                f"Failed to open {self.port}: {e}",
                throttle_duration_sec=3.0,
            )

    def close_port(self):
        if self.fd is not None:
            try:
                os.close(self.fd)
            except OSError:
                pass
            self.fd = None

    def on_led_cmd(self, msg: Int32):
        value = int(msg.data)

        if value not in self.color_names:
            self.get_logger().warn(f"Invalid LED command: {value}. Valid range is 0~7.")
            return

        self.send_led_value(value)

    def send_led_value(self, value: int):
        self.ensure_port_open()

        if self.fd is None:
            self.get_logger().error(f"Cannot send LED command {value}: Bluetooth port is not open.")
            return

        payload = str(value).encode("ascii")

        try:
            os.write(self.fd, payload)
            self.last_cmd = value
            self.get_logger().info(f"Sent LED command: {value} = {self.color_names[value]}")

        except BlockingIOError:
            self.get_logger().warn("Bluetooth port is busy. Try again.")

        except OSError as e:
            self.get_logger().error(f"Bluetooth write failed: {e}. Closing port.")
            self.close_port()

    def destroy_node(self):
        self.close_port()
        super().destroy_node()


def main():
    rclpy.init()
    node = BluetoothLedBridge()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()