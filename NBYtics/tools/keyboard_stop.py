#!/usr/bin/env python3
import sys
import select
import termios
import tty

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class KeyboardStopNode(Node):
    def __init__(self):
        super().__init__("keyboard_stop_node")
        self.pub = self.create_publisher(String, "/voice_interrupt", 10)
        self.get_logger().info("Keyboard stop node started")
        self.get_logger().info("Press x to publish stop, q to quit")

    def publish_stop(self):
        msg = String()
        msg.data = "stop"
        self.pub.publish(msg)
        self.get_logger().warn("Keyboard interrupt -> /voice_interrupt = stop")


def main():
    rclpy.init()
    node = KeyboardStopNode()

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)

    try:
        tty.setraw(fd)

        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.01)

            dr, _, _ = select.select([sys.stdin], [], [], 0.05)
            if not dr:
                continue

            ch = sys.stdin.read(1)
            if ch in ("x", "X"):
                node.publish_stop()
            elif ch in ("q", "Q"):
                node.get_logger().info("Quit keyboard stop node")
                break

    except KeyboardInterrupt:
        pass
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
