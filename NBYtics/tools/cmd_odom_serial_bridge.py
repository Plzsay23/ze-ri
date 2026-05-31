#!/usr/bin/env python3
import math
import re
import threading
import time

import serial

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster


def yaw_to_quat(yaw: float):
    return 0.0, 0.0, math.sin(yaw * 0.5), math.cos(yaw * 0.5)


class CmdOdomSerialBridge(Node):
    def __init__(self):
        super().__init__("cmd_odom_serial_bridge")

        self.declare_parameter("port", "/dev/arduino")
        self.declare_parameter("baudrate", 115200)
        self.declare_parameter("input_topic", "/safe_cmd")
        self.declare_parameter("cmd_period_sec", 0.08)
        self.declare_parameter("command_timeout_sec", 0.8)
        self.declare_parameter("append_newline", False)

        self.declare_parameter("ticks_per_rev", 3464.0)
        self.declare_parameter("wheel_radius", 0.075)
        self.declare_parameter("lx", 0.1575)
        self.declare_parameter("ly", 0.2125)
        self.declare_parameter("odom_frame", "odom")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("publish_tf", True)

        # RViz에서 좌우/회전이 반전된 상태 보정용
        self.declare_parameter("invert_vy", True)
        self.declare_parameter("invert_wz", True)

        self.port = str(self.get_parameter("port").value)
        self.baudrate = int(self.get_parameter("baudrate").value)
        self.input_topic = str(self.get_parameter("input_topic").value)
        self.cmd_period_sec = float(self.get_parameter("cmd_period_sec").value)
        self.command_timeout_sec = float(self.get_parameter("command_timeout_sec").value)
        self.append_newline = bool(self.get_parameter("append_newline").value)

        self.ticks_per_rev = float(self.get_parameter("ticks_per_rev").value)
        self.wheel_radius = float(self.get_parameter("wheel_radius").value)
        self.lx = float(self.get_parameter("lx").value)
        self.ly = float(self.get_parameter("ly").value)
        self.odom_frame = str(self.get_parameter("odom_frame").value)
        self.base_frame = str(self.get_parameter("base_frame").value)
        self.publish_tf = bool(self.get_parameter("publish_tf").value)
        self.invert_vy = bool(self.get_parameter("invert_vy").value)
        self.invert_wz = bool(self.get_parameter("invert_wz").value)

        self.cmd_map = {
            "W": "W", "A": "A", "S": "S", "D": "D", "X": "X",
            "Q": "Q", "E": "E", "R": "R", "T": "T", "F": "F", "G": "G", "L": "L", "M": "M",
            "FORWARD": "W", "BACKWARD": "S",
            "LEFT": "A", "RIGHT": "D", "STOP": "X",
        }

        self.current_cmd = "X"
        self.last_input_time = time.time()
        self.last_logged_cmd = None
        self.lock = threading.Lock()
        self.serial_lock = threading.Lock()
        self.oneshot_commands = {"L", "M"}

        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.prev_ticks = None
        self.prev_time = None
        self.pattern = re.compile(r'^ENC\s+(-?\d+)\s+(-?\d+)\s+(-?\d+)\s+(-?\d+)$')

        self.odom_pub = self.create_publisher(Odometry, "/odom", 20)
        self.tf_broadcaster = TransformBroadcaster(self)
        self.tx_pub = self.create_publisher(String, "/bridge/tx", 10)

        self.sub = self.create_subscription(String, self.input_topic, self.cmd_callback, 10)

        self.get_logger().info(f"Opening serial: {self.port} @ {self.baudrate}")
        self.ser = serial.Serial(self.port, self.baudrate, timeout=0.05)
        time.sleep(1.5)
        self.ser.reset_input_buffer()

        self.reader_thread = threading.Thread(target=self.read_loop, daemon=True)
        self.reader_thread.start()

        self.cmd_timer = self.create_timer(self.cmd_period_sec, self.write_current_cmd)

        self.get_logger().info(
            f"cmd_odom_serial_bridge started | input={self.input_topic}, "
            f"cmd_period={self.cmd_period_sec}, timeout={self.command_timeout_sec}, "
            f"invert_vy={self.invert_vy}, invert_wz={self.invert_wz}"
        )

    def send_serial_cmd(self, cmd: str):
        try:
            payload = cmd + ("\n" if self.append_newline else "")
            with self.serial_lock:
                self.ser.write(payload.encode("utf-8"))
                self.ser.flush()

            tx = String()
            tx.data = cmd
            self.tx_pub.publish(tx)

            if cmd != self.last_logged_cmd:
                self.get_logger().info(f"serial cmd -> {cmd}")
                self.last_logged_cmd = cmd

        except Exception as e:
            self.get_logger().error(f"serial write failed: {e}")


    def cmd_callback(self, msg: String):
        key = msg.data.strip().upper()
        if key not in self.cmd_map:
            self.get_logger().warn(f"unknown cmd: {msg.data}")
            return

        serial_cmd = self.cmd_map[key]
        if serial_cmd in self.oneshot_commands:
            self.send_serial_cmd(serial_cmd)
            self.last_input_time = time.time()
            return

        with self.lock:
            self.current_cmd = self.cmd_map[key]
            self.last_input_time = time.time()

    def write_current_cmd(self):
        with self.lock:
            if time.time() - self.last_input_time > self.command_timeout_sec:
                cmd = "X"
                self.current_cmd = "X"
            else:
                cmd = self.current_cmd

        try:
            payload = cmd + ("\n" if self.append_newline else "")
            self.ser.write(payload.encode("utf-8"))
            self.ser.flush()

            tx = String()
            tx.data = cmd
            self.tx_pub.publish(tx)

            if cmd != self.last_logged_cmd:
                self.get_logger().info(f"serial cmd -> {cmd}")
                self.last_logged_cmd = cmd

        except Exception as e:
            self.get_logger().error(f"serial write failed: {e}")

    def read_loop(self):
        while rclpy.ok():
            try:
                line = self.ser.readline().decode(errors="ignore").strip()
                if not line:
                    continue

                m = self.pattern.match(line)
                if m:
                    ticks = tuple(int(m.group(i)) for i in range(1, 5))
                    self.process_ticks(ticks)

            except Exception as e:
                self.get_logger().warn(f"serial read error: {e}")
                time.sleep(0.1)

    def process_ticks(self, ticks):
        now_msg = self.get_clock().now()
        now = now_msg.nanoseconds * 1e-9

        if self.prev_ticks is None:
            self.prev_ticks = ticks
            self.prev_time = now
            return

        dt = now - self.prev_time
        if dt <= 0.0 or dt > 1.0:
            self.prev_ticks = ticks
            self.prev_time = now
            return

        d_lf = ticks[0] - self.prev_ticks[0]
        d_rf = ticks[1] - self.prev_ticks[1]
        d_lr = ticks[2] - self.prev_ticks[2]
        d_rr = ticks[3] - self.prev_ticks[3]

        self.prev_ticks = ticks
        self.prev_time = now

        meter_per_tick = (2.0 * math.pi * self.wheel_radius) / self.ticks_per_rev

        v_lf = d_lf * meter_per_tick / dt
        v_rf = d_rf * meter_per_tick / dt
        v_lr = d_lr * meter_per_tick / dt
        v_rr = d_rr * meter_per_tick / dt

        vx = (v_lf + v_rf + v_lr + v_rr) / 4.0
        vy = (-v_lf + v_rf + v_lr - v_rr) / 4.0
        wz = (-v_lf + v_rf - v_lr + v_rr) / (4.0 * (self.lx + self.ly))

        if self.invert_vy:
            vy = -vy
        if self.invert_wz:
            wz = -wz

        cos_yaw = math.cos(self.yaw)
        sin_yaw = math.sin(self.yaw)

        self.x += (vx * cos_yaw - vy * sin_yaw) * dt
        self.y += (vx * sin_yaw + vy * cos_yaw) * dt
        self.yaw += wz * dt
        self.yaw = math.atan2(math.sin(self.yaw), math.cos(self.yaw))

        self.publish_odom(now_msg, vx, vy, wz)

    def publish_odom(self, stamp, vx, vy, wz):
        qx, qy, qz, qw = yaw_to_quat(self.yaw)

        odom = Odometry()
        odom.header.stamp = stamp.to_msg()
        odom.header.frame_id = self.odom_frame
        odom.child_frame_id = self.base_frame

        odom.pose.pose.position.x = self.x
        odom.pose.pose.position.y = self.y
        odom.pose.pose.position.z = 0.0
        odom.pose.pose.orientation.x = qx
        odom.pose.pose.orientation.y = qy
        odom.pose.pose.orientation.z = qz
        odom.pose.pose.orientation.w = qw

        odom.twist.twist.linear.x = vx
        odom.twist.twist.linear.y = vy
        odom.twist.twist.angular.z = wz

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
            tf.header.frame_id = self.odom_frame
            tf.child_frame_id = self.base_frame
            tf.transform.translation.x = self.x
            tf.transform.translation.y = self.y
            tf.transform.translation.z = 0.0
            tf.transform.rotation.x = qx
            tf.transform.rotation.y = qy
            tf.transform.rotation.z = qz
            tf.transform.rotation.w = qw
            self.tf_broadcaster.sendTransform(tf)

    def destroy_node(self):
        try:
            self.ser.write(b"X")
            self.ser.flush()
            self.ser.close()
        except Exception:
            pass
        return super().destroy_node()


def main():
    rclpy.init()
    node = CmdOdomSerialBridge()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
