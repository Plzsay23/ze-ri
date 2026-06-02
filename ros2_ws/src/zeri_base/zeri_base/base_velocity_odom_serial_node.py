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
except ImportError:
    serial = None


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def yaw_to_quat(yaw):
    half = yaw * 0.5
    return 0.0, 0.0, math.sin(half), math.cos(half)


class BaseVelocityOdomSerialNode(Node):
    """
    /cmd_vel -> Arduino velocity line protocol + encoder odom.

    Serial protocol to Arduino:
      V <vx_norm> <vy_norm> <wz_norm>\n
    where each value is in [-1, 1].
    Arduino performs mecanum wheel mixing, so forward+turn is simultaneous.
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
        self.declare_parameter("send_hz", 20.0)

        self.declare_parameter("max_linear_x", 0.25)
        self.declare_parameter("max_linear_y", 0.25)
        self.declare_parameter("max_angular_z", 0.70)

        self.declare_parameter("linear_deadband", 0.02)
        self.declare_parameter("angular_deadband", 0.04)
        self.declare_parameter("enable_strafe", False)

        self.declare_parameter("invert_vx", False)
        self.declare_parameter("invert_vy", False)
        self.declare_parameter("invert_wz", False)

        self.declare_parameter("arduino_pwm", 60)
        self.declare_parameter("set_pwm_on_start", True)

        self.declare_parameter("log_sent_command", True)
        self.declare_parameter("log_all_commands", False)
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
        self.send_hz = float(self.get_parameter("send_hz").value)

        self.max_linear_x = float(self.get_parameter("max_linear_x").value)
        self.max_linear_y = float(self.get_parameter("max_linear_y").value)
        self.max_angular_z = float(self.get_parameter("max_angular_z").value)

        self.linear_deadband = float(self.get_parameter("linear_deadband").value)
        self.angular_deadband = float(self.get_parameter("angular_deadband").value)
        self.enable_strafe = bool(self.get_parameter("enable_strafe").value)

        self.invert_vx = bool(self.get_parameter("invert_vx").value)
        self.invert_vy = bool(self.get_parameter("invert_vy").value)
        self.invert_wz = bool(self.get_parameter("invert_wz").value)

        self.arduino_pwm = int(self.get_parameter("arduino_pwm").value)
        self.set_pwm_on_start = bool(self.get_parameter("set_pwm_on_start").value)

        self.log_sent_command = bool(self.get_parameter("log_sent_command").value)
        self.log_all_commands = bool(self.get_parameter("log_all_commands").value)
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
        self.last_line = ""
        self.last_zero_sent_time = 0.0

        self.prev_enc = None

        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.last_odom_stamp = self.get_clock().now()

        self.cmd_sub = self.create_subscription(Twist, self.cmd_topic, self.on_cmd, 10)
        self.odom_pub = self.create_publisher(Odometry, self.odom_topic, 10)
        self.tf_broadcaster = TransformBroadcaster(self)

        if self.set_pwm_on_start:
            self.write_line(f"P {self.arduino_pwm}\n", force_log=True)

        self.read_timer = self.create_timer(0.01, self.read_serial_once)
        self.send_timer = self.create_timer(1.0 / max(self.send_hz, 1.0), self.send_velocity_once)

        self.get_logger().info(
            "base velocity odom serial node started: "
            f"port={self.port}, baudrate={self.baudrate}, cmd={self.cmd_topic}, "
            f"send_hz={self.send_hz}, max_x={self.max_linear_x}, max_wz={self.max_angular_z}, "
            f"pwm={self.arduino_pwm}"
        )

    def now_sec(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def on_cmd(self, msg):
        self.last_cmd = msg
        self.last_cmd_time = self.now_sec()

    def write_line(self, line, force_log=False):
        try:
            self.ser.write(line.encode("ascii"))
            self.ser.flush()
        except Exception as e:
            self.get_logger().error(f"serial write failed: {repr(e)}")
            return

        clean = line.strip()
        if self.log_sent_command or force_log:
            if self.log_all_commands or force_log or clean != self.last_line:
                self.get_logger().info(f"sent: {clean}")

        self.last_line = clean

    def normalized_cmd(self, cmd):
        vx = float(cmd.linear.x)
        vy = float(cmd.linear.y)
        wz = float(cmd.angular.z)

        if abs(vx) < self.linear_deadband:
            vx = 0.0
        if abs(vy) < self.linear_deadband:
            vy = 0.0
        if abs(wz) < self.angular_deadband:
            wz = 0.0

        if not self.enable_strafe:
            vy = 0.0

        nx = vx / max(abs(self.max_linear_x), 1e-6)
        ny = vy / max(abs(self.max_linear_y), 1e-6)
        nw = wz / max(abs(self.max_angular_z), 1e-6)

        if self.invert_vx:
            nx = -nx
        if self.invert_vy:
            ny = -ny
        if self.invert_wz:
            nw = -nw

        nx = clamp(nx, -1.0, 1.0)
        ny = clamp(ny, -1.0, 1.0)
        nw = clamp(nw, -1.0, 1.0)

        return nx, ny, nw

    def send_velocity_once(self):
        now = self.now_sec()
        cmd_fresh = (now - self.last_cmd_time) <= self.cmd_timeout_sec

        if not cmd_fresh:
            if now - self.last_zero_sent_time > 0.20:
                self.write_line("V 0.000 0.000 0.000\n")
                self.last_zero_sent_time = now
            return

        nx, ny, nw = self.normalized_cmd(self.last_cmd)
        self.write_line(f"V {nx:.3f} {ny:.3f} {nw:.3f}\n")

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

        dx_body = (fl + fr + rl + rr_m) * 0.25
        dy_body = (-fl + fr + rl - rr_m) * 0.25

        base_sum = max(self.lx + self.ly, 1e-6)
        dyaw = (-fl + fr - rl + rr_m) / (4.0 * base_sum)

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

        qx, qy, qz, qw = yaw_to_quat(self.yaw)

        odom = Odometry()
        odom.header.stamp = stamp.to_msg()
        odom.header.frame_id = self.odom_frame_id
        odom.child_frame_id = self.base_frame_id

        odom.pose.pose.position.x = self.x
        odom.pose.pose.position.y = self.y
        odom.pose.pose.position.z = 0.0
        odom.pose.pose.orientation.x = qx
        odom.pose.pose.orientation.y = qy
        odom.pose.pose.orientation.z = qz
        odom.pose.pose.orientation.w = qw

        odom.twist.twist.linear.x = dx_body / dt
        odom.twist.twist.linear.y = dy_body / dt
        odom.twist.twist.angular.z = dyaw / dt

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
            tf.transform.rotation.x = qx
            tf.transform.rotation.y = qy
            tf.transform.rotation.z = qz
            tf.transform.rotation.w = qw
            self.tf_broadcaster.sendTransform(tf)

    def destroy_node(self):
        try:
            self.write_line("V 0.000 0.000 0.000\n", force_log=True)
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
    node = BaseVelocityOdomSerialNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
