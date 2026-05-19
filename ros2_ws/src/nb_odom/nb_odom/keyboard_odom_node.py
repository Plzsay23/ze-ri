#!/usr/bin/env python3
import math
import re
import sys
import termios
import threading
import time
import tty
import select

import serial

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster


def yaw_to_quat(yaw: float):
    return 0.0, 0.0, math.sin(yaw * 0.5), math.cos(yaw * 0.5)


HELP_TEXT = """
Keyboard control:
  w : forward
  s : backward
  a : turn left
  d : turn right
  q : strafe left
  e : strafe right
  x : stop

  l : pwm up
  m : pwm down

  Ctrl-C : quit

주의:
- 한 번 누르면 계속 명령을 반복 전송함
- 멈추려면 x 누르기
"""


class KeyboardOdomNode(Node):
    def __init__(self):
        super().__init__('keyboard_odom_node')

        self.declare_parameter('port', '/dev/ttyUSB1')
        self.declare_parameter('baudrate', 115200)
        self.declare_parameter('ticks_per_rev', 3464.0)
        self.declare_parameter('wheel_radius', 0.075)
        self.declare_parameter('lx', 0.1575)
        self.declare_parameter('ly', 0.2125)
        self.declare_parameter('odom_frame', 'odom')
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('publish_tf', True)
        self.declare_parameter('cmd_period', 0.10)

        self.port = self.get_parameter('port').value
        self.baudrate = int(self.get_parameter('baudrate').value)
        self.ticks_per_rev = float(self.get_parameter('ticks_per_rev').value)
        self.wheel_radius = float(self.get_parameter('wheel_radius').value)
        self.lx = float(self.get_parameter('lx').value)
        self.ly = float(self.get_parameter('ly').value)
        self.odom_frame = self.get_parameter('odom_frame').value
        self.base_frame = self.get_parameter('base_frame').value
        self.publish_tf = bool(self.get_parameter('publish_tf').value)
        self.cmd_period = float(self.get_parameter('cmd_period').value)

        self.odom_pub = self.create_publisher(Odometry, '/odom', 20)
        self.tf_broadcaster = TransformBroadcaster(self)

        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0

        self.prev_ticks = None
        self.prev_time = None

        self.pattern = re.compile(r'^ENC\s+(-?\d+)\s+(-?\d+)\s+(-?\d+)\s+(-?\d+)$')

        self.current_cmd = 'x'
        self.running = True

        self.get_logger().info(f'Opening serial: {self.port} @ {self.baudrate}')
        self.ser = serial.Serial(self.port, self.baudrate, timeout=0.2)
        time.sleep(1.0)
        self.ser.reset_input_buffer()

        self.get_logger().info(HELP_TEXT)

        self.reader_thread = threading.Thread(target=self.read_loop, daemon=True)
        self.writer_thread = threading.Thread(target=self.write_loop, daemon=True)
        self.keyboard_thread = threading.Thread(target=self.keyboard_loop, daemon=True)

        self.reader_thread.start()
        self.writer_thread.start()
        self.keyboard_thread.start()

    def keyboard_loop(self):
        old_settings = termios.tcgetattr(sys.stdin)

        try:
            tty.setcbreak(sys.stdin.fileno())

            while self.running and rclpy.ok():
                rlist, _, _ = select.select([sys.stdin], [], [], 0.1)

                if not rlist:
                    continue

                key = sys.stdin.read(1)

                if key == '\x03':
                    self.running = False
                    break

                if key in ['w', 's', 'a', 'd', 'q', 'e', 'x', 'l', 'm',
                           'W', 'S', 'A', 'D', 'Q', 'E', 'X', 'L', 'M']:
                    self.current_cmd = key.lower()
                    self.get_logger().info(f'KEY: {self.current_cmd}')

                    # l/m은 속도 변경용이라 한 번만 보내고 이동 명령은 유지하지 않음
                    if self.current_cmd in ['l', 'm']:
                        try:
                            self.ser.write((self.current_cmd + '\n').encode())
                            self.ser.flush()
                        except Exception as e:
                            self.get_logger().warn(f'Serial write error: {e}')
                        self.current_cmd = 'x'

        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

    def write_loop(self):
        while self.running and rclpy.ok():
            try:
                cmd = self.current_cmd

                if cmd in ['w', 's', 'a', 'd', 'q', 'e', 'x']:
                    self.ser.write((cmd + '\n').encode())
                    self.ser.flush()

            except Exception as e:
                self.get_logger().warn(f'Serial write error: {e}')

            time.sleep(self.cmd_period)

    def read_loop(self):
        while self.running and rclpy.ok():
            try:
                line = self.ser.readline().decode(errors='ignore').strip()

                if not line:
                    continue

                match = self.pattern.match(line)

                if match:
                    ticks = tuple(int(match.group(i)) for i in range(1, 5))
                    self.process_ticks(ticks)
                else:
                    # READY, ACK 같은 메시지는 로그로만 확인
                    if not line.startswith('ENC'):
                        self.get_logger().info(f'Arduino: {line}')

            except Exception as e:
                self.get_logger().warn(f'Serial read error: {e}')
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

        # ENC 순서: LF RF LR RR
        vx = (v_lf + v_rf + v_lr + v_rr) / 4.0
        vy = (-v_lf + v_rf + v_lr - v_rr) / 4.0
        wz = (-v_lf + v_rf - v_lr + v_rr) / (4.0 * (self.lx + self.ly))

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

    def shutdown(self):
        self.running = False

        try:
            self.ser.write(b'x\n')
            self.ser.flush()
            time.sleep(0.1)
            self.ser.close()
        except Exception:
            pass


def main(args=None):
    rclpy.init(args=args)
    node = KeyboardOdomNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
