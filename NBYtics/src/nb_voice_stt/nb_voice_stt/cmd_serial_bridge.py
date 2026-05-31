import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import serial


class CmdSerialBridge(Node):
    def __init__(self):
        super().__init__("cmd_serial_bridge")

        self.declare_parameter("port", "/dev/arduino_nb")
        self.declare_parameter("baudrate", 9600)
        self.declare_parameter("input_topic", "/safe_cmd")
        self.declare_parameter("command_timeout_sec", 1.5)
        self.declare_parameter("repeat_same_command", True)

        self.port = self.get_parameter("port").value
        self.baudrate = int(self.get_parameter("baudrate").value)
        self.input_topic = self.get_parameter("input_topic").value
        self.command_timeout_sec = float(self.get_parameter("command_timeout_sec").value)
        self.repeat_same_command = bool(self.get_parameter("repeat_same_command").value)

        try:
            self.serial_conn = serial.Serial(self.port, self.baudrate, timeout=0.1)
            time.sleep(2.0)  # Arduino reset 대기
        except Exception as e:
            self.get_logger().error(f"failed to open serial port {self.port}: {e}")
            raise

        # 새 문자 명령 + 예전 문자열 명령 둘 다 허용
        self.cmd_map = {
            "W": "W",
            "A": "A",
            "S": "S",
            "D": "D",
            "X": "X",
            "Q": "Q",
            "E": "E",
            "R": "R",
            "T": "T",
            "F": "F",
            "G": "G",
            "L": "L",
            "M": "M",

            "FORWARD": "W",
            "BACKWARD": "S",
            "LEFT": "A",
            "RIGHT": "D",
            "STOP": "X",
        }

        self.oneshot_commands = {"L", "M"}

        self.last_sent = None
        self.last_cmd_time = time.time()

        self.sub = self.create_subscription(
            String,
            self.input_topic,
            self.cmd_callback,
            10,
        )

        self.timer = self.create_timer(0.1, self.timer_callback)

        self.get_logger().info(
            f"Serial bridge started: topic={self.input_topic}, port={self.port}, baudrate={self.baudrate}"
        )
        self.tx_pub = self.create_publisher(String, "/bridge/tx", 10)

    def send_char(self, ch: str):
        try:
            self.serial_conn.write(ch.encode("utf-8"))
            self.serial_conn.flush()

            msg = String()
            msg.data = ch
            self.tx_pub.publish(msg)

            self.get_logger().info(f"sent to arduino: {repr(ch)}")
        except Exception as e:
            self.get_logger().error(f"serial write failed: {e}")

    def cmd_callback(self, msg: String):
        ros_cmd = msg.data.strip().upper()

        if ros_cmd not in self.cmd_map:
            self.get_logger().warn(f"unknown cmd: {ros_cmd}")
            return

        serial_cmd = self.cmd_map[ros_cmd]
        if serial_cmd in self.oneshot_commands:
            self.send_char(serial_cmd)
            self.last_cmd_time = time.time()
            return


        if (not self.repeat_same_command) and (serial_cmd == self.last_sent):
            self.last_cmd_time = time.time()
            return

        self.send_char(serial_cmd)
        self.last_sent = serial_cmd
        self.last_cmd_time = time.time()

    def timer_callback(self):
        # 일정 시간 동안 명령이 없으면 자동 정지
        dt = time.time() - self.last_cmd_time
        if dt > self.command_timeout_sec and self.last_sent != "X":
            self.send_char("X")
            self.last_sent = "X"

    def destroy_node(self):
        try:
            if self.serial_conn and self.serial_conn.is_open:
                self.serial_conn.write(b"X")
                self.serial_conn.flush()
                self.serial_conn.close()
        except Exception:
            pass
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = CmdSerialBridge()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()