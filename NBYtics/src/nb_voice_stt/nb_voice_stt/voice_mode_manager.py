import time
from typing import Optional

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class VoiceModeManager(Node):
    def __init__(self):
        super().__init__("voice_mode_manager")

        self.declare_parameter("voice_cmd_topic", "/voice_cmd")
        self.declare_parameter("voice_interrupt_topic", "/voice_interrupt")
        self.declare_parameter("auto_cmd_topic", "/auto_cmd")
        self.declare_parameter("mode_select_topic", "/mode_select")

        self.declare_parameter("output_cmd_topic", "/mode_cmd")
        self.declare_parameter("state_topic", "/mode_state")

        self.declare_parameter("publish_rate_hz", 5.0)
        self.declare_parameter("default_mode", "MANUAL")
        self.declare_parameter("allow_auto_mode", True)

        self.voice_cmd_topic = str(self.get_parameter("voice_cmd_topic").value)
        self.voice_interrupt_topic = str(self.get_parameter("voice_interrupt_topic").value)
        self.auto_cmd_topic = str(self.get_parameter("auto_cmd_topic").value)
        self.mode_select_topic = str(self.get_parameter("mode_select_topic").value)
        self.output_cmd_topic = str(self.get_parameter("output_cmd_topic").value)
        self.state_topic = str(self.get_parameter("state_topic").value)

        self.publish_rate_hz = float(self.get_parameter("publish_rate_hz").value)
        self.default_mode = str(self.get_parameter("default_mode").value).strip().upper()
        self.allow_auto_mode = bool(self.get_parameter("allow_auto_mode").value)

        if self.publish_rate_hz <= 0.0:
            self.publish_rate_hz = 5.0

        if self.default_mode not in ["MANUAL", "AUTO"]:
            self.default_mode = "MANUAL"

        # persistent motion commands
        self.valid_motion_cmds = {
            "W", "A", "S", "D", "X",
            "Q", "E", "R", "T", "F", "G",
        }

        # one-shot commands
        self.valid_oneshot_cmds = {"L", "M"}

        self.mode = self.default_mode
        self.manual_cmd = "X"
        self.auto_cmd = "X"

        self.last_output_cmd = None
        self.last_state_text = None
        self.last_interrupt_time = 0.0

        self.cmd_pub = self.create_publisher(String, self.output_cmd_topic, 10)
        self.state_pub = self.create_publisher(String, self.state_topic, 10)

        self.voice_cmd_sub = self.create_subscription(
            String, self.voice_cmd_topic, self.voice_cmd_callback, 10
        )
        self.voice_interrupt_sub = self.create_subscription(
            String, self.voice_interrupt_topic, self.voice_interrupt_callback, 10
        )
        self.auto_cmd_sub = self.create_subscription(
            String, self.auto_cmd_topic, self.auto_cmd_callback, 10
        )
        self.mode_select_sub = self.create_subscription(
            String, self.mode_select_topic, self.mode_select_callback, 10
        )

        self.timer = self.create_timer(1.0 / self.publish_rate_hz, self.timer_callback)

        self.get_logger().info("VoiceModeManager started")
        self.get_logger().info(f"voice_cmd_topic={self.voice_cmd_topic}")
        self.get_logger().info(f"voice_interrupt_topic={self.voice_interrupt_topic}")
        self.get_logger().info(f"auto_cmd_topic={self.auto_cmd_topic}")
        self.get_logger().info(f"mode_select_topic={self.mode_select_topic}")
        self.get_logger().info(f"output_cmd_topic={self.output_cmd_topic}")
        self.get_logger().info(f"default_mode={self.mode}")
        self.publish_state(force=True)

    def normalize_motion_command(self, text: str) -> Optional[str]:
        s = text.strip().upper()
        if not s:
            return None

        # direct char commands
        if s in self.valid_motion_cmds or s in self.valid_oneshot_cmds:
            return s

        # compatibility with word commands
        mapping = {
            "FORWARD": "W",
            "BACKWARD": "S",
            "LEFT": "A",
            "RIGHT": "D",
            "STOP": "X",
        }
        if s in mapping:
            return mapping[s]

        return None

    def normalize_mode(self, text: str) -> Optional[str]:
        s = text.strip().upper()
        if s in ["MANUAL", "AUTO", "STOP"]:
            return s
        return None

    def normalize_interrupt(self, text: str) -> Optional[str]:
        s = text.strip().lower()
        if s in ["stop", "x", "interrupt_stop"]:
            return "STOP"
        return None

    def get_active_cmd(self) -> str:
        if self.mode == "AUTO":
            return self.auto_cmd
        return self.manual_cmd

    def publish_cmd(self, cmd: str):
        msg = String()
        msg.data = cmd
        self.cmd_pub.publish(msg)
        self.last_output_cmd = cmd

    def publish_state(self, force: bool = False):
        state_text = (
            f"mode={self.mode}, "
            f"manual_cmd={self.manual_cmd}, "
            f"auto_cmd={self.auto_cmd}, "
            f"active_cmd={self.get_active_cmd()}"
        )
        if force or state_text != self.last_state_text:
            msg = String()
            msg.data = state_text
            self.state_pub.publish(msg)
            self.last_state_text = state_text
            self.get_logger().info(state_text)

    def set_manual_cmd(self, cmd: str):
        self.manual_cmd = cmd
        self.publish_state(force=True)

    def set_auto_cmd(self, cmd: str):
        self.auto_cmd = cmd
        self.publish_state(force=True)

    def force_stop(self, reason: str = ""):
        self.manual_cmd = "X"
        self.auto_cmd = "X"
        self.publish_cmd("X")
        self.publish_state(force=True)
        if reason:
            self.get_logger().warn(f"force_stop: {reason}")
        else:
            self.get_logger().warn("force_stop")

    def voice_cmd_callback(self, msg: String):
        cmd = self.normalize_motion_command(msg.data)
        if cmd is None:
            self.get_logger().warn(f"unknown voice command: {msg.data}")
            return

        # one-shot speed commands
        if cmd in self.valid_oneshot_cmds:
            if self.mode != "MANUAL":
                self.get_logger().info(f"ignore oneshot cmd in non-manual mode: {cmd}")
                return

            self.publish_cmd(cmd)
            self.get_logger().info(f"oneshot voice cmd: {cmd}")
            return

        # persistent motion commands
        self.set_manual_cmd(cmd)
        self.get_logger().info(f"manual cmd updated: {cmd}")

        if self.mode == "MANUAL":
            self.publish_cmd(cmd)

    def voice_interrupt_callback(self, msg: String):
        it = self.normalize_interrupt(msg.data)
        if it != "STOP":
            self.get_logger().warn(f"unknown interrupt command: {msg.data}")
            return

        self.last_interrupt_time = time.monotonic()
        self.force_stop(reason="voice interrupt stop")

    def auto_cmd_callback(self, msg: String):
        cmd = self.normalize_motion_command(msg.data)
        if cmd is None:
            self.get_logger().warn(f"unknown auto command: {msg.data}")
            return

        if cmd in self.valid_oneshot_cmds:
            self.get_logger().info(f"ignore auto oneshot cmd: {cmd}")
            return

        self.set_auto_cmd(cmd)
        self.get_logger().info(f"auto cmd updated: {cmd}")

        if self.mode == "AUTO":
            self.publish_cmd(cmd)

    def mode_select_callback(self, msg: String):
        mode = self.normalize_mode(msg.data)
        if mode is None:
            self.get_logger().warn(f"unknown mode select: {msg.data}")
            return

        if mode == "AUTO" and not self.allow_auto_mode:
            self.get_logger().warn("AUTO mode requested but allow_auto_mode is False")
            return

        if mode == "STOP":
            self.mode = "MANUAL"
            self.force_stop(reason="mode select stop")
            return

        if mode != self.mode:
            self.mode = mode
            self.get_logger().info(f"mode changed to: {self.mode}")

        self.publish_cmd(self.get_active_cmd())
        self.publish_state(force=True)

    def timer_callback(self):
        active_cmd = self.get_active_cmd()
        self.publish_cmd(active_cmd)

    def destroy_node(self):
        try:
            self.publish_cmd("X")
        except Exception:
            pass
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = VoiceModeManager()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
