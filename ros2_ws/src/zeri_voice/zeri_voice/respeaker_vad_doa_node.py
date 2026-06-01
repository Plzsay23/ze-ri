#!/usr/bin/env python3

import struct
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Float32, String


class MinimalReSpeakerTuning:
    """
    Minimal USB control reader for ReSpeaker USB Mic Array.

    Publishes:
      - VOICEACTIVITY as VAD
      - DOAANGLE as direction-of-arrival angle

    Default USB VID/PID:
      - VID: 0x2886
      - PID: 0x0018
    """

    TIMEOUT = 100000

    PARAMETERS = {
        "VOICEACTIVITY": (19, 32, "int"),
        "DOAANGLE": (21, 0, "int"),
    }

    def __init__(self, vid: int = 0x2886, pid: int = 0x0018):
        import usb.core
        import usb.util

        self.usb = usb
        self.dev = usb.core.find(idVendor=vid, idProduct=pid)

        if self.dev is None:
            raise RuntimeError(
                f"ReSpeaker device not found: vid=0x{vid:04x}, pid=0x{pid:04x}"
            )

    def read(self, name: str):
        if name not in self.PARAMETERS:
            raise KeyError(name)

        param_id, offset, typ = self.PARAMETERS[name]

        cmd = 0x80 | offset
        if typ == "int":
            cmd |= 0x40

        response = self.dev.ctrl_transfer(
            self.usb.util.CTRL_IN
            | self.usb.util.CTRL_TYPE_VENDOR
            | self.usb.util.CTRL_RECIPIENT_DEVICE,
            0,
            cmd,
            param_id,
            8,
            self.TIMEOUT,
        )

        raw = bytes(response)
        a, b = struct.unpack("ii", raw)

        if typ == "int":
            return int(a)

        return float(a) * (2.0 ** float(b))

    def close(self):
        try:
            self.usb.util.dispose_resources(self.dev)
        except Exception:
            pass


class ReSpeakerVadDoaNode(Node):
    def __init__(self):
        super().__init__("respeaker_vad_doa_node")

        self.declare_parameter("vid", "0x2886")
        self.declare_parameter("pid", "0x0018")
        self.declare_parameter("poll_hz", 20.0)

        self.declare_parameter("vad_topic", "/zeri/audio/vad")
        self.declare_parameter("doa_topic", "/zeri/audio/doa_deg")
        self.declare_parameter("state_topic", "/zeri/audio/state")

        self.declare_parameter("log_active_only", True)

        vid_str = str(self.get_parameter("vid").value)
        pid_str = str(self.get_parameter("pid").value)

        self.vid = int(vid_str, 0)
        self.pid = int(pid_str, 0)
        self.poll_hz = float(self.get_parameter("poll_hz").value)
        self.log_active_only = bool(self.get_parameter("log_active_only").value)

        vad_topic = str(self.get_parameter("vad_topic").value)
        doa_topic = str(self.get_parameter("doa_topic").value)
        state_topic = str(self.get_parameter("state_topic").value)

        self.vad_pub = self.create_publisher(Bool, vad_topic, 10)
        self.doa_pub = self.create_publisher(Float32, doa_topic, 10)
        self.state_pub = self.create_publisher(String, state_topic, 10)

        self.device = MinimalReSpeakerTuning(self.vid, self.pid)

        self.last_vad = None
        self.last_log_time = 0.0

        period = 1.0 / max(self.poll_hz, 1.0)
        self.timer = self.create_timer(period, self.on_timer)

        self.get_logger().info(
            "ReSpeaker VAD/DOA node started: "
            f"vid=0x{self.vid:04x}, pid=0x{self.pid:04x}, "
            f"poll_hz={self.poll_hz}, vad_topic={vad_topic}, doa_topic={doa_topic}"
        )

    def on_timer(self):
        try:
            vad_raw = self.device.read("VOICEACTIVITY")
            doa_raw = self.device.read("DOAANGLE")
        except Exception as e:
            msg = String()
            msg.data = f"ERROR {repr(e)}"
            self.state_pub.publish(msg)
            self.get_logger().warn(f"failed to read ReSpeaker: {repr(e)}")
            return

        vad = bool(int(vad_raw))
        doa = float(doa_raw) % 360.0

        vad_msg = Bool()
        vad_msg.data = vad
        self.vad_pub.publish(vad_msg)

        doa_msg = Float32()
        doa_msg.data = doa
        self.doa_pub.publish(doa_msg)

        state_msg = String()
        state_msg.data = f"vad={int(vad)} doa_deg={doa:.1f}"
        self.state_pub.publish(state_msg)

        now = time.time()
        should_log = False

        if self.last_vad is None or self.last_vad != vad:
            should_log = True
        elif vad and now - self.last_log_time > 1.0:
            should_log = True
        elif not self.log_active_only and now - self.last_log_time > 2.0:
            should_log = True

        if should_log:
            self.get_logger().info(state_msg.data)
            self.last_log_time = now

        self.last_vad = vad

    def destroy_node(self):
        try:
            self.device.close()
        except Exception:
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ReSpeakerVadDoaNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
