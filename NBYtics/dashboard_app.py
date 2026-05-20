import math
import threading
import time

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import LaserScan


class DashboardSubscriber(Node):
    def __init__(self):
        super().__init__("dashboard_subscriber")

        self.state = {
            "stt_text": "",
            "voice_cmd": "",
            "safe_cmd": "",
            "bridge_tx": "",
            "scan_time": 0.0,
            "range_min": 0.0,
            "range_max": 0.0,
            "front_min": None,
            "left_min": None,
            "right_min": None,
            "scan_points": 0,
            "scan_ranges": [],
            "scan_angle_min": 0.0,
            "scan_angle_increment": 0.0,
        }

        self.create_subscription(String, "/stt/text", self.stt_cb, 10)
        self.create_subscription(String, "/voice_cmd", self.voice_cb, 10)
        self.create_subscription(String, "/safe_cmd", self.safe_cb, 10)
        self.create_subscription(String, "/bridge/tx", self.bridge_cb, 10)
        self.create_subscription(LaserScan, "/scan", self.scan_cb, 10)

    def stt_cb(self, msg):
        self.state["stt_text"] = msg.data

    def voice_cb(self, msg):
        self.state["voice_cmd"] = msg.data

    def safe_cb(self, msg):
        self.state["safe_cmd"] = msg.data

    def bridge_cb(self, msg):
        self.state["bridge_tx"] = msg.data

    def scan_cb(self, msg: LaserScan):
        ranges = np.array(msg.ranges, dtype=np.float32)

        valid = np.isfinite(ranges)
        valid &= ranges > msg.range_min
        valid &= ranges < msg.range_max

        filtered = np.where(valid, ranges, np.nan)

        self.state["scan_time"] = time.time()
        self.state["range_min"] = float(msg.range_min)
        self.state["range_max"] = float(msg.range_max)
        self.state["scan_points"] = len(msg.ranges)
        self.state["scan_ranges"] = filtered.tolist()
        self.state["scan_angle_min"] = float(msg.angle_min)
        self.state["scan_angle_increment"] = float(msg.angle_increment)

        def sector_min(center_deg, width_deg=30.0):
            angle_min = msg.angle_min
            angle_inc = msg.angle_increment
            vals = []

            for i, r in enumerate(filtered):
                if math.isnan(r):
                    continue
                ang_deg = math.degrees(angle_min + i * angle_inc)
                diff = ((ang_deg - center_deg + 180) % 360) - 180
                if abs(diff) <= width_deg / 2.0:
                    vals.append(float(r))

            if not vals:
                return None
            return min(vals)

        self.state["front_min"] = sector_min(0.0, 40.0)
        self.state["left_min"] = sector_min(90.0, 40.0)
        self.state["right_min"] = sector_min(-90.0, 40.0)


def ros_spin_thread(node):
    rclpy.spin(node)


@st.cache_resource
def start_ros():
    rclpy.init(args=None)
    node = DashboardSubscriber()
    thread = threading.Thread(target=ros_spin_thread, args=(node,), daemon=True)
    thread.start()
    return node


def draw_scan(scan_ranges, angle_min, angle_increment):
    if not scan_ranges:
        st.write("라이다 데이터 없음")
        return

    rs = np.array(scan_ranges, dtype=np.float32)
    valid = np.isfinite(rs)

    if valid.sum() == 0:
        st.write("유효한 라이다 데이터 없음")
        return

    idx = np.arange(len(rs))
    angles = angle_min + idx * angle_increment

    x = rs[valid] * np.cos(angles[valid])
    y = rs[valid] * np.sin(angles[valid])

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(x, y, s=4)
    ax.set_title("Lidar Scan")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.axis("equal")
    ax.grid(True)

    st.pyplot(fig)


def main():
    st.set_page_config(page_title="NBYtics Robot Dashboard", layout="wide")
    st.title("NBYtics Robot Dashboard")

    node = start_ros()
    state = node.state

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Voice Cmd", state["voice_cmd"] if state["voice_cmd"] else "-")
    c2.metric("Safe Cmd", state["safe_cmd"] if state["safe_cmd"] else "-")
    c3.metric("Bridge TX", state["bridge_tx"] if state["bridge_tx"] else "-")
    c4.metric("Scan Points", state["scan_points"])

    st.subheader("STT")
    st.write(state["stt_text"] if state["stt_text"] else "(대기 중)")

    c5, c6, c7 = st.columns(3)
    c5.metric("Front Min (m)", "-" if state["front_min"] is None else f"{state['front_min']:.2f}")
    c6.metric("Left Min (m)", "-" if state["left_min"] is None else f"{state['left_min']:.2f}")
    c7.metric("Right Min (m)", "-" if state["right_min"] is None else f"{state['right_min']:.2f}")

    st.subheader("Lidar Raw View")
    draw_scan(
        state["scan_ranges"],
        state["scan_angle_min"],
        state["scan_angle_increment"],
    )

    st.caption("자동 새로고침이 필요하면 브라우저 리로드 또는 streamlit-autorefresh 사용")

if __name__ == "__main__":
    main()
