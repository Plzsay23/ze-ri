#!/usr/bin/env python3
# zeri_dashboard_server.py

import argparse
import asyncio
import base64
import json
import threading
import time
from typing import Any, Dict, Optional

import cv2
import numpy as np
import rclpy
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import String


INDEX_HTML = r"""
<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8" />
  <title>Ze-Ri ROS2 VLM Dashboard</title>
  <meta name="viewport" content="width=device-width, initial-scale=1" />

  <style>
    :root {
      --bg: #f3f6fb;
      --card: #0f63d8;
      --card2: #0b56c4;
      --border: #062b63;
      --text: #ffffff;
      --muted: #dbeafe;
      --dark-panel: #071f49;
      --dark: #0f172a;
    }

    * {
      box-sizing: border-box;
    }

    body {
      margin: 0;
      background: var(--bg);
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--dark);
      overflow: auto;
    }

    header {
      height: 42px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0 14px;
      background: #0f172a;
      color: white;
      border-bottom: 3px solid #2563eb;
    }

    header .title {
      font-size: 18px;
      font-weight: 900;
      letter-spacing: 0.2px;
    }

    header .status {
      font-size: 13px;
      font-weight: 700;
      color: #bfdbfe;
      display: flex;
      align-items: center;
      gap: 12px;
    }

    .dot {
      width: 9px;
      height: 9px;
      border-radius: 50%;
      background: #ef4444;
      display: inline-block;
    }

    .dot.ok {
      background: #22c55e;
    }

    .grid {
      display: grid;
      grid-template-columns: 1.05fr 1.05fr 0.95fr;
      grid-template-rows: 280px 280px;
      gap: 12px;
      padding: 12px 10px 10px 10px;
      height: auto;
      max-width: 1500px;
      margin: 0 auto;
    }

    .card {
      background: linear-gradient(135deg, var(--card), var(--card2));
      border: 3px solid var(--border);
      border-radius: 8px;
      box-shadow: 0 4px 10px rgba(15, 23, 42, 0.24);
      color: var(--text);
      overflow: hidden;
      display: flex;
      flex-direction: column;
      min-width: 0;
      min-height: 0;
    }

    .card-title {
      height: 32px;
      flex: 0 0 32px;
      padding: 6px 10px;
      font-size: 16px;
      font-weight: 900;
      background: rgba(0, 0, 0, 0.17);
      border-bottom: 1px solid rgba(255, 255, 255, 0.18);
      display: flex;
      align-items: center;
      justify-content: space-between;
    }

    .card-title small {
      font-size: 10px;
      color: var(--muted);
      font-weight: 800;
      text-align: right;
      max-width: 55%;
      overflow: hidden;
      white-space: nowrap;
      text-overflow: ellipsis;
    }

    .media-wrap {
      position: relative;
      flex: 1;
      min-height: 0;
      display: flex;
      align-items: center;
      justify-content: center;
      background: var(--dark-panel);
    }

    .media-wrap img {
      width: 100%;
      height: 100%;
      object-fit: contain;
      image-rendering: auto;
      background: var(--dark-panel);
    }

    .placeholder {
      color: #bfdbfe;
      font-size: 22px;
      font-weight: 900;
      text-align: center;
      padding: 14px;
    }

    .text-panel {
      flex: 1;
      min-height: 0;
      padding: 10px;
      overflow: auto;
      background: rgba(5, 20, 48, 0.20);
    }

    .metric {
      margin-bottom: 8px;
      padding: 8px 10px;
      background: rgba(255, 255, 255, 0.13);
      border: 1px solid rgba(255, 255, 255, 0.18);
      border-radius: 8px;
    }

    .metric-label {
      color: #dbeafe;
      font-size: 12px;
      font-weight: 900;
      margin-bottom: 4px;
    }

    .metric-value {
      color: white;
      font-size: 17px;
      font-weight: 800;
      white-space: pre-wrap;
      word-break: break-word;
    }

    .metric-value.small {
      font-size: 12px;
      line-height: 1.35;
      font-weight: 700;
    }

    .status-big {
      flex: 1;
      min-height: 0;
      padding: 14px;
      display: flex;
      flex-direction: column;
      justify-content: center;
      gap: 10px;
      background: rgba(5, 20, 48, 0.15);
    }

    .status-line {
      padding: 12px;
      border-radius: 8px;
      background: rgba(255, 255, 255, 0.13);
      border: 1px solid rgba(255, 255, 255, 0.18);
    }

    .status-line .label {
      font-size: 12px;
      font-weight: 900;
      color: #dbeafe;
      margin-bottom: 6px;
    }

    .status-line .value {
      font-size: 24px;
      font-weight: 900;
      color: white;
      word-break: break-word;
    }

    @media (max-width: 1200px) {
      body {
        overflow-y: auto;
      }

      .grid {
        grid-template-columns: 1fr;
        grid-template-rows: repeat(6, 300px);
        height: auto;
      }
    }
  </style>
</head>

<body>
  <header>
    <div class="title">Ze-Ri ROS2 VLM Dashboard</div>
    <div class="status">
      <span><span id="ws-dot" class="dot"></span> <span id="ws-state">DISCONNECTED</span></span>
      <span>|</span>
      <span id="clock">-</span>
    </div>
  </header>

  <main class="grid">
    <section class="card">
      <div class="card-title">
        <span>RGB 채널</span>
        <small id="rgb-ts">대기중</small>
      </div>
      <div class="media-wrap">
        <img id="rgb-img" alt="RGB Stream" style="display:none" />
        <div id="rgb-placeholder" class="placeholder">RGB 데이터 대기중</div>
      </div>
    </section>

    <section class="card">
      <div class="card-title">
        <span>2D Map</span>
        <small id="map-ts">대기중</small>
      </div>
      <div class="media-wrap">
        <img id="map-img" alt="2D Map" style="display:none" />
        <div id="map-placeholder" class="placeholder">Map 데이터 대기중</div>
      </div>
    </section>

    <section class="card">
      <div class="card-title">
        <span>모바일 베이스 상태</span>
        <small id="base-ts">대기중</small>
      </div>
      <div class="status-big">
        <div class="status-line">
          <div class="label">상태 / 명령</div>
          <div id="base-status" class="value">데이터 대기중</div>
        </div>
      </div>
    </section>

    <section class="card">
      <div class="card-title">
        <span>Depth 채널</span>
        <small id="depth-ts">대기중</small>
      </div>
      <div class="media-wrap">
        <img id="depth-img" alt="Depth Stream" style="display:none" />
        <div id="depth-placeholder" class="placeholder">Depth 데이터 대기중</div>
      </div>
    </section>

    <section class="card">
      <div class="card-title">
        <span>STT / VLM 출력</span>
        <small id="vlm-ts">대기중</small>
      </div>
      <div class="text-panel">
        <div class="metric">
          <div class="metric-label">STT 입력값</div>
          <div id="stt-text" class="metric-value">대기중</div>
        </div>

        <div class="metric">
          <div class="metric-label">TTS 출력</div>
          <div id="robot-speech" class="metric-value">대기중</div>
        </div>

        <div class="metric">
          <div class="metric-label">VLM 상태</div>
          <div id="vlm-status" class="metric-value small">대기중</div>
        </div>

        <div class="metric">
          <div class="metric-label">VLM 출력</div>
          <div id="vlm-output" class="metric-value small">대기중</div>
        </div>

        <div class="metric">
          <div class="metric-label">VLM 출력 근거</div>
          <div id="vlm-reason" class="metric-value small">대기중</div>
        </div>
      </div>
    </section>

    <section class="card">
      <div class="card-title">
        <span>로봇팔 상태</span>
        <small id="arm-ts">대기중</small>
      </div>
      <div class="status-big">
        <div class="status-line">
          <div class="label">상태</div>
          <div id="arm-status" class="value">데이터 대기중</div>
        </div>
      </div>
    </section>
  </main>

  <script>
    const $ = (id) => document.getElementById(id);

    function setImg(imgId, placeholderId, dataUrl) {
      const img = $(imgId);
      const ph = $(placeholderId);

      if (dataUrl) {
        img.src = dataUrl;
        img.style.display = "block";
        ph.style.display = "none";
      } else {
        img.style.display = "none";
        ph.style.display = "block";
      }
    }

    function fmtTs(ts, count) {
      if (!ts) return "대기중";
      const d = new Date(ts * 1000);
      const t = d.toLocaleTimeString();
      if (count !== undefined && count !== null) {
        return `${t} | ${count}`;
      }
      return t;
    }

    function safeText(v, fallback = "대기중") {
      if (v === null || v === undefined || v === "") return fallback;
      if (typeof v === "object") return JSON.stringify(v, null, 2);
      return String(v);
    }

    function updateDashboard(data) {
      setImg("rgb-img", "rgb-placeholder", data.rgb_image);
      setImg("depth-img", "depth-placeholder", data.depth_image);
      setImg("map-img", "map-placeholder", data.map_image);

      $("rgb-ts").textContent = fmtTs(data.timestamps?.rgb, data.counts?.rgb);
      $("depth-ts").textContent = fmtTs(data.timestamps?.depth, data.counts?.depth);
      $("map-ts").textContent = fmtTs(data.timestamps?.map, data.counts?.map);
      $("vlm-ts").textContent = fmtTs(data.timestamps?.vlm || data.timestamps?.stt);
      $("base-ts").textContent = fmtTs(data.timestamps?.base);
      $("arm-ts").textContent = fmtTs(data.timestamps?.arm);

      $("stt-text").textContent = safeText(data.stt_text);
      $("robot-speech").textContent = safeText(data.robot_speech || data.vlm_decision?.robot_speech);
      $("vlm-status").textContent = safeText(data.inference_status);

      const d = data.vlm_decision || {};
      const vlmLines = [
        `need_oxygen_mask: ${safeText(d.need_oxygen_mask, "-")}`,
        `selected_task: ${safeText(d.selected_task, "-")}`,
        `adapter_id: ${safeText(d.adapter_id, "-")}`,
        `confidence: ${safeText(d.confidence, "-")}`,
        `mock_action: ${safeText(d.mock_action, "-")}`,
        `latency_sec: ${safeText(d.latency_sec, "-")}`,
        `rgb_topic: ${safeText(d.vlm_input_rgb_topic, "-")}`,
        `depth_topic: ${safeText(d.vlm_input_depth_topic, "-")}`
      ];

      $("vlm-output").textContent = vlmLines.join("\n");
      $("vlm-reason").textContent = safeText(d.reason);
      $("base-status").textContent = safeText(data.base_status);
      $("arm-status").textContent = safeText(data.arm_status);
    }

    function connect() {
      const proto = location.protocol === "https:" ? "wss" : "ws";
      const ws = new WebSocket(`${proto}://${location.host}/ws`);

      ws.onopen = () => {
        $("ws-dot").classList.add("ok");
        $("ws-state").textContent = "CONNECTED";
      };

      ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        updateDashboard(data);
      };

      ws.onclose = () => {
        $("ws-dot").classList.remove("ok");
        $("ws-state").textContent = "DISCONNECTED";
        setTimeout(connect, 1000);
      };

      ws.onerror = () => {
        ws.close();
      };
    }

    setInterval(() => {
      $("clock").textContent = new Date().toLocaleString();
    }, 500);

    connect();
  </script>
</body>
</html>
"""


class SharedState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.data: Dict[str, Any] = {
            "rgb_image": None,
            "depth_image": None,
            "map_image": None,
            "stt_text": "",
            "vlm_decision": {},
            "robot_speech": "",
            "inference_status": "대기중",
            "base_status": "데이터 대기중",
            "arm_status": "데이터 대기중",
            "timestamps": {},
            "counts": {
                "rgb": 0,
                "depth": 0,
                "map": 0,
            },
        }

    def update(self, **kwargs: Any) -> None:
        with self.lock:
            for key, value in kwargs.items():
                self.data[key] = value

    def update_timestamp(self, key: str) -> None:
        with self.lock:
            self.data.setdefault("timestamps", {})[key] = time.time()

    def increment_count(self, key: str) -> None:
        with self.lock:
            self.data.setdefault("counts", {})
            self.data["counts"][key] = int(self.data["counts"].get(key, 0)) + 1

    def snapshot(self) -> Dict[str, Any]:
        with self.lock:
            return json.loads(json.dumps(self.data, ensure_ascii=False))


STATE = SharedState()
app = FastAPI()


def make_qos(reliability: str, depth: int) -> QoSProfile:
    if reliability == "best_effort":
        rel = ReliabilityPolicy.BEST_EFFORT
    else:
        rel = ReliabilityPolicy.RELIABLE

    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=depth,
        reliability=rel,
    )


def encode_image_to_data_url(
    image_bgr: Optional[np.ndarray],
    ext: str = ".jpg",
    quality: int = 80,
) -> Optional[str]:
    if image_bgr is None:
        return None

    params = []
    if ext.lower() in [".jpg", ".jpeg"]:
        params = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]

    ok, encoded = cv2.imencode(ext, image_bgr, params)

    if not ok:
        return None

    mime = "image/jpeg" if ext.lower() in [".jpg", ".jpeg"] else "image/png"
    b64 = base64.b64encode(encoded.tobytes()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def image_msg_to_bgr(msg: Image) -> Optional[np.ndarray]:
    encoding = msg.encoding.lower()
    height = int(msg.height)
    width = int(msg.width)
    step = int(msg.step)

    if height <= 0 or width <= 0:
        return None

    raw = bytes(msg.data)

    try:
        if encoding in ("bgr8", "rgb8"):
            channels = 3
            arr = np.frombuffer(raw, dtype=np.uint8)
            row_pixels = step // channels
            arr = arr.reshape((height, row_pixels, channels))[:, :width, :]

            if encoding == "rgb8":
                arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

            return arr.copy()

        if encoding in ("bgra8", "rgba8"):
            channels = 4
            arr = np.frombuffer(raw, dtype=np.uint8)
            row_pixels = step // channels
            arr = arr.reshape((height, row_pixels, channels))[:, :width, :]

            if encoding == "rgba8":
                arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
            else:
                arr = cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)

            return arr.copy()

        if encoding in ("mono8", "8uc1"):
            arr = np.frombuffer(raw, dtype=np.uint8)
            row_pixels = step
            arr = arr.reshape((height, row_pixels))[:, :width]
            return cv2.cvtColor(arr.copy(), cv2.COLOR_GRAY2BGR)

        if encoding in ("16uc1", "mono16", "32fc1"):
            return depth_msg_to_colormap(msg)

    except Exception:
        return None

    return None


def depth_msg_to_colormap(msg: Image) -> Optional[np.ndarray]:
    encoding = msg.encoding.lower()
    height = int(msg.height)
    width = int(msg.width)
    step = int(msg.step)

    if height <= 0 or width <= 0:
        return None

    raw = bytes(msg.data)

    try:
        if encoding in ("16uc1", "mono16"):
            arr = np.frombuffer(raw, dtype=np.uint16)
            row_pixels = step // np.dtype(np.uint16).itemsize
            depth = arr.reshape((height, row_pixels))[:, :width].astype(np.float32)
            valid = depth[depth > 0]

        elif encoding == "32fc1":
            arr = np.frombuffer(raw, dtype=np.float32)
            row_pixels = step // np.dtype(np.float32).itemsize
            depth = arr.reshape((height, row_pixels))[:, :width].astype(np.float32)
            valid = depth[np.isfinite(depth) & (depth > 0)]

        elif encoding in ("mono8", "8uc1"):
            arr = np.frombuffer(raw, dtype=np.uint8)
            row_pixels = step
            gray = arr.reshape((height, row_pixels))[:, :width]
            return cv2.applyColorMap(gray.copy(), cv2.COLORMAP_JET)

        else:
            return image_msg_to_bgr(msg)

        if valid.size == 0:
            gray = np.zeros((height, width), dtype=np.uint8)
        else:
            lo = float(np.percentile(valid, 5))
            hi = float(np.percentile(valid, 95))

            if hi <= lo:
                hi = lo + 1.0

            clipped = np.clip(depth, lo, hi)
            norm = ((clipped - lo) / (hi - lo) * 255.0).astype(np.uint8)
            norm[depth <= 0] = 0
            gray = norm

        colored = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
        return colored

    except Exception:
        return None


def occupancy_grid_to_png_data_url(msg: OccupancyGrid) -> Optional[str]:
    width = int(msg.info.width)
    height = int(msg.info.height)

    if width <= 0 or height <= 0:
        return None

    try:
        grid = np.array(msg.data, dtype=np.int16).reshape((height, width))

        img = np.zeros((height, width), dtype=np.uint8)
        img[grid < 0] = 127
        img[grid == 0] = 255
        img[(grid > 0) & (grid < 100)] = 180
        img[grid >= 100] = 0

        img = np.flipud(img)

        bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        scale = max(1, min(4, 640 // max(1, width)))
        bgr = cv2.resize(
            bgr,
            (width * scale, height * scale),
            interpolation=cv2.INTER_NEAREST,
        )

        return encode_image_to_data_url(bgr, ext=".png")

    except Exception:
        return None


def parse_json_string_or_raw(text: str) -> Dict[str, Any]:
    text = text.strip()

    if not text:
        return {}

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
        return {"value": parsed}
    except json.JSONDecodeError:
        return {"raw": text}


class ZeriDashboardNode(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("zeri_dashboard_server_node")

        self.args = args

        image_qos = make_qos(args.image_qos, args.image_qos_depth)
        text_qos = make_qos("reliable", 10)
        map_qos = make_qos(args.map_qos, 5)

        self.create_subscription(Image, args.rgb_topic, self.rgb_callback, image_qos)
        self.create_subscription(Image, args.depth_topic, self.depth_callback, image_qos)
        self.create_subscription(OccupancyGrid, args.map_topic, self.map_callback, map_qos)

        self.create_subscription(String, args.stt_topic, self.stt_callback, text_qos)
        self.create_subscription(String, args.vlm_decision_topic, self.vlm_decision_callback, text_qos)
        self.create_subscription(String, args.robot_speech_topic, self.robot_speech_callback, text_qos)
        self.create_subscription(String, args.inference_status_topic, self.inference_status_callback, text_qos)

        self.create_subscription(String, args.base_status_topic, self.base_status_callback, text_qos)
        self.create_subscription(String, args.arm_status_topic, self.arm_status_callback, text_qos)

        self.get_logger().info("Ze-Ri Dashboard subscriptions:")
        self.get_logger().info(f"  RGB:              {args.rgb_topic}")
        self.get_logger().info(f"  Depth:            {args.depth_topic}")
        self.get_logger().info(f"  Map:              {args.map_topic}")
        self.get_logger().info(f"  STT:              {args.stt_topic}")
        self.get_logger().info(f"  VLM decision:     {args.vlm_decision_topic}")
        self.get_logger().info(f"  Robot speech:     {args.robot_speech_topic}")
        self.get_logger().info(f"  Inference status: {args.inference_status_topic}")
        self.get_logger().info(f"  Base status:      {args.base_status_topic}")
        self.get_logger().info(f"  Arm status:       {args.arm_status_topic}")
        self.get_logger().info(f"  Image QoS:        {args.image_qos}")

    def rgb_callback(self, msg: Image) -> None:
        bgr = image_msg_to_bgr(msg)
        data_url = encode_image_to_data_url(
            bgr,
            ext=".jpg",
            quality=self.args.jpeg_quality,
        )

        if data_url:
            STATE.update(rgb_image=data_url)
            STATE.update_timestamp("rgb")
            STATE.increment_count("rgb")
            self.get_logger().debug("RGB frame updated.")

    def depth_callback(self, msg: Image) -> None:
        bgr = depth_msg_to_colormap(msg)
        data_url = encode_image_to_data_url(
            bgr,
            ext=".jpg",
            quality=self.args.jpeg_quality,
        )

        if data_url:
            STATE.update(depth_image=data_url)
            STATE.update_timestamp("depth")
            STATE.increment_count("depth")
            self.get_logger().debug("Depth frame updated.")

    def map_callback(self, msg: OccupancyGrid) -> None:
        data_url = occupancy_grid_to_png_data_url(msg)

        if data_url:
            STATE.update(map_image=data_url)
            STATE.update_timestamp("map")
            STATE.increment_count("map")

    def stt_callback(self, msg: String) -> None:
        STATE.update(stt_text=msg.data)
        STATE.update_timestamp("stt")

    def vlm_decision_callback(self, msg: String) -> None:
        parsed = parse_json_string_or_raw(msg.data)

        if "stt_text" in parsed:
            STATE.update(stt_text=str(parsed.get("stt_text", "")))

        if "robot_speech" in parsed:
            STATE.update(robot_speech=str(parsed.get("robot_speech", "")))

        STATE.update(vlm_decision=parsed)
        STATE.update_timestamp("vlm")

    def robot_speech_callback(self, msg: String) -> None:
        STATE.update(robot_speech=msg.data)
        STATE.update_timestamp("vlm")

    def inference_status_callback(self, msg: String) -> None:
        STATE.update(inference_status=msg.data)
        STATE.update_timestamp("vlm")

    def base_status_callback(self, msg: String) -> None:
        STATE.update(base_status=msg.data)
        STATE.update_timestamp("base")

    def arm_status_callback(self, msg: String) -> None:
        STATE.update(arm_status=msg.data)
        STATE.update_timestamp("arm")


@app.get("/")
async def index() -> HTMLResponse:
    return HTMLResponse(INDEX_HTML)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()

    try:
        while True:
            await websocket.send_text(json.dumps(STATE.snapshot(), ensure_ascii=False))
            await asyncio.sleep(0.15)
    except WebSocketDisconnect:
        return


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ze-Ri ROS2 Web Dashboard")

    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--jpeg-quality", type=int, default=75)

    parser.add_argument("--image-qos", choices=["reliable", "best_effort"], default="reliable")
    parser.add_argument("--map-qos", choices=["reliable", "best_effort"], default="reliable")
    parser.add_argument("--image-qos-depth", type=int, default=10)

    parser.add_argument("--rgb-topic", default="/zeri/vlm/input_rgb")
    parser.add_argument("--depth-topic", default="/zeri/vlm/input_depth")
    parser.add_argument("--map-topic", default="/map")

    parser.add_argument("--stt-topic", default="/zeri/stt/text")
    parser.add_argument("--vlm-decision-topic", default="/zeri/vlm/decision")
    parser.add_argument("--robot-speech-topic", default="/zeri/vlm/robot_speech")
    parser.add_argument("--inference-status-topic", default="/zeri/vlm/inference_status")

    parser.add_argument("--base-status-topic", default="/zeri/mobile_base/status")
    parser.add_argument("--arm-status-topic", default="/zeri/arm/status")

    return parser


def main() -> None:
    parser = build_arg_parser()
    args, ros_args = parser.parse_known_args()

    rclpy.init(args=ros_args)

    node = ZeriDashboardNode(args)

    spin_thread = threading.Thread(
        target=rclpy.spin,
        args=(node,),
        daemon=True,
    )
    spin_thread.start()

    print("")
    print("====================================================")
    print(" Ze-Ri ROS2 Dashboard")
    print(f" URL: http://{args.host}:{args.port}")
    print(" Remote:")
    print(f" http://<ROBOT_IP>:{args.port}")
    print("====================================================")
    print("")

    try:
        config = uvicorn.Config(
            app,
            host=args.host,
            port=args.port,
            log_level="warning",
        )
        server = uvicorn.Server(config)
        server.run()
    finally:
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()

        spin_thread.join(timeout=2.0)


if __name__ == "__main__":
    main()