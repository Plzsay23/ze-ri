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
  <title>Ze-Ri ROS2 Dashboard</title>
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <style>
    :root {
      --bg: #f3f6fb;
      --card: #0f63d8;
      --card2: #0b56c4;
      --border: #062b63;
      --text: #ffffff;
      --muted: #dbeafe;
      --panel-bg: #0b3f91;
      --danger: #ffdddd;
      --ok: #d7ffe1;
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
    }

    header {
      height: 52px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0 18px;
      background: #0f172a;
      color: white;
      border-bottom: 3px solid #1d4ed8;
    }

    header .title {
      font-size: 20px;
      font-weight: 800;
      letter-spacing: 0.3px;
    }

    header .status {
      font-size: 14px;
      font-weight: 600;
      color: #bfdbfe;
    }

    .grid {
      display: grid;
      grid-template-columns: 1.05fr 1.05fr 0.95fr;
      grid-template-rows: calc((100vh - 84px) / 2) calc((100vh - 84px) / 2);
      gap: 14px;
      padding: 14px;
    }

    .card {
      background: linear-gradient(135deg, var(--card), var(--card2));
      border: 3px solid var(--border);
      border-radius: 8px;
      box-shadow: 0 6px 14px rgba(15, 23, 42, 0.22);
      color: var(--text);
      overflow: hidden;
      display: flex;
      flex-direction: column;
      min-width: 0;
      min-height: 0;
    }

    .card-title {
      height: 38px;
      flex: 0 0 38px;
      padding: 8px 12px;
      font-size: 17px;
      font-weight: 800;
      background: rgba(0, 0, 0, 0.18);
      border-bottom: 1px solid rgba(255, 255, 255, 0.18);
      display: flex;
      align-items: center;
      justify-content: space-between;
    }

    .card-title small {
      font-size: 11px;
      color: var(--muted);
      font-weight: 600;
    }

    .media-wrap {
      position: relative;
      flex: 1;
      min-height: 0;
      display: flex;
      align-items: center;
      justify-content: center;
      background: #071f49;
    }

    .media-wrap img {
      width: 100%;
      height: 100%;
      object-fit: contain;
      image-rendering: auto;
      background: #071f49;
    }

    .placeholder {
      color: #bfdbfe;
      font-size: 24px;
      font-weight: 800;
      text-align: center;
      padding: 16px;
    }

    .text-panel {
      flex: 1;
      min-height: 0;
      padding: 14px;
      overflow: auto;
      background: rgba(5, 20, 48, 0.22);
    }

    .metric {
      margin-bottom: 12px;
      padding: 10px 12px;
      background: rgba(255, 255, 255, 0.12);
      border: 1px solid rgba(255, 255, 255, 0.16);
      border-radius: 8px;
    }

    .metric-label {
      color: #dbeafe;
      font-size: 13px;
      font-weight: 800;
      margin-bottom: 4px;
    }

    .metric-value {
      color: white;
      font-size: 18px;
      font-weight: 700;
      white-space: pre-wrap;
      word-break: break-word;
    }

    .metric-value.small {
      font-size: 14px;
      line-height: 1.4;
      font-weight: 600;
    }

    .status-big {
      flex: 1;
      min-height: 0;
      padding: 18px;
      display: flex;
      flex-direction: column;
      justify-content: center;
      gap: 12px;
      background: rgba(5, 20, 48, 0.18);
    }

    .status-line {
      padding: 14px;
      border-radius: 8px;
      background: rgba(255, 255, 255, 0.13);
      border: 1px solid rgba(255, 255, 255, 0.18);
    }

    .status-line .label {
      font-size: 13px;
      font-weight: 800;
      color: #dbeafe;
      margin-bottom: 4px;
    }

    .status-line .value {
      font-size: 26px;
      font-weight: 900;
      color: white;
      word-break: break-word;
    }

    .footer {
      height: 32px;
      padding: 6px 14px;
      font-size: 12px;
      color: #334155;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }

    .badge {
      display: inline-flex;
      align-items: center;
      gap: 6px;
    }

    .dot {
      width: 9px;
      height: 9px;
      border-radius: 50%;
      background: #ef4444;
    }

    .dot.ok {
      background: #22c55e;
    }

    @media (max-width: 1200px) {
      .grid {
        grid-template-columns: 1fr;
        grid-template-rows: repeat(6, 360px);
      }
      body {
        overflow-y: auto;
      }
    }
  </style>
</head>
<body>
  <header>
    <div class="title">Ze-Ri ROS2 VLM Dashboard</div>
    <div class="status">
      <span class="badge"><span id="ws-dot" class="dot"></span><span id="ws-state">DISCONNECTED</span></span>
      &nbsp; | &nbsp;
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
          <div class="metric-label">VLM 출력</div>
          <div id="vlm-output" class="metric-value small">대기중</div>
        </div>
        <div class="metric">
          <div class="metric-label">VLM 출력 근거</div>
          <div id="vlm-reason" class="metric-value small">대기중</div>
        </div>
        <div class="metric">
          <div class="metric-label">로봇 발화문 / TTS 예정</div>
          <div id="robot-speech" class="metric-value">대기중</div>
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

  <div class="footer">
    <div>RGB / Depth / Map / STT / VLM / Base / Arm topics are monitored through rclpy.</div>
    <div id="last-update">last update: -</div>
  </div>

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

    function fmtTs(ts) {
      if (!ts) return "대기중";
      const d = new Date(ts * 1000);
      return d.toLocaleTimeString();
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

      $("rgb-ts").textContent = fmtTs(data.timestamps?.rgb);
      $("depth-ts").textContent = fmtTs(data.timestamps?.depth);
      $("map-ts").textContent = fmtTs(data.timestamps?.map);
      $("vlm-ts").textContent = fmtTs(data.timestamps?.vlm || data.timestamps?.stt);
      $("base-ts").textContent = fmtTs(data.timestamps?.base);
      $("arm-ts").textContent = fmtTs(data.timestamps?.arm);

      $("stt-text").textContent = safeText(data.stt_text);

      const d = data.vlm_decision || {};
      const vlmLines = [
        `need_oxygen_mask: ${safeText(d.need_oxygen_mask, "-")}`,
        `selected_task: ${safeText(d.selected_task, "-")}`,
        `adapter_id: ${safeText(d.adapter_id, "-")}`,
        `confidence: ${safeText(d.confidence, "-")}`,
        `mock_action: ${safeText(d.mock_action, "-")}`,
        `latency_sec: ${safeText(d.latency_sec, "-")}`
      ];
      $("vlm-output").textContent = vlmLines.join("\n");

      $("vlm-reason").textContent = safeText(d.reason);
      $("robot-speech").textContent = safeText(data.robot_speech || d.robot_speech);

      $("base-status").textContent = safeText(data.base_status);
      $("arm-status").textContent = safeText(data.arm_status);

      $("last-update").textContent = "last update: " + new Date().toLocaleTimeString();
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
            "base_status": "데이터 대기중",
            "arm_status": "데이터 대기중",
            "timestamps": {},
        }

    def update(self, **kwargs: Any) -> None:
        with self.lock:
            for key, value in kwargs.items():
                self.data[key] = value

    def update_timestamp(self, key: str) -> None:
        with self.lock:
            self.data.setdefault("timestamps", {})[key] = time.time()

    def snapshot(self) -> Dict[str, Any]:
        with self.lock:
            return json.loads(json.dumps(self.data, ensure_ascii=False))


STATE = SharedState()
app = FastAPI()


def encode_image_to_data_url(image_bgr: np.ndarray, ext: str = ".jpg", quality: int = 80) -> Optional[str]:
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

    raw = msg.data

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

        # depth를 일반 이미지로 실수로 이 함수에 넣었을 때도 시각화
        if encoding in ("16uc1", "mono16"):
            return depth_msg_to_colormap(msg)

        if encoding == "32fc1":
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

    try:
        if encoding in ("16uc1", "mono16"):
            arr = np.frombuffer(msg.data, dtype=np.uint16)
            row_pixels = step // np.dtype(np.uint16).itemsize
            depth = arr.reshape((height, row_pixels))[:, :width].astype(np.float32)

            # RealSense depth 16UC1은 보통 mm 단위인 경우가 많음.
            # dashboard에서는 상대적 시각화만 하므로 그대로 정규화.
            valid = depth[depth > 0]

        elif encoding == "32fc1":
            arr = np.frombuffer(msg.data, dtype=np.float32)
            row_pixels = step // np.dtype(np.float32).itemsize
            depth = arr.reshape((height, row_pixels))[:, :width].astype(np.float32)

            valid = depth[np.isfinite(depth) & (depth > 0)]

        elif encoding in ("mono8", "8uc1"):
            arr = np.frombuffer(msg.data, dtype=np.uint8)
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

        # ROS OccupancyGrid:
        # -1 unknown, 0 free, 100 occupied
        img = np.zeros((height, width), dtype=np.uint8)
        img[grid < 0] = 127
        img[grid == 0] = 255
        img[(grid > 0) & (grid < 100)] = 180
        img[grid >= 100] = 0

        # map은 보통 origin이 좌하단 기준이므로 화면 보기 좋게 상하 반전
        img = np.flipud(img)

        bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        bgr = cv2.resize(bgr, (max(320, width * 2), max(320, height * 2)), interpolation=cv2.INTER_NEAREST)

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

        sensor_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=3,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )

        text_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
        )

        self.create_subscription(Image, args.rgb_topic, self.rgb_callback, sensor_qos)
        self.create_subscription(Image, args.depth_topic, self.depth_callback, sensor_qos)
        self.create_subscription(OccupancyGrid, args.map_topic, self.map_callback, 10)

        self.create_subscription(String, args.stt_topic, self.stt_callback, text_qos)
        self.create_subscription(String, args.vlm_decision_topic, self.vlm_decision_callback, text_qos)
        self.create_subscription(String, args.robot_speech_topic, self.robot_speech_callback, text_qos)

        self.create_subscription(String, args.base_status_topic, self.base_status_callback, text_qos)
        self.create_subscription(String, args.arm_status_topic, self.arm_status_callback, text_qos)

        self.get_logger().info("Ze-Ri Dashboard subscriptions:")
        self.get_logger().info(f"  RGB:            {args.rgb_topic}")
        self.get_logger().info(f"  Depth:          {args.depth_topic}")
        self.get_logger().info(f"  Map:            {args.map_topic}")
        self.get_logger().info(f"  STT:            {args.stt_topic}")
        self.get_logger().info(f"  VLM decision:   {args.vlm_decision_topic}")
        self.get_logger().info(f"  Robot speech:   {args.robot_speech_topic}")
        self.get_logger().info(f"  Base status:    {args.base_status_topic}")
        self.get_logger().info(f"  Arm status:     {args.arm_status_topic}")

    def rgb_callback(self, msg: Image) -> None:
        bgr = image_msg_to_bgr(msg)
        data_url = encode_image_to_data_url(bgr, ext=".jpg", quality=self.args.jpeg_quality)
        if data_url:
            STATE.update(rgb_image=data_url)
            STATE.update_timestamp("rgb")

    def depth_callback(self, msg: Image) -> None:
        bgr = depth_msg_to_colormap(msg)
        data_url = encode_image_to_data_url(bgr, ext=".jpg", quality=self.args.jpeg_quality)
        if data_url:
            STATE.update(depth_image=data_url)
            STATE.update_timestamp("depth")

    def map_callback(self, msg: OccupancyGrid) -> None:
        data_url = occupancy_grid_to_png_data_url(msg)
        if data_url:
            STATE.update(map_image=data_url)
            STATE.update_timestamp("map")

    def stt_callback(self, msg: String) -> None:
        STATE.update(stt_text=msg.data)
        STATE.update_timestamp("stt")

    def vlm_decision_callback(self, msg: String) -> None:
        parsed = parse_json_string_or_raw(msg.data)

        # decision JSON에 stt_text / robot_speech가 같이 들어오면 dashboard state도 갱신
        if "stt_text" in parsed:
            STATE.update(stt_text=str(parsed.get("stt_text", "")))

        if "robot_speech" in parsed:
            STATE.update(robot_speech=str(parsed.get("robot_speech", "")))

        STATE.update(vlm_decision=parsed)
        STATE.update_timestamp("vlm")

    def robot_speech_callback(self, msg: String) -> None:
        STATE.update(robot_speech=msg.data)
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
            await asyncio.sleep(0.1)
    except WebSocketDisconnect:
        return


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ze-Ri ROS2 Web Dashboard")

    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--jpeg-quality", type=int, default=75)

    parser.add_argument("--rgb-topic", default="/camera/camera/color/image_raw")
    parser.add_argument("--depth-topic", default="/camera/camera/depth/image_rect_raw")
    parser.add_argument("--map-topic", default="/map")

    parser.add_argument("--stt-topic", default="/zeri/stt/text")
    parser.add_argument("--vlm-decision-topic", default="/zeri/vlm/decision")
    parser.add_argument("--robot-speech-topic", default="/zeri/vlm/robot_speech")

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
    print(" If using remote browser, open:")
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
        rclpy.shutdown()
        spin_thread.join(timeout=2.0)


if __name__ == "__main__":
    main()