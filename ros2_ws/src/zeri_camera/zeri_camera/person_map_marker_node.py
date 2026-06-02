#!/usr/bin/env python3

import json
import math
import time
from pathlib import Path

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.time import Time

from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Bool
from visualization_msgs.msg import Marker, MarkerArray

import tf2_ros


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def quat_rotate(qx, qy, qz, qw, v):
    # ROS quaternion rotate: v' = q * v * q^-1
    x, y, z = v

    # uv = qvec x v
    uvx = qy * z - qz * y
    uvy = qz * x - qx * z
    uvz = qx * y - qy * x

    # uuv = qvec x uv
    uuvx = qy * uvz - qz * uvy
    uuvy = qz * uvx - qx * uvz
    uuvz = qx * uvy - qy * uvx

    uvx *= 2.0 * qw
    uvy *= 2.0 * qw
    uvz *= 2.0 * qw

    uuvx *= 2.0
    uuvy *= 2.0
    uuvz *= 2.0

    return (
        x + uvx + uuvx,
        y + uvy + uuvy,
        z + uvz + uuvz,
    )


def dist_xy(a, b):
    return math.hypot(float(a["x"]) - float(b["x"]), float(a["y"]) - float(b["y"]))


class PersonMapMarkerNode(Node):
    """
    YOLO person detection -> fixed map marker.

    핵심 정책:
      - 사람이 검출되면 map frame 좌표로 변환
      - 안정적으로 N회 이상 같은 위치에서 보이면 marker 생성
      - 생성된 marker는 절대 이동하지 않음
      - 같은 위치에서 다시 검출되면 last_seen/hits만 갱신하고 위치는 유지
      - 사람이 시야에서 사라져도 marker는 삭제하지 않음
      - marker는 JSON 파일에 저장되어 노드 재시작 후에도 유지
    """

    def __init__(self):
        super().__init__("person_map_marker_node")

        self.declare_parameter("rgb_topic", "/camera/camera/color/image_raw")
        self.declare_parameter("depth_topic", "/camera/camera/aligned_depth_to_color/image_raw")
        self.declare_parameter("camera_info_topic", "/camera/camera/color/camera_info")
        self.declare_parameter("vad_topic", "/zeri/audio/vad")

        self.declare_parameter("marker_topic", "/zeri/person_markers")
        self.declare_parameter("target_frame", "map")

        self.declare_parameter("model_path", "yolov8n.pt")
        self.declare_parameter("device", "cuda:0")
        self.declare_parameter("imgsz", 320)
        self.declare_parameter("conf_thres", 0.45)
        self.declare_parameter("infer_hz", 4.0)

        self.declare_parameter("use_vad_gate", False)
        self.declare_parameter("voice_hold_sec", 3.0)

        self.declare_parameter("rgb_timeout_sec", 1.0)
        self.declare_parameter("depth_timeout_sec", 1.0)
        self.declare_parameter("info_timeout_sec", 5.0)

        self.declare_parameter("depth_min_m", 0.35)
        self.declare_parameter("depth_max_m", 4.0)
        self.declare_parameter("min_depth_pixels", 60)

        # false positive 방지용 candidate confirm
        self.declare_parameter("candidate_radius_m", 0.70)
        self.declare_parameter("candidate_min_hits", 3)
        self.declare_parameter("candidate_confirm_sec", 0.6)
        self.declare_parameter("candidate_timeout_sec", 2.0)

        # 이미 생성된 marker와 같은 사람인지 판단하는 반경
        self.declare_parameter("dedupe_radius_m", 0.90)

        # marker 표시 설정
        self.declare_parameter("marker_z_mode", "ground")  # ground 또는 detected
        self.declare_parameter("ground_z", 0.05)
        self.declare_parameter("sphere_scale", 0.35)
        self.declare_parameter("text_z_offset", 0.65)

        self.declare_parameter(
            "storage_path",
            str(Path.home() / "ze-ri/data/person_markers.json"),
        )

        self.rgb_topic = str(self.get_parameter("rgb_topic").value)
        self.depth_topic = str(self.get_parameter("depth_topic").value)
        self.camera_info_topic = str(self.get_parameter("camera_info_topic").value)
        self.vad_topic = str(self.get_parameter("vad_topic").value)

        self.marker_topic = str(self.get_parameter("marker_topic").value)
        self.target_frame = str(self.get_parameter("target_frame").value)

        self.model_path = str(self.get_parameter("model_path").value)
        self.device = str(self.get_parameter("device").value)
        self.imgsz = int(self.get_parameter("imgsz").value)
        self.conf_thres = float(self.get_parameter("conf_thres").value)
        self.infer_hz = float(self.get_parameter("infer_hz").value)

        self.use_vad_gate = bool(self.get_parameter("use_vad_gate").value)
        self.voice_hold_sec = float(self.get_parameter("voice_hold_sec").value)

        self.rgb_timeout_sec = float(self.get_parameter("rgb_timeout_sec").value)
        self.depth_timeout_sec = float(self.get_parameter("depth_timeout_sec").value)
        self.info_timeout_sec = float(self.get_parameter("info_timeout_sec").value)

        self.depth_min_m = float(self.get_parameter("depth_min_m").value)
        self.depth_max_m = float(self.get_parameter("depth_max_m").value)
        self.min_depth_pixels = int(self.get_parameter("min_depth_pixels").value)

        self.candidate_radius_m = float(self.get_parameter("candidate_radius_m").value)
        self.candidate_min_hits = int(self.get_parameter("candidate_min_hits").value)
        self.candidate_confirm_sec = float(self.get_parameter("candidate_confirm_sec").value)
        self.candidate_timeout_sec = float(self.get_parameter("candidate_timeout_sec").value)

        self.dedupe_radius_m = float(self.get_parameter("dedupe_radius_m").value)

        self.marker_z_mode = str(self.get_parameter("marker_z_mode").value)
        self.ground_z = float(self.get_parameter("ground_z").value)
        self.sphere_scale = float(self.get_parameter("sphere_scale").value)
        self.text_z_offset = float(self.get_parameter("text_z_offset").value)

        self.storage_path = Path(str(self.get_parameter("storage_path").value))
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)

        self.last_rgb = None
        self.last_rgb_stamp = 0.0
        self.last_rgb_frame = ""

        self.last_depth = None
        self.last_depth_stamp = 0.0
        self.last_depth_frame = ""

        self.camera_info = None
        self.camera_info_stamp = 0.0
        self.camera_frame = ""

        self.last_voice_time = 0.0

        self.markers = []
        self.candidates = []
        self.next_id = 1

        self.tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.marker_pub = self.create_publisher(MarkerArray, self.marker_topic, 10)

        self.create_subscription(Image, self.rgb_topic, self.on_rgb, 10)
        self.create_subscription(Image, self.depth_topic, self.on_depth, 10)
        self.create_subscription(CameraInfo, self.camera_info_topic, self.on_camera_info, 10)
        self.create_subscription(Bool, self.vad_topic, self.on_vad, 10)

        self.get_logger().info(f"loading YOLO model: {self.model_path}")
        from ultralytics import YOLO
        self.model = YOLO(self.model_path)

        self.load_storage()

        self.timer = self.create_timer(1.0 / max(self.infer_hz, 1.0), self.on_timer)
        self.marker_timer = self.create_timer(0.5, self.publish_markers)

        self.get_logger().info(
            "person map marker node started: "
            f"rgb={self.rgb_topic}, depth={self.depth_topic}, info={self.camera_info_topic}, "
            f"target_frame={self.target_frame}, marker_topic={self.marker_topic}, "
            f"use_vad_gate={self.use_vad_gate}, storage={self.storage_path}"
        )

    def on_vad(self, msg: Bool):
        if bool(msg.data):
            self.last_voice_time = time.time()

    def on_rgb(self, msg: Image):
        try:
            self.last_rgb = self.decode_rgb(msg)
            self.last_rgb_stamp = time.time()
            self.last_rgb_frame = msg.header.frame_id
        except Exception as e:
            self.get_logger().warn(f"RGB decode failed: {repr(e)}")

    def on_depth(self, msg: Image):
        try:
            self.last_depth = self.decode_depth(msg)
            self.last_depth_stamp = time.time()
            self.last_depth_frame = msg.header.frame_id
        except Exception as e:
            self.get_logger().warn(f"Depth decode failed: {repr(e)}")

    def on_camera_info(self, msg: CameraInfo):
        self.camera_info = msg
        self.camera_info_stamp = time.time()
        self.camera_frame = msg.header.frame_id

    def decode_rgb(self, msg: Image):
        h = int(msg.height)
        w = int(msg.width)
        step = int(msg.step)
        enc = msg.encoding.lower()

        raw = np.frombuffer(msg.data, dtype=np.uint8).reshape(h, step)

        if enc == "rgb8":
            return raw[:, :w * 3].reshape(h, w, 3).copy()

        if enc == "bgr8":
            arr = raw[:, :w * 3].reshape(h, w, 3).copy()
            return arr[:, :, ::-1].copy()

        if enc in ("rgba8", "bgra8"):
            arr = raw[:, :w * 4].reshape(h, w, 4).copy()
            arr = arr[:, :, :3]
            if enc == "bgra8":
                arr = arr[:, :, ::-1]
            return arr.copy()

        raise RuntimeError(f"unsupported RGB encoding: {msg.encoding}")

    def decode_depth(self, msg: Image):
        h = int(msg.height)
        w = int(msg.width)
        step = int(msg.step)
        enc = msg.encoding

        if enc in ("16UC1", "mono16"):
            dtype = np.dtype(">u2" if msg.is_bigendian else "<u2")
            row_bytes = w * 2
            raw = np.frombuffer(msg.data, dtype=np.uint8).reshape(h, step)
            raw = raw[:, :row_bytes].copy()
            arr = raw.view(dtype).reshape(h, w)
            return arr.astype(np.float32) / 1000.0

        if enc == "32FC1":
            dtype = np.dtype(">f4" if msg.is_bigendian else "<f4")
            row_bytes = w * 4
            raw = np.frombuffer(msg.data, dtype=np.uint8).reshape(h, step)
            raw = raw[:, :row_bytes].copy()
            arr = raw.view(dtype).reshape(h, w)
            return arr.astype(np.float32)

        raise RuntimeError(f"unsupported depth encoding: {enc}")

    def load_storage(self):
        if not self.storage_path.exists():
            return

        try:
            data = json.loads(self.storage_path.read_text())
            self.markers = data.get("markers", [])
            if self.markers:
                self.next_id = max(int(m["id"]) for m in self.markers) + 1
            self.get_logger().info(f"loaded {len(self.markers)} fixed person markers")
        except Exception as e:
            self.get_logger().warn(f"failed to load marker storage: {repr(e)}")
            self.markers = []

    def save_storage(self):
        try:
            payload = {
                "target_frame": self.target_frame,
                "markers": self.markers,
            }
            self.storage_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
        except Exception as e:
            self.get_logger().warn(f"failed to save marker storage: {repr(e)}")

    def run_yolo(self, rgb):
        try:
            results = self.model.predict(
                source=rgb,
                imgsz=self.imgsz,
                conf=self.conf_thres,
                device=self.device,
                verbose=False,
            )
        except Exception as e:
            self.get_logger().warn(f"YOLO failed on {self.device}: {repr(e)}")
            if self.device != "cpu":
                results = self.model.predict(
                    source=rgb,
                    imgsz=self.imgsz,
                    conf=self.conf_thres,
                    device="cpu",
                    verbose=False,
                )
            else:
                return []

        detections = []
        if not results:
            return detections

        r = results[0]
        if r.boxes is None:
            return detections

        for b in r.boxes:
            try:
                cls_id = int(b.cls.item())
                conf = float(b.conf.item())
                xyxy = b.xyxy.cpu().numpy().reshape(-1).tolist()
            except Exception:
                continue

            # COCO person class
            if cls_id != 0:
                continue

            if conf < self.conf_thres:
                continue

            detections.append({
                "box": [float(v) for v in xyxy],
                "conf": conf,
            })

        return detections

    def depth_point_from_box(self, box, rgb_shape, depth, cam_info):
        rgb_h, rgb_w = rgb_shape[:2]
        depth_h, depth_w = depth.shape

        x1, y1, x2, y2 = box
        x1 = clamp(x1, 0, rgb_w - 1)
        x2 = clamp(x2, 0, rgb_w - 1)
        y1 = clamp(y1, 0, rgb_h - 1)
        y2 = clamp(y2, 0, rgb_h - 1)

        if x2 <= x1 or y2 <= y1:
            return None

        bw = x2 - x1
        bh = y2 - y1

        # 사람 몸통~하단 중심부. 경계/배경 섞임 방지.
        rx1 = x1 + 0.30 * bw
        rx2 = x1 + 0.70 * bw
        ry1 = y1 + 0.35 * bh
        ry2 = y1 + 0.90 * bh

        sx = depth_w / float(rgb_w)
        sy = depth_h / float(rgb_h)

        dx1 = int(clamp(rx1 * sx, 0, depth_w - 1))
        dx2 = int(clamp(rx2 * sx, 0, depth_w - 1))
        dy1 = int(clamp(ry1 * sy, 0, depth_h - 1))
        dy2 = int(clamp(ry2 * sy, 0, depth_h - 1))

        if dx2 <= dx1 or dy2 <= dy1:
            return None

        roi = depth[dy1:dy2, dx1:dx2]

        valid = np.isfinite(roi)
        valid &= roi >= self.depth_min_m
        valid &= roi <= self.depth_max_m

        count = int(np.count_nonzero(valid))
        if count < self.min_depth_pixels:
            return None

        z = float(np.median(roi[valid]))

        # bbox 중심 x, 몸통/하단 중간 y를 대표점으로 사용
        u_rgb = 0.5 * (x1 + x2)
        v_rgb = y1 + 0.65 * bh

        # CameraInfo는 color image 기준이라고 가정
        fx = float(cam_info.k[0])
        fy = float(cam_info.k[4])
        cx = float(cam_info.k[2])
        cy = float(cam_info.k[5])

        if fx <= 1e-6 or fy <= 1e-6:
            return None

        x_cam = (u_rgb - cx) / fx * z
        y_cam = (v_rgb - cy) / fy * z
        z_cam = z

        return {
            "camera_frame": cam_info.header.frame_id,
            "x": x_cam,
            "y": y_cam,
            "z": z_cam,
            "depth_n": count,
            "u": u_rgb,
            "v": v_rgb,
        }

    def transform_point_to_map(self, point):
        source_frame = point["camera_frame"]

        try:
            tf = self.tf_buffer.lookup_transform(
                self.target_frame,
                source_frame,
                Time(),
                timeout=Duration(seconds=0.15),
            )
        except Exception as e:
            self.get_logger().warn(
                f"TF unavailable {self.target_frame} <- {source_frame}: {repr(e)}"
            )
            return None

        t = tf.transform.translation
        q = tf.transform.rotation

        rx, ry, rz = quat_rotate(
            q.x, q.y, q.z, q.w,
            (point["x"], point["y"], point["z"]),
        )

        return {
            "x": float(t.x + rx),
            "y": float(t.y + ry),
            "z": float(t.z + rz),
            "source_frame": source_frame,
        }

    def is_near_existing_marker(self, p):
        for m in self.markers:
            if dist_xy(p, m) <= self.dedupe_radius_m:
                # 위치는 절대 갱신하지 않음.
                m["last_seen"] = time.time()
                m["seen_count"] = int(m.get("seen_count", 1)) + 1
                self.save_storage()
                return True
        return False

    def update_candidates(self, p, det):
        now = time.time()

        # 오래된 candidate 제거
        self.candidates = [
            c for c in self.candidates
            if now - float(c["last_seen"]) <= self.candidate_timeout_sec
        ]

        best = None
        best_d = 999.0

        for c in self.candidates:
            d = dist_xy(p, c)
            if d < best_d:
                best_d = d
                best = c

        if best is None or best_d > self.candidate_radius_m:
            self.candidates.append({
                "x": p["x"],
                "y": p["y"],
                "z": p["z"],
                "detected_z": p["detected_z"],
                "first_seen": now,
                "last_seen": now,
                "hits": 1,
                "conf": det["conf"],
                "depth_n": det["depth_n"],
                "source_frame": p.get("source_frame", ""),
            })
            return

        # candidate는 marker 확정 전까지만 위치를 부드럽게 평균냄
        a = 0.35
        best["x"] = a * p["x"] + (1.0 - a) * best["x"]
        best["y"] = a * p["y"] + (1.0 - a) * best["y"]
        best["z"] = a * p["z"] + (1.0 - a) * best["z"]
        best["detected_z"] = a * p["detected_z"] + (1.0 - a) * best["detected_z"]
        best["last_seen"] = now
        best["hits"] = int(best["hits"]) + 1
        best["conf"] = max(float(best.get("conf", 0.0)), float(det["conf"]))
        best["depth_n"] = max(int(best.get("depth_n", 0)), int(det["depth_n"]))

        age = now - float(best["first_seen"])

        if best["hits"] >= self.candidate_min_hits and age >= self.candidate_confirm_sec:
            self.create_fixed_marker(best)
            self.candidates.remove(best)

    def create_fixed_marker(self, c):
        now = time.time()

        marker_z = float(c["z"])
        if self.marker_z_mode == "ground":
            marker_z = self.ground_z

        marker = {
            "id": self.next_id,
            "x": float(c["x"]),
            "y": float(c["y"]),
            "z": marker_z,
            "detected_z": float(c.get("detected_z", c["z"])),
            "created_at": now,
            "last_seen": now,
            "seen_count": int(c.get("hits", 1)),
            "conf": float(c.get("conf", 0.0)),
            "depth_n": int(c.get("depth_n", 0)),
            "target_frame": self.target_frame,
            "source_frame": str(c.get("source_frame", "")),
            "label": f"PERSON_{self.next_id}",
        }

        self.next_id += 1
        self.markers.append(marker)
        self.save_storage()

        self.get_logger().warn(
            f"FIXED PERSON MARKER CREATED id={marker['id']} "
            f"x={marker['x']:.2f} y={marker['y']:.2f} z={marker['z']:.2f}"
        )

    def on_timer(self):
        now = time.time()

        if self.use_vad_gate:
            if now - self.last_voice_time > self.voice_hold_sec:
                return

        if self.last_rgb is None or now - self.last_rgb_stamp > self.rgb_timeout_sec:
            return

        if self.last_depth is None or now - self.last_depth_stamp > self.depth_timeout_sec:
            return

        if self.camera_info is None or now - self.camera_info_stamp > self.info_timeout_sec:
            return

        rgb = self.last_rgb.copy()
        depth = self.last_depth.copy()
        cam_info = self.camera_info

        detections = self.run_yolo(rgb)
        if not detections:
            return

        for det in detections:
            p_cam = self.depth_point_from_box(det["box"], rgb.shape, depth, cam_info)
            if p_cam is None:
                continue

            p_map = self.transform_point_to_map(p_cam)
            if p_map is None:
                continue

            p_map["detected_z"] = p_map["z"]

            # ground marker이면 표시 z만 바꾸고 x,y는 그대로 둠.
            if self.marker_z_mode == "ground":
                p_for_marker = dict(p_map)
                p_for_marker["z"] = self.ground_z
            else:
                p_for_marker = p_map

            det["depth_n"] = p_cam["depth_n"]

            if self.is_near_existing_marker(p_for_marker):
                continue

            self.update_candidates(p_for_marker, det)

    def make_sphere_marker(self, m, stamp):
        marker = Marker()
        marker.header.frame_id = self.target_frame
        marker.header.stamp = stamp
        marker.ns = "fixed_person_positions"
        marker.id = int(m["id"])
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD

        marker.pose.position.x = float(m["x"])
        marker.pose.position.y = float(m["y"])
        marker.pose.position.z = float(m["z"])

        marker.pose.orientation.w = 1.0

        marker.scale.x = self.sphere_scale
        marker.scale.y = self.sphere_scale
        marker.scale.z = self.sphere_scale

        marker.color.r = 1.0
        marker.color.g = 0.10
        marker.color.b = 0.05
        marker.color.a = 0.95

        # lifetime 0 = 계속 유지. 노드가 계속 재발행함.
        return marker

    def make_text_marker(self, m, stamp):
        marker = Marker()
        marker.header.frame_id = self.target_frame
        marker.header.stamp = stamp
        marker.ns = "fixed_person_labels"
        marker.id = 100000 + int(m["id"])
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD

        marker.pose.position.x = float(m["x"])
        marker.pose.position.y = float(m["y"])
        marker.pose.position.z = float(m["z"]) + self.text_z_offset

        marker.pose.orientation.w = 1.0

        marker.scale.z = 0.28

        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 1.0
        marker.color.a = 1.0

        marker.text = f"PERSON {int(m['id'])}"

        return marker

    def publish_markers(self):
        stamp = self.get_clock().now().to_msg()
        arr = MarkerArray()

        for m in self.markers:
            arr.markers.append(self.make_sphere_marker(m, stamp))
            arr.markers.append(self.make_text_marker(m, stamp))

        self.marker_pub.publish(arr)


def main(args=None):
    rclpy.init(args=args)
    node = PersonMapMarkerNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
