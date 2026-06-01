#!/usr/bin/env python3

import time
import math
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import Image
from std_msgs.msg import Bool, String
from geometry_msgs.msg import Twist


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


class CameraPersonFollowNode(Node):
    """
    YOLO person detector + depth distance follower.

    입력:
      /zeri/vlm/input_rgb
      /zeri/vlm/input_depth
      /zeri/audio/vad

    출력:
      /cmd_vel_raw
      /zeri/person_follow/state

    핵심:
      - DOA 사용 안 함
      - depth nearest blob 사용 안 함
      - YOLO person bbox만 target으로 사용
      - 사람이 없으면 정지
      - bbox 중심 x로 방향 제어
      - bbox 내부 median depth로 전진/정지 판단
      - linear.x + angular.z 동시 출력 가능
      - linear.y는 항상 0
    """

    def __init__(self):
        super().__init__("camera_person_follow_node")

        self.declare_parameter("rgb_topic", "/zeri/vlm/input_rgb")
        self.declare_parameter("depth_topic", "/zeri/vlm/input_depth")
        self.declare_parameter("vad_topic", "/zeri/audio/vad")
        self.declare_parameter("cmd_topic", "/cmd_vel_raw")
        self.declare_parameter("state_topic", "/zeri/person_follow/state")

        self.declare_parameter("model_path", "yolov8n.pt")
        self.declare_parameter("device", "cuda:0")
        self.declare_parameter("imgsz", 320)
        self.declare_parameter("conf_thres", 0.45)
        self.declare_parameter("infer_hz", 6.0)

        self.declare_parameter("use_vad_gate", True)
        self.declare_parameter("voice_hold_sec", 3.0)

        self.declare_parameter("rgb_timeout_sec", 1.0)
        self.declare_parameter("depth_timeout_sec", 1.0)

        # bbox 내부 depth 계산용
        self.declare_parameter("depth_min_m", 0.35)
        self.declare_parameter("depth_max_m", 4.00)
        self.declare_parameter("min_depth_pixels", 80)

        # 사람 추종 거리
        self.declare_parameter("target_distance_m", 1.00)
        self.declare_parameter("distance_deadband_m", 0.18)
        self.declare_parameter("too_close_m", 0.55)

        # 화면 중심 제어
        self.declare_parameter("center_deadband_norm", 0.12)
        self.declare_parameter("forward_max_err_norm", 0.65)

        self.declare_parameter("forward_speed", 0.16)
        self.declare_parameter("turn_kp", 0.70)
        self.declare_parameter("min_turn_speed", 0.30)
        self.declare_parameter("max_turn_speed", 0.45)
        self.declare_parameter("invert_turn", False)

        # target smoothing
        self.declare_parameter("smooth_alpha", 0.55)

        self.rgb_topic = str(self.get_parameter("rgb_topic").value)
        self.depth_topic = str(self.get_parameter("depth_topic").value)
        self.vad_topic = str(self.get_parameter("vad_topic").value)
        self.cmd_topic = str(self.get_parameter("cmd_topic").value)
        self.state_topic = str(self.get_parameter("state_topic").value)

        self.model_path = str(self.get_parameter("model_path").value)
        self.device = str(self.get_parameter("device").value)
        self.imgsz = int(self.get_parameter("imgsz").value)
        self.conf_thres = float(self.get_parameter("conf_thres").value)
        self.infer_hz = float(self.get_parameter("infer_hz").value)

        self.use_vad_gate = bool(self.get_parameter("use_vad_gate").value)
        self.voice_hold_sec = float(self.get_parameter("voice_hold_sec").value)

        self.rgb_timeout_sec = float(self.get_parameter("rgb_timeout_sec").value)
        self.depth_timeout_sec = float(self.get_parameter("depth_timeout_sec").value)

        self.depth_min_m = float(self.get_parameter("depth_min_m").value)
        self.depth_max_m = float(self.get_parameter("depth_max_m").value)
        self.min_depth_pixels = int(self.get_parameter("min_depth_pixels").value)

        self.target_distance_m = float(self.get_parameter("target_distance_m").value)
        self.distance_deadband_m = float(self.get_parameter("distance_deadband_m").value)
        self.too_close_m = float(self.get_parameter("too_close_m").value)

        self.center_deadband_norm = float(self.get_parameter("center_deadband_norm").value)
        self.forward_max_err_norm = float(self.get_parameter("forward_max_err_norm").value)

        self.forward_speed = float(self.get_parameter("forward_speed").value)
        self.turn_kp = float(self.get_parameter("turn_kp").value)
        self.min_turn_speed = float(self.get_parameter("min_turn_speed").value)
        self.max_turn_speed = float(self.get_parameter("max_turn_speed").value)
        self.invert_turn = bool(self.get_parameter("invert_turn").value)

        self.smooth_alpha = float(self.get_parameter("smooth_alpha").value)

        self.last_rgb = None
        self.last_rgb_stamp = 0.0
        self.last_rgb_encoding = ""

        self.last_depth = None
        self.last_depth_stamp = 0.0
        self.last_depth_encoding = ""

        self.vad = False
        self.last_voice_time = 0.0

        self.smooth_cx = None
        self.smooth_z = None
        self.last_box = None

        self.last_state_log_time = 0.0
        self.last_mode = None

        self.cmd_pub = self.create_publisher(Twist, self.cmd_topic, 10)
        self.state_pub = self.create_publisher(String, self.state_topic, 10)

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )

        self.create_subscription(Image, self.rgb_topic, self.on_rgb, sensor_qos)
        self.create_subscription(Image, self.depth_topic, self.on_depth, sensor_qos)
        self.create_subscription(Bool, self.vad_topic, self.on_vad, 10)

        self.get_logger().info(f"loading YOLO model: {self.model_path}")
        try:
            from ultralytics import YOLO
            self.model = YOLO(self.model_path)
        except Exception as e:
            raise RuntimeError(f"failed to load YOLO model {self.model_path}: {repr(e)}")

        period = 1.0 / max(self.infer_hz, 1.0)
        self.timer = self.create_timer(period, self.on_timer)

        self.get_logger().info(
            "camera person follow started: "
            f"rgb={self.rgb_topic}, depth={self.depth_topic}, vad={self.vad_topic}, "
            f"cmd={self.cmd_topic}, model={self.model_path}, device={self.device}"
        )

    def on_vad(self, msg: Bool):
        self.vad = bool(msg.data)
        if self.vad:
            self.last_voice_time = time.time()

    def on_rgb(self, msg: Image):
        try:
            rgb = self.decode_rgb(msg)
        except Exception as e:
            self.get_logger().warn(f"failed to decode rgb: {repr(e)}")
            return

        self.last_rgb = rgb
        self.last_rgb_stamp = time.time()
        self.last_rgb_encoding = msg.encoding

    def on_depth(self, msg: Image):
        try:
            depth = self.decode_depth(msg)
        except Exception as e:
            self.get_logger().warn(f"failed to decode depth: {repr(e)}")
            return

        self.last_depth = depth
        self.last_depth_stamp = time.time()
        self.last_depth_encoding = msg.encoding

    def decode_rgb(self, msg: Image):
        h = int(msg.height)
        w = int(msg.width)
        step = int(msg.step)
        enc = msg.encoding.lower()

        raw = np.frombuffer(msg.data, dtype=np.uint8).reshape(h, step)

        if enc == "rgb8":
            arr = raw[:, :w * 3].reshape(h, w, 3).copy()
            return arr

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

    def publish_stop(self, reason):
        self.cmd_pub.publish(Twist())
        self.publish_state(reason)

    def publish_state(self, text):
        msg = String()
        msg.data = text
        self.state_pub.publish(msg)

        now = time.time()
        mode = text.split(" ", 1)[0]

        if mode != self.last_mode or now - self.last_state_log_time > 1.0:
            self.get_logger().info(text)
            self.last_mode = mode
            self.last_state_log_time = now

    def depth_for_box(self, depth, box):
        h, w = depth.shape
        x1, y1, x2, y2 = box

        x1 = int(clamp(x1, 0, w - 1))
        x2 = int(clamp(x2, 0, w - 1))
        y1 = int(clamp(y1, 0, h - 1))
        y2 = int(clamp(y2, 0, h - 1))

        if x2 <= x1 or y2 <= y1:
            return None, 0

        bw = x2 - x1
        bh = y2 - y1

        # bbox 경계는 배경이 섞이기 쉬우므로 중앙부만 사용
        ix1 = int(x1 + 0.25 * bw)
        ix2 = int(x1 + 0.75 * bw)
        iy1 = int(y1 + 0.15 * bh)
        iy2 = int(y1 + 0.90 * bh)

        roi = depth[iy1:iy2, ix1:ix2]

        valid = np.isfinite(roi)
        valid &= roi >= self.depth_min_m
        valid &= roi <= self.depth_max_m

        count = int(np.count_nonzero(valid))
        if count < self.min_depth_pixels:
            return None, count

        z = float(np.median(roi[valid]))
        return z, count

    def choose_person(self, detections, depth, img_w, img_h):
        persons = []

        for det in detections:
            box = det["box"]
            conf = det["conf"]

            x1, y1, x2, y2 = box
            bw = max(0.0, x2 - x1)
            bh = max(0.0, y2 - y1)
            area = bw * bh

            if area <= 1.0:
                continue

            z, depth_count = self.depth_for_box(depth, box)

            cx = 0.5 * (x1 + x2)
            cy = 0.5 * (y1 + y2)

            center_prior = 1.0 - min(1.0, abs(cx - img_w * 0.5) / max(img_w * 0.5, 1.0))
            area_norm = area / float(img_w * img_h)

            # depth가 있으면 가까운 사람 우선.
            if z is not None:
                z_score = 1.0 / max(z * z, 0.05)
            else:
                z_score = 0.2

            continuity = 0.0
            if self.last_box is not None:
                lx1, ly1, lx2, ly2 = self.last_box
                lcx = 0.5 * (lx1 + lx2)
                lcy = 0.5 * (ly1 + ly2)
                dist = math.hypot((cx - lcx) / img_w, (cy - lcy) / img_h)
                continuity = max(0.0, 1.0 - dist)

            score = (
                2.0 * conf
                + 3.0 * area_norm
                + 1.5 * center_prior
                + 1.0 * z_score
                + 1.5 * continuity
            )

            persons.append(
                {
                    "box": box,
                    "conf": conf,
                    "cx": cx,
                    "cy": cy,
                    "z": z,
                    "depth_count": depth_count,
                    "score": score,
                }
            )

        if not persons:
            return None

        persons.sort(key=lambda p: p["score"], reverse=True)
        return persons[0]

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
            self.get_logger().warn(f"YOLO predict failed on device={self.device}: {repr(e)}")
            if self.device != "cpu":
                self.get_logger().warn("retrying YOLO on cpu")
                results = self.model.predict(
                    source=rgb,
                    imgsz=self.imgsz,
                    conf=self.conf_thres,
                    device="cpu",
                    verbose=False,
                )
            else:
                return []

        if not results:
            return []

        r = results[0]

        if r.boxes is None:
            return []

        boxes = r.boxes
        detections = []

        for b in boxes:
            try:
                cls_id = int(b.cls.item())
                conf = float(b.conf.item())
                xyxy = b.xyxy.cpu().numpy().reshape(-1).tolist()
            except Exception:
                continue

            # COCO person class = 0
            if cls_id != 0:
                continue

            if conf < self.conf_thres:
                continue

            detections.append(
                {
                    "box": [float(v) for v in xyxy],
                    "conf": conf,
                }
            )

        return detections

    def smooth_target(self, cx, z):
        a = clamp(self.smooth_alpha, 0.0, 1.0)

        if self.smooth_cx is None:
            self.smooth_cx = cx
            self.smooth_z = z
        else:
            self.smooth_cx = a * cx + (1.0 - a) * self.smooth_cx
            if z is not None and self.smooth_z is not None:
                self.smooth_z = a * z + (1.0 - a) * self.smooth_z
            elif z is not None:
                self.smooth_z = z

        return self.smooth_cx, self.smooth_z

    def make_turn(self, err_norm):
        signed_err = err_norm

        if self.invert_turn:
            signed_err = -signed_err

        # target이 오른쪽이면 err_norm > 0.
        # ROS angular.z 기준 오른쪽 회전은 음수로 사용.
        wz = -self.turn_kp * signed_err

        if abs(wz) > 1e-6:
            wz_abs = clamp(abs(wz), self.min_turn_speed, self.max_turn_speed)
            wz = math.copysign(wz_abs, wz)

        return wz

    def on_timer(self):
        now = time.time()

        if self.use_vad_gate:
            voice_active = (now - self.last_voice_time) <= self.voice_hold_sec
            if not voice_active:
                self.smooth_cx = None
                self.smooth_z = None
                self.last_box = None
                self.publish_stop("WAIT_VOICE")
                return

        if self.last_rgb is None:
            self.publish_stop("NO_RGB")
            return

        if self.last_depth is None:
            self.publish_stop("NO_DEPTH")
            return

        if now - self.last_rgb_stamp > self.rgb_timeout_sec:
            self.publish_stop("RGB_TIMEOUT")
            return

        if now - self.last_depth_stamp > self.depth_timeout_sec:
            self.publish_stop("DEPTH_TIMEOUT")
            return

        rgb = self.last_rgb.copy()
        depth = self.last_depth

        img_h, img_w = rgb.shape[:2]

        detections = self.run_yolo(rgb)

        if not detections:
            self.smooth_cx = None
            self.smooth_z = None
            self.last_box = None
            self.publish_stop("NO_PERSON")
            return

        person = self.choose_person(detections, depth, img_w, img_h)

        if person is None:
            self.smooth_cx = None
            self.smooth_z = None
            self.last_box = None
            self.publish_stop(f"NO_VALID_PERSON dets={len(detections)}")
            return

        cx, z = self.smooth_target(person["cx"], person["z"])
        self.last_box = person["box"]

        err_norm = (cx - img_w * 0.5) / max(img_w * 0.5, 1.0)
        abs_err = abs(err_norm)

        cmd = Twist()

        # 방향 보정은 항상 가능
        if abs_err > self.center_deadband_norm:
            cmd.angular.z = self.make_turn(err_norm)

        # 전진은 사람이 보이고, 너무 가깝지 않고, FOV 안에 어느 정도 있을 때만
        z_for_control = z

        if z_for_control is None:
            # depth가 없으면 방향만 맞추고 전진 금지
            cmd.linear.x = 0.0
            mode = "TURN_NO_DEPTH" if abs_err > self.center_deadband_norm else "PERSON_NO_DEPTH"
        elif z_for_control <= self.too_close_m:
            cmd = Twist()
            mode = "TOO_CLOSE"
        else:
            if (
                z_for_control > self.target_distance_m + self.distance_deadband_m
                and abs_err <= self.forward_max_err_norm
            ):
                cmd.linear.x = self.forward_speed
            else:
                cmd.linear.x = 0.0

            if cmd.linear.x > 0.0 and abs(cmd.angular.z) > 0.0:
                mode = "FORWARD_STEER_PERSON"
            elif cmd.linear.x > 0.0:
                mode = "FORWARD_PERSON"
            elif abs(cmd.angular.z) > 0.0:
                mode = "TURN_PERSON"
            else:
                mode = "HOLD_PERSON"

        # q/e 금지
        cmd.linear.y = 0.0

        self.cmd_pub.publish(cmd)

        x1, y1, x2, y2 = person["box"]
        self.publish_state(
            f"{mode} "
            f"persons={len(detections)} "
            f"conf={person['conf']:.2f} "
            f"bbox=({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}) "
            f"cx={cx:.1f}/{img_w} "
            f"z={-1.0 if z is None else z:.2f} "
            f"depth_n={person['depth_count']} "
            f"err={err_norm:.2f} "
            f"vx={cmd.linear.x:.3f} "
            f"wz={cmd.angular.z:.3f}"
        )


def main(args=None):
    rclpy.init(args=args)
    node = CameraPersonFollowNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.cmd_pub.publish(Twist())
        except Exception:
            pass
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
