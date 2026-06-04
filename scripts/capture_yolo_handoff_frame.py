#!/usr/bin/env python3
import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


def ros_image_to_bgr(msg):
    h = int(msg.height)
    w = int(msg.width)
    step = int(msg.step)
    enc = str(msg.encoding).lower()

    raw = np.frombuffer(bytes(msg.data), dtype=np.uint8)

    def rows_view(channels):
        arr = raw.reshape(h, step)
        arr = arr[:, : w * channels]
        return arr.reshape(h, w, channels)

    if enc == "bgr8":
        return rows_view(3).copy(), enc

    if enc == "rgb8":
        rgb = rows_view(3)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), enc

    if enc == "bgra8":
        bgra = rows_view(4)
        return cv2.cvtColor(bgra, cv2.COLOR_BGRA2BGR), enc

    if enc == "rgba8":
        rgba = rows_view(4)
        return cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR), enc

    if enc in ("mono8", "8uc1"):
        arr = raw.reshape(h, step)[:, :w].reshape(h, w)
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR), enc

    raise RuntimeError(f"Unsupported ROS image encoding: {enc}, size={w}x{h}, step={step}")


def capture_ros_image(topic: str, timeout_sec: float):
    import rclpy
    from sensor_msgs.msg import Image

    rclpy.init(args=None)
    node = rclpy.create_node("capture_yolo_handoff_frame_once")

    box = {"msg": None}

    def cb(msg):
        if box["msg"] is None:
            box["msg"] = msg

    sub = node.create_subscription(Image, topic, cb, 1)

    print(f"[ROS] waiting image: {topic}")
    deadline = time.time() + timeout_sec

    try:
        while rclpy.ok() and box["msg"] is None and time.time() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
    finally:
        node.destroy_subscription(sub)
        node.destroy_node()
        rclpy.shutdown()

    if box["msg"] is None:
        raise TimeoutError(f"No image received from {topic} within {timeout_sec:.1f}s")

    bgr, enc = ros_image_to_bgr(box["msg"])
    return bgr, {
        "source": "ros",
        "topic": topic,
        "encoding": enc,
        "width": int(box["msg"].width),
        "height": int(box["msg"].height),
        "step": int(box["msg"].step),
    }


def capture_camera(camera: str):
    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        # allow numeric camera index fallback
        try:
            cap = cv2.VideoCapture(int(camera))
        except Exception:
            pass

    if not cap.isOpened():
        raise RuntimeError(f"Failed to open camera: {camera}")

    ok, frame = cap.read()
    cap.release()

    if not ok or frame is None:
        raise RuntimeError(f"Failed to read frame from camera: {camera}")

    return frame, {
        "source": "camera",
        "camera": camera,
        "width": int(frame.shape[1]),
        "height": int(frame.shape[0]),
        "encoding": "bgr8",
    }


def draw_yolo(model, bgr, conf: float, imgsz: int, device: str):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    results = model.predict(
        source=rgb,
        conf=conf,
        imgsz=imgsz,
        device=device,
        verbose=False,
    )

    drawn = bgr.copy()
    detections = []

    if not results or results[0].boxes is None:
        return drawn, detections

    names = model.names

    for i, box in enumerate(results[0].boxes):
        xyxy = box.xyxy[0].detach().cpu().numpy().astype(float).tolist()
        score = float(box.conf[0].detach().cpu().item()) if box.conf is not None else 0.0
        cls_id = int(box.cls[0].detach().cpu().item()) if box.cls is not None else -1

        if isinstance(names, dict):
            cls_name = names.get(cls_id, str(cls_id))
        else:
            cls_name = str(cls_id)

        x1, y1, x2, y2 = [int(round(v)) for v in xyxy]

        cv2.rectangle(drawn, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{cls_name} {score:.2f}"
        cv2.putText(
            drawn,
            label,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        detections.append(
            {
                "index": i,
                "class_id": cls_id,
                "class_name": cls_name,
                "confidence": score,
                "xyxy": xyxy,
                "width": x2 - x1,
                "height": y2 - y1,
                "center_xy": [(x1 + x2) / 2.0, (y1 + y2) / 2.0],
            }
        )

    return drawn, detections


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["ros", "camera"], default="ros")
    parser.add_argument("--topic", default="/zeri/vla/left/handoff_image")
    parser.add_argument("--camera", default="/dev/cam_left")
    parser.add_argument("--model", default=str(Path.home() / "ze-ri/models/hand_yolo.pt"))
    parser.add_argument("--device", default="0")
    parser.add_argument("--conf", type=float, default=0.01)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--timeout-sec", type=float, default=10.0)
    parser.add_argument("--out-dir", default=str(Path.home() / "ze-ri/debug/yolo_handoff"))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_path = out_dir / f"{ts}_raw.jpg"
    result_path = out_dir / f"{ts}_yolo.jpg"
    json_path = out_dir / f"{ts}_result.json"

    if args.mode == "ros":
        bgr, meta = capture_ros_image(args.topic, args.timeout_sec)
    else:
        bgr, meta = capture_camera(args.camera)

    cv2.imwrite(str(raw_path), bgr)

    model = YOLO(args.model)
    drawn, detections = draw_yolo(
        model=model,
        bgr=bgr,
        conf=args.conf,
        imgsz=args.imgsz,
        device=args.device,
    )

    cv2.imwrite(str(result_path), drawn)

    payload = {
        "timestamp": ts,
        "model": args.model,
        "model_names": model.names,
        "conf": args.conf,
        "imgsz": args.imgsz,
        "device": args.device,
        "capture": meta,
        "num_detections": len(detections),
        "detections": detections,
        "raw_image": str(raw_path),
        "result_image": str(result_path),
    }

    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))

    print()
    print("========== YOLO HANDOFF CAPTURE ==========")
    print("raw_image:", raw_path)
    print("result_image:", result_path)
    print("result_json:", json_path)
    print("num_detections:", len(detections))
    for d in detections:
        print(
            f"[{d['index']}] {d['class_name']} "
            f"conf={d['confidence']:.4f} "
            f"xyxy={[round(x, 1) for x in d['xyxy']]}"
        )
    print("==========================================")
    print()


if __name__ == "__main__":
    main()
