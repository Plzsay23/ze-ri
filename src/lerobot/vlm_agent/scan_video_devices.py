# scan_video_devices.py

import cv2
from pathlib import Path


def test_camera(dev_path: str, width: int = 640, height: int = 480) -> bool:
    cap = cv2.VideoCapture(dev_path, cv2.CAP_V4L2)

    if not cap.isOpened():
        print(f"[FAIL] {dev_path}: open failed")
        return False

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    ok = False
    frame = None

    for _ in range(10):
        ret, frame = cap.read()
        if ret and frame is not None:
            ok = True
            break

    if not ok:
        print(f"[FAIL] {dev_path}: read failed")
        cap.release()
        return False

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    out_path = f"preview_{Path(dev_path).name}.jpg"
    cv2.imwrite(out_path, frame)

    print(f"[OK] {dev_path}")
    print(f"     resolution: {actual_w}x{actual_h}")
    print(f"     fps: {fps}")
    print(f"     preview: {out_path}")

    cap.release()
    return True


def main():
    devices = sorted(Path("/dev").glob("video*"))

    if not devices:
        print("No /dev/video* devices found.")
        return

    for dev in devices:
        test_camera(str(dev))


if __name__ == "__main__":
    main()