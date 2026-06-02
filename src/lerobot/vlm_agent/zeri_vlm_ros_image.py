# zeri_vlm_ros_image.py
import copy
from typing import Optional

import cv2
import numpy as np
from PIL import Image as PILImage
from sensor_msgs.msg import Image as RosImage


def ros_image_to_bgr(msg: RosImage) -> Optional[np.ndarray]:
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

    except Exception:
        return None

    return None


def ros_image_to_pil_rgb(msg: RosImage) -> Optional[PILImage.Image]:
    bgr = ros_image_to_bgr(msg)

    if bgr is None:
        return None

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return PILImage.fromarray(rgb)


def pil_rgb_to_ros_image(
    image: PILImage.Image,
    stamp,
    frame_id: str = "zeri_vlm_input_rgb",
) -> RosImage:
    if image.mode != "RGB":
        image = image.convert("RGB")

    arr = np.asarray(image, dtype=np.uint8)
    arr = np.ascontiguousarray(arr)

    msg = RosImage()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.height = int(arr.shape[0])
    msg.width = int(arr.shape[1])
    msg.encoding = "rgb8"
    msg.is_bigendian = False
    msg.step = int(arr.shape[1] * 3)
    msg.data = arr.tobytes()

    return msg


def clone_depth_snapshot_msg(
    msg: RosImage,
    stamp,
    frame_id: str = "zeri_vlm_input_depth",
) -> RosImage:
    cloned = copy.deepcopy(msg)
    cloned.header.stamp = stamp
    cloned.header.frame_id = frame_id
    return cloned
