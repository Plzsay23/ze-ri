#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
import cv2


class DepthVizNode(Node):
    def __init__(self):
        super().__init__('depth_viz_node')
        self.bridge = CvBridge()

        self.declare_parameter('input_topic', '/camera/camera/depth/image_rect_raw')
        self.declare_parameter('output_topic', '/camera/camera/depth_viz')
        self.declare_parameter('min_m', 0.2)
        self.declare_parameter('max_m', 4.0)

        input_topic = self.get_parameter('input_topic').value
        output_topic = self.get_parameter('output_topic').value
        self.min_m = float(self.get_parameter('min_m').value)
        self.max_m = float(self.get_parameter('max_m').value)

        self.sub = self.create_subscription(Image, input_topic, self.cb, 10)
        self.pub = self.create_publisher(Image, output_topic, 10)

        self.get_logger().info(f'input_topic={input_topic}')
        self.get_logger().info(f'output_topic={output_topic}')
        self.get_logger().info(f'min_m={self.min_m}, max_m={self.max_m}')

    def cb(self, msg: Image):
        try:
            depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except Exception as e:
            self.get_logger().error(f'cv_bridge error: {e}')
            return

        depth = np.array(depth)

        # uint16(Z16)면 보통 mm 단위인 경우가 많음
        if depth.dtype == np.uint16:
            depth_m = depth.astype(np.float32) / 1000.0
        else:
            depth_m = depth.astype(np.float32)

        # 유효 범위만 사용
        valid = np.isfinite(depth_m) & (depth_m > 0.0)
        if not np.any(valid):
            viz = np.zeros((depth.shape[0], depth.shape[1], 3), dtype=np.uint8)
            out = self.bridge.cv2_to_imgmsg(viz, encoding='bgr8')
            out.header = msg.header
            self.pub.publish(out)
            return

        clipped = np.clip(depth_m, self.min_m, self.max_m)

        # 가까울수록 밝게 보이게 반전 정규화
        norm = (self.max_m - clipped) / max(self.max_m - self.min_m, 1e-6)
        norm[~valid] = 0.0
        gray = (norm * 255.0).astype(np.uint8)

        color = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
        color[~valid] = (0, 0, 0)

        out = self.bridge.cv2_to_imgmsg(color, encoding='bgr8')
        out.header = msg.header
        self.pub.publish(out)


def main():
    rclpy.init()
    node = DepthVizNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
