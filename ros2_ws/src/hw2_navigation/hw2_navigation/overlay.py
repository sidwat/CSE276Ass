"""Image annotation only; does not estimate pose or control motion."""
import cv2
import numpy as np
from cv_bridge import CvBridge
from hw2_navigation.core import stamp_ns


class OverlayRenderer:
    def __init__(self):
        self.bridge = CvBridge()
        self.stamp = -1_000_000_000
        self.corners = []

    def update(self, detections):
        self.stamp = stamp_ns(detections.header.stamp)
        self.corners = [(d.id, [(p.x, p.y) for p in d.corners])
                        for d in detections.detections]

    def render(self, msg, reason):
        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8').copy()
        if abs(stamp_ns(msg.header.stamp)-self.stamp) < 100_000_000:
            for tag_id, points in self.corners:
                pts = np.asarray(points, dtype=np.int32)
                cv2.polylines(img, [pts], True, (0, 255, 0), 2)
                cv2.putText(img, str(tag_id), tuple(pts[0]),
                            cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 255, 0), 2)
        cv2.putText(img, reason, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    .5, (0, 200, 255), 1)
        result = self.bridge.cv2_to_imgmsg(img, encoding='bgr8')
        result.header = msg.header
        return result


def main(args=None):
    """Optional image debugging process; not part of pose processing or control."""
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from apriltag_msgs.msg import AprilTagDetectionArray
    from sensor_msgs.msg import Image
    from std_msgs.msg import String

    class OverlayNode(Node):
        def __init__(self):
            super().__init__('hw2_overlay')
            self.renderer = OverlayRenderer()
            self.reason = 'waiting_for_status'
            self.publisher = self.create_publisher(Image, '/hw2/overlay', 1)
            self.create_subscription(AprilTagDetectionArray, '/detections', self.renderer.update, 10)
            self.create_subscription(String, '/hw2/status', self.on_status, 10)
            self.create_subscription(Image, '/camera/image_rect', self.on_image, qos_profile_sensor_data)

        def on_status(self, msg):
            import json
            self.reason = json.loads(msg.data).get('reason', '')

        def on_image(self, msg):
            if self.publisher.get_subscription_count():
                self.publisher.publish(self.renderer.render(msg, self.reason))

    rclpy.init(args=args)
    node = OverlayNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
