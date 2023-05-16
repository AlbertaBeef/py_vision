import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class VideoSubscriber(Node):
    def __init__(self):
        super().__init__('usbcam_subscriber')
        self.subscription_ = self.create_subscription(Image, 'usb_camera/image', self.callback, 10)
        self.cv_bridge_ = CvBridge()

    def callback(self, msg):
        # Convert the ROS2 message to an OpenCV image
        frame = self.cv_bridge_.imgmsg_to_cv2(msg, 'bgr8')

        # Display the image
        cv2.imshow('Video Stream', frame)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = VideoSubscriber()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()