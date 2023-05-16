import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class USBImagePublisher(Node):
    def __init__(self):
        super().__init__('usbcam_publisher')
        self.publisher_ = self.create_publisher(Image, 'usb_camera/image', 10)
        self.timer_ = self.create_timer(0.1, self.publish_image)
        self.bridge_ = CvBridge()
        
        # Open the camera
        self.cap = cv2.VideoCapture(0)

        # Check if the camera is opened correctly
        if not self.cap.isOpened():
            self.get_logger().error('Could not open USB camera')
            return

        # Set the resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)        

    def publish_image(self):
        # Read an image from the camera
        ret, frame = self.cap.read()

        # Convert the image to a ROS2 message
        msg = self.bridge_.cv2_to_imgmsg(frame, encoding='bgr8')
        msg.header.stamp = self.get_clock().now().to_msg()

        # Publish the message
        self.publisher_.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = USBImagePublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
