# Created by Chat-GPT (https://chat.openai.com/)
# using following prompt: write python code for ROS2 subscriber node for video

import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class VideoSubscriberNode(Node):
    def __init__(self):
        super().__init__('video_subscriber')
        self.subscription_ = self.create_subscription(
            Image,
            'image_raw',
            self.process_video,
            10
        )
        self.cv_bridge = CvBridge()

    def process_video(self, msg):
        try:
            # Convert the received image message to an OpenCV format
            frame = self.cv_bridge.imgmsg_to_cv2(msg)

            # Display the frame
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imshow('Video', frame_bgr)
            cv2.waitKey(1)
        except Exception as e:
            self.get_logger().error(f'Error processing video frame: {str(e)}')

def main(args=None):
    rclpy.init(args=args)
    video_subscriber = VideoSubscriberNode()
    rclpy.spin(video_subscriber)

    video_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
