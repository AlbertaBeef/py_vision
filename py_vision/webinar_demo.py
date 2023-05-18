# Copyright 2023 Avnet, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import rclpy
from rclpy.node import Node

#from std_msgs.msg import String
from geometry_msgs.msg import Twist

from std_srvs.srv import Empty

from sensor_msgs.msg import Image
# import CV BRIDGE
from cv_bridge import CvBridge, CvBridgeError
import cv2

import numpy as np
from ctypes import *
from typing import List
import xir
import pathlib
import vart
import time
import sys
import argparse
import glob
import subprocess
import re

# Vitis-AI implementation of ASL model

def get_subgraph (g):
    sub = []
    root = g.get_root_subgraph()
    sub = [ s for s in root.toposort_child_subgraph()
            if s.has_attr("device") and s.get_attr("device").upper() == "DPU"]
    return sub

def get_child_subgraph_dpu(graph: "Graph") -> List["Subgraph"]:
    assert graph is not None, "'graph' should not be None."
    root_subgraph = graph.get_root_subgraph()
    assert (root_subgraph is not None), "Failed to get root subgraph of input Graph object."
    if root_subgraph.is_leaf:
        return []
    child_subgraphs = root_subgraph.toposort_child_subgraph()
    assert child_subgraphs is not None and len(child_subgraphs) > 0
    return [
        cs
        for cs in child_subgraphs
        if cs.has_attr("device") and cs.get_attr("device").upper() == "DPU"
    ]


def get_dpu_architecture():
    dpu_architecture = ''
    proc = subprocess.run(['xdputil','query'], capture_output=True, encoding='utf8')
    for line in proc.stdout.splitlines():
        if line.find("DPU Arch") > 0 :
           dpu_architecture = re.search('(.+?)_B(.+?)_(.+?)',line).group(2)
           dpu_architecture = "B"+dpu_architecture
           return dpu_architecture


class WebinarDemo(Node):

    def __init__(self):
        super().__init__('webinar_demo')
        self.subscriber_ = self.create_subscription(Image,'image_raw',self.listener_callback,10)
        self.subscriber_  # prevent unused variable warning
        self.publisher1 = self.create_publisher(Image, 'vision/asl', 10)
        self.publisher2 = self.create_publisher(Twist, 'turtle1/cmd_vel', 10)
        # Open ASL model
        self.id_to_class = {
          0 :"A",
          1 :"B",
          2 :"C",
          3 :"D",
          4 :"E",
          5 :"F",
          6 :"G",
          7 :"H",
          8 :"I",
          9 :"J",
          10:"K",
          11:"L",
          12:"M",
          13:"N",
          14:"O",
          15:"P",
          16:"Q",
          17:"R",
          18:"S",
          19:"T",
          20:"U",
          21:"V",
          22:"W",
          23:"X",
          24:"Y",
          25:"Z",
          26:"{del}",
          27:"{nothing}",
          28:"{space}"
        }
        # Parameters (for text overlay)
        self.scale = 1.0
        self.text_fontType = cv2.FONT_HERSHEY_SIMPLEX
        self.text_fontSize = 0.75*self.scale
        self.text_color    = (255,0,0)
        self.text_lineSize = max( 1, int(2*self.scale) )
        self.text_lineType = cv2.LINE_AA
        self.text_x = int(10*self.scale)
        self.text_y = int(30*self.scale)
        
        # Determine DPU architecture and Vitis-AI .xmodel file
        self.dpu_architecture = get_dpu_architecture() 
        self.get_logger().info("DPU="+self.dpu_architecture)
        self.model_path = '/home/root/asl_classification_vitis_ai/model_mobilenetv2/'+self.dpu_architecture+'/asl_classifier.xmodel'
        self.get_logger().info("MODEL="+self.model_path)
        
        # Create DPU runner
        self.g = xir.Graph.deserialize(self.model_path)
        self.subgraphs = get_child_subgraph_dpu(self.g)
        assert len(self.subgraphs) == 1 # only one DPU kernel
        self.dpu = vart.Runner.create_runner(self.subgraphs[0], "run")
        # input scaling
        self.input_fixpos = self.dpu.get_input_tensors()[0].get_attr("fix_point")
        self.input_scale = 2**self.input_fixpos
        print('[INFO] input_fixpos=',self.input_fixpos,' input_scale=',self.input_scale)

        # Get input/output tensors
        self.inputTensors = self.dpu.get_input_tensors()
        self.outputTensors = self.dpu.get_output_tensors()
        self.inputShape = self.inputTensors[0].dims
        self.outputShape = self.outputTensors[0].dims

    def listener_callback(self, msg):
        bridge = CvBridge()
        cv2_image = bridge.imgmsg_to_cv2(msg,desired_encoding = "rgb8")

        # 224x224 ROI for classification
        y1 = (128)
        y2 = (128+224)
        x1 = (208)
        x2 = (208+224)
        roi_img = cv2_image[ y1:y2, x1:x2, : ]
                
        cv2.rectangle(cv2_image, (x1,y1), (x2,y2), (0, 255, 0), 2)
        #cv2_image = cv2.flip(cv2_image,1) # horizontal flip for ease of use

        # ASL pre-processing
        asl_img = cv2.cvtColor(roi_img, cv2.COLOR_BGR2RGB)
        asl_img = asl_img / 255.0
        asl_x = []
        asl_x.append( asl_img )
        asl_x = np.array(asl_x)

        """ Prepare input/output buffers """
        inputData = []
        inputData.append(np.empty((self.inputShape),dtype=np.float32,order='C'))
        inputImage = inputData[0]
        inputImage[0,...] = asl_img

        outputData = []
        outputData.append(np.empty((self.outputShape),dtype=np.int8,order='C'))

        """ Execute model on DPU """
        job_id = self.dpu.execute_async( inputData, outputData )
        self.dpu.wait(job_id)

        # ASL post-processing
        OutputData = outputData[0].reshape(1,29)
        asl_y = np.reshape( OutputData, (-1,29) )
        asl_id  = np.argmax(asl_y[0])
        asl_sign = self.id_to_class[asl_id]

        asl_text = '['+str(asl_id)+']='+asl_sign
        cv2.putText(cv2_image,asl_text,
        	(self.text_x,self.text_y),
        	self.text_fontType,self.text_fontSize,
        	self.text_color,self.text_lineSize,self.text_lineType)
        
        self.actionDetected = ""
        if asl_sign == 'A':
          self.actionDetected = "A : Advance"
          # Create message to backup (+ve value on x axis)
          msg = Twist()
          msg.linear.x = 2.0
          msg.linear.y = 0.0
          msg.linear.z = 0.0
          msg.angular.x = 0.0
          msg.angular.y = 0.0
          msg.angular.z = 0.0
          self.publisher2.publish(msg)
        if asl_sign == 'B':
          self.actionDetected = "B : Back-Up"
          # Create message to backup (-ve value on x axis)
          msg = Twist()
          msg.linear.x = -2.0
          msg.linear.y = 0.0
          msg.linear.z = 0.0
          msg.angular.x = 0.0
          msg.angular.y = 0.0
          msg.angular.z = 0.0
          self.publisher2.publish(msg)
        if asl_sign == 'L':
          self.actionDetected = "L : Turn Left"
          # Create message to turn left (+ve value on z axis)
          msg = Twist()
          msg.linear.x = 0.0
          msg.linear.y = 0.0
          msg.linear.z = 0.0
          msg.angular.x = 0.0
          msg.angular.y = 0.0
          msg.angular.z = 2.0
          self.publisher2.publish(msg)
        if asl_sign == 'R':
          self.actionDetected = "R : Turn Right"
          # Create message to turn in right (-ve value on z axis)
          msg = Twist()
          msg.linear.x = 0.0
          msg.linear.y = 0.0
          msg.linear.z = 0.0
          msg.angular.x = 0.0
          msg.angular.y = 0.0
          msg.angular.z = -2.0
          self.publisher2.publish(msg)
        if asl_sign == '{del}':
          self.actionDetected = "{del} : Reset Turtle"
          # Create message to reset turtlesim
          try:
              self.cli = self.create_client(Empty, 'reset')
              while not self.cli.wait_for_service(timeout_sec=1.0):
                 self.get_logger().info('[ERROR] service not available, throwing exception ...')
                 raise Exception('[ERROR] service not available')
              self.req = Empty.Request()
              self.future = self.cli.call_async(self.req)
              #rclpy.spin_until_future_complete(self, self.future)
              #ret = self.future.result()
          except:
              self.get_logger().info("Failed to reset turtlesim!")
              self.cli = None
        if asl_sign == '{space}':
          self.actionDetected = "{space} : Clear Lines"
          # Create message to clear turtlesim
          try:
              self.cli = self.create_client(Empty, 'clear')
              while not self.cli.wait_for_service(timeout_sec=1.0):
                 self.get_logger().info('[ERROR] service not available, throwing exception ...')
                 raise Exception('[ERROR] service not available')
              self.req = Empty.Request()
              self.future = self.cli.call_async(self.req)
              #rclpy.spin_until_future_complete(self, self.future)
              #ret = self.future.result()
          except:
              self.get_logger().info("Failed to clear turtlesim!")
              self.cli = None

        self.get_logger().info(self.actionDetected)
        cv2.putText(cv2_image,self.actionDetected,
          	(208,108-10),self.text_fontType,self.text_fontSize,
          	(0,255,0),self.text_lineSize,self.text_lineType)
        
        # DISPLAY
        cv2_bgr_image = cv2.cvtColor(cv2_image, cv2.COLOR_RGB2BGR)
        cv2.imshow('webinar_demo',cv2_bgr_image)
        cv2.waitKey(1)

        # CONVERT BACK TO ROS & PUBLISH
        image_ros = bridge.cv2_to_imgmsg(cv2_image)        
        self.publisher1.publish(image_ros)


def main(args=None):
    rclpy.init(args=args)

    webinar_demo = WebinarDemo()

    rclpy.spin(webinar_demo)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    webinar_demo.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
