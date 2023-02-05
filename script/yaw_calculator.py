#!/usr/bin/env python
import rospy
import time as t
import math as m
from std_msgs.msg import Bool
from std_msgs.msg import Float32
from hardwired.msg import state
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

class ROS_NODE:
   def __init__(self):
      rospy.init_node('yaw_calculator', anonymous=True)
      self.pub = rospy.Publisher('yaw', Float32, queue_size=1)
      self.bridge = CvBridge()
      rospy.Subscriber('state', state, self.state_related_cb)
      rospy.Subscriber('vector_map', Image, self.vector_map_related_cb)
      rospy.Subscriber('condition', Bool, self.cond_related_cb)
      self.cond = Bool()
      self.cond.data = False
   def vector_map_related_cb(self, data):
      self.vector_map_msg = data
      self.vector_map_detected = True
   def state_related_cb(self, data):
      self.state_msg = data
      self.state_detected = True
   def cond_related_cb(self, data):
      self.cond = data

node = ROS_NODE()
yaw_msg = Float32()
yaw_msg.data = rospy.get_param("yaw_calculator/default_yaw")
while not rospy.is_shutdown():
   if node.cond.data==True:
      vector_map = node.bridge.imgmsg_to_cv2(node.vector_map_msg, desired_encoding="passthrough")
      x, y, actual_yaw = node.state_msg.x, node.state_msg.y, node.state_msg.yaw
      desired_yaw = vector_map[y,x]
      yaw_msg.data = (desired_yaw - actual_yaw)*180/m.pi
      node.pub.publish(yaw_msg)
   elif node.cond.data==False:
      print("waiting for vector_space")
      t.sleep(2)