#!/usr/bin/env python
import rospy
import cv2 as cv
import math as m
import time as t
from hardwired.msg import state
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from cv_bridge import CvBridge, CvBridgeError

class ROS_NODE:
	def __init__(self):
		rospy.init_node('display', anonymous=True)
		self.bridge = CvBridge()
		rospy.Subscriber('vector_space', Image, self.vector_space_related_cb)
		rospy.Subscriber('state', state, self.state_related_cb)
		rospy.Subscriber('condition', Bool, self.cond_related_cb)
		self.cond = Bool()
		self.cond.data = False
	def vector_space_related_cb(self, data):
		self.vector_space_msg = data
	def state_related_cb(self, data):
		self.state_msg = data
	def cond_related_cb(self, data):
		self.cond = data

node = ROS_NODE()
while not rospy.is_shutdown():
	if node.cond.data==True:
		vector_space = node.bridge.imgmsg_to_cv2(node.vector_space_msg, desired_encoding="passthrough")
		x, y, yaw = node.state_msg.x, node.state_msg.y, node.state_msg.yaw
		dis = 12
		n_x = int(x+dis*m.cos(yaw))
		n_y = int(y-dis*m.sin(yaw))
		vector_space = cv.arrowedLine(vector_space, (x,y), (n_x,n_y), (0,255,0), 2, tipLength = 0.3)
		vector_space = cv.resize(vector_space, (600, 600))
		cv.imshow("vector_space", vector_space)
		cv.waitKey(1)
	elif node.cond.data==False:
		cv.destroyAllWindows()
		print("waiting for vector_space")
		t.sleep(2)