#!/usr/bin/env python
import rospy
import cv2 as cv
import math as m
import time as t
from sensor_msgs.msg import Image
from hardwired.msg import state
from cv_bridge import CvBridge, CvBridgeError

ARUCO_DICT = {
	"DICT_4X4_50": cv.aruco.DICT_4X4_50,
	"DICT_4X4_100": cv.aruco.DICT_4X4_100,
	"DICT_4X4_250": cv.aruco.DICT_4X4_250,
	"DICT_4X4_1000": cv.aruco.DICT_4X4_1000,
	"DICT_5X5_50": cv.aruco.DICT_5X5_50,
	"DICT_5X5_100": cv.aruco.DICT_5X5_100,
	"DICT_5X5_250": cv.aruco.DICT_5X5_250,
	"DICT_5X5_1000": cv.aruco.DICT_5X5_1000,
	"DICT_6X6_50": cv.aruco.DICT_6X6_50,
	"DICT_6X6_100": cv.aruco.DICT_6X6_100,
	"DICT_6X6_250": cv.aruco.DICT_6X6_250,
	"DICT_6X6_1000": cv.aruco.DICT_6X6_1000,
	"DICT_7X7_50": cv.aruco.DICT_7X7_50,
	"DICT_7X7_100": cv.aruco.DICT_7X7_100,
	"DICT_7X7_250": cv.aruco.DICT_7X7_250,
	"DICT_7X7_1000": cv.aruco.DICT_7X7_1000,
	"DICT_ARUCO_ORIGINAL": cv.aruco.DICT_ARUCO_ORIGINAL,
	"DICT_APRILTAG_16h5": cv.aruco.DICT_APRILTAG_16h5,
	"DICT_APRILTAG_25h9": cv.aruco.DICT_APRILTAG_25h9,
	"DICT_APRILTAG_36h10": cv.aruco.DICT_APRILTAG_36h10,
	"DICT_APRILTAG_36h11": cv.aruco.DICT_APRILTAG_36h11}
def aruco_vector(corners, ids, rejected):
	ids = ids.flatten()
	markerCorner, markerID = corners[0], ids[0]
	corners = markerCorner.reshape((4, 2))
	(top_left, top_right, bottom_right, bottom_left) = corners
	px, py = int((top_left[0]+bottom_right[0])/2.0), int((top_left[1]+bottom_right[1])/2.0)
	yaw = m.atan2((bottom_left[1]-top_left[1]), top_left[0]-bottom_left[0])
	return [px, py], yaw
class ROS_NODE:
	def __init__(self):
		rospy.init_node('state_finder', anonymous=True)
		self.pub = rospy.Publisher('state', state, queue_size=1)
		self.bridge = CvBridge()
		rospy.Subscriber('image', Image, self.img_related_cb)
	def img_related_cb(self, data):
		self.img = self.bridge.imgmsg_to_cv2(data, desired_encoding="passthrough")

node = ROS_NODE()
aruco_type = rospy.get_param("state_finder/aruco_type", default="DICT_4X4_50")
arucoDict = cv.aruco.Dictionary_get(ARUCO_DICT[aruco_type])
arucoParams = cv.aruco.DetectorParameters_create()
state_msg = state()
state_msg.x, state_msg.y, state_msg.yaw = rospy.get_param("state_finder/default_px"), rospy.get_param("state_finder/default_py"), rospy.get_param("state_finder/default_yaw")
while not rospy.is_shutdown():
	try:
		corners, ids, rejected = cv.aruco.detectMarkers(node.img, arucoDict, parameters=arucoParams)
		pos, yaw = aruco_vector(corners, ids, rejected)
		state_msg.x, state_msg.y = pos
		state_msg.yaw = yaw
		print("Bot detected")
	except:
		print("Bot not detected")
	node.pub.publish(state_msg)