#!/usr/bin/env python
import rospy
import cv2 as cv
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

rospy.init_node('image_generator', anonymous=True)
bridge = CvBridge()
pub = rospy.Publisher('image', Image, queue_size=1)

cap = cv.VideoCapture(0)
while not rospy.is_shutdown():
	ret, img = cap.read()
	img = cv.resize(img, (224, 224))
	img_msg = bridge.cv2_to_imgmsg(img, encoding="passthrough")
	pub.publish(img_msg)
	cv.imshow("img", img)
	cv.waitKey(1)