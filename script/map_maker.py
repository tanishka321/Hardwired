#!/usr/bin/env python
import rospy
import os
import functools
import threading
import random
import colorsys
import tensorflow as tf
import cv2 as cv
import numpy as np
import time as t
import math as m
from cmath import inf
from hardwired.msg import state
from std_msgs.msg import Bool
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

INPUT_SHAPE = (416, 416)
def compose(*funcs):
   if funcs:
     return functools.reduce(lambda f, g: lambda *a, **kw: g(f(*a, **kw)), funcs)
   else:
     raise ValueError('Composition of empty sequence not supported.')
@functools.wraps(tf.keras.layers.Conv2D)
def DarknetConv2D(*args, **kwargs):
   darknet_conv_kwargs = {'kernel_regularizer': tf.keras.regularizers.l2(5e-4)}
   darknet_conv_kwargs['padding'] = 'valid' if kwargs.get('strides')==(2,2) else 'same'
   darknet_conv_kwargs.update(kwargs)
   return tf.keras.layers.Conv2D(*args, **darknet_conv_kwargs)
def DarknetConv2D_BN_Leaky(*args, **kwargs):
   no_bias_kwargs = {'use_bias': False}
   no_bias_kwargs.update(kwargs)
   return compose(
      DarknetConv2D(*args, **no_bias_kwargs),
      tf.keras.layers.BatchNormalization(),
      tf.keras.layers.LeakyReLU(alpha=0.1))
def resblock_body(x, num_filters, num_blocks):
   x = tf.keras.layers.ZeroPadding2D(((1,0),(1,0)))(x)
   x = DarknetConv2D_BN_Leaky(num_filters, (3,3), strides=(2,2))(x)
   for i in range(num_blocks):
      y = compose(
         DarknetConv2D_BN_Leaky(num_filters//2, (1,1)),
         DarknetConv2D_BN_Leaky(num_filters, (3,3)))(x)
      x = tf.keras.layers.Add()([x,y])
   return x
def darknet_body(x):
   x = DarknetConv2D_BN_Leaky(32, (3,3))(x)
   x = resblock_body(x, 64, 1)
   x = resblock_body(x, 128, 2)
   x = resblock_body(x, 256, 8)
   x = resblock_body(x, 512, 8)
   x = resblock_body(x, 1024, 4)
   return x
def make_last_layers(x, num_filters, out_filters):
   x = compose(
      DarknetConv2D_BN_Leaky(num_filters, (1,1)),
      DarknetConv2D_BN_Leaky(num_filters*2, (3,3)),
      DarknetConv2D_BN_Leaky(num_filters, (1,1)),
      DarknetConv2D_BN_Leaky(num_filters*2, (3,3)),
      DarknetConv2D_BN_Leaky(num_filters, (1,1)))(x)
   y = compose(
      DarknetConv2D_BN_Leaky(num_filters*2, (3,3)),
      DarknetConv2D(out_filters, (1,1)))(x)
   return x, y
def yolo_body(inputs, num_anchors, num_classes):
   darknet = tf.keras.models.Model(inputs, darknet_body(inputs))
   x, y1 = make_last_layers(darknet.output, 512, num_anchors*(num_classes+5))
   x = compose(
         DarknetConv2D_BN_Leaky(256, (1,1)),
         tf.keras.layers.UpSampling2D(2))(x)
   x = tf.keras.layers.Concatenate()([x,darknet.layers[152].output])
   x, y2 = make_last_layers(x, 256, num_anchors*(num_classes+5))
   x = compose(
         DarknetConv2D_BN_Leaky(128, (1,1)),
         tf.keras.layers.UpSampling2D(2))(x)
   x = tf.keras.layers.Concatenate()([x,darknet.layers[92].output])
   x, y3 = make_last_layers(x, 128, num_anchors*(num_classes+5))
   return tf.keras.models.Model(inputs, [y1,y2,y3])
def get_classes(classes_path):
   with open(classes_path) as f:
      class_names = f.readlines()
   class_names = [c.strip() for c in class_names]
   return class_names
def get_anchors(anchors_path):
   with open(anchors_path) as f:
      anchors = f.readline()
   anchors = [float(x) for x in anchors.split(',')]
   return np.array(anchors).reshape(-1, 2)
class Decode(object):
   def __init__(self, obj_threshold, nms_threshold, input_shape, _yolo, all_classes):
      self._t1 = obj_threshold
      self._t2 = nms_threshold
      self.input_shape = input_shape
      self.all_classes = all_classes
      self.num_classes = len(self.all_classes)
      self._yolo = _yolo
   def detect_image(self, image):
      pimage = self.process_image(np.copy(image))
      boxes, scores, classes = self.predict(pimage, image.shape)
      return boxes, scores, classes
   def multi_thread_post(self, batch_img, outs, i, draw_image, result_image, result_boxes, result_scores, result_classes):
      a1 = np.reshape(outs[0][i], (1, self.input_shape[0] // 32, self.input_shape[1] // 32, 3, 5 + self.num_classes))
      a2 = np.reshape(outs[1][i], (1, self.input_shape[0] // 16, self.input_shape[1] // 16, 3, 5 + self.num_classes))
      a3 = np.reshape(outs[2][i], (1, self.input_shape[0] // 8, self.input_shape[1] // 8, 3, 5 + self.num_classes))
      boxes, scores, classes = self._yolo_out([a1, a2, a3], batch_img[i].shape)
      if boxes is not None and draw_image:
         self.draw(batch_img[i], boxes, scores, classes)
      result_image[i] = batch_img[i]
      result_boxes[i] = boxes
      result_scores[i] = scores
      result_classes[i] = classes
   def detect_batch(self, batch_img, draw_image):
      batch_size = len(batch_img)
      result_image, result_boxes, result_scores, result_classes = [None] * batch_size, [None] * batch_size, [None] * batch_size, [None] * batch_size
      batch = []
      for image in batch_img:
         pimage = self.process_image(np.copy(image))
         batch.append(pimage)
      batch = np.concatenate(batch, axis=0)
      outs = self._yolo.predict(batch)
      threads = []
      for i in range(batch_size):
         t = threading.Thread(target=self.multi_thread_post, args=(
            batch_img, outs, i, draw_image, result_image, result_boxes, result_scores, result_classes))
         threads.append(t)
         t.start()
      for t in threads:
         t.join()
      return result_image, result_boxes, result_scores, result_classes
   def draw(self, image, boxes, scores, classes):
      image_h, image_w, _ = image.shape
      hsv_tuples = [(1.0 * x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
      colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
      colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
      random.seed(0)
      random.shuffle(colors)
      random.seed(None)
      for box, score, cl in zip(boxes, scores, classes):
         x0, y0, x1, y1 = box
         left = max(0, np.floor(x0 + 0.5).astype(int))
         top = max(0, np.floor(y0 + 0.5).astype(int))
         right = min(image.shape[1], np.floor(x1 + 0.5).astype(int))
         bottom = min(image.shape[0], np.floor(y1 + 0.5).astype(int))
         bbox_color = colors[cl]
         bbox_thick = 1
         cv.rectangle(image, (left, top), (right, bottom), bbox_color, bbox_thick)
         bbox_mess = '%s: %.2f' % (self.all_classes[cl], score)
         t_size = cv.getTextSize(bbox_mess, 0, 0.5, thickness=1)[0]
         cv.rectangle(image, (left, top), (left + t_size[0], top - t_size[1] - 3), bbox_color, -1)
         cv.putText(image, bbox_mess, (left, top - 2), cv.FONT_HERSHEY_SIMPLEX,
                     0.5, (0, 0, 0), 1, lineType=cv.LINE_AA)
   def process_image(self, img):
      img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
      h, w = img.shape[:2]
      scale_x = float(self.input_shape[1]) / w
      scale_y = float(self.input_shape[0]) / h
      img = cv.resize(img, None, None, fx=scale_x, fy=scale_y, interpolation=cv.INTER_CUBIC)
      pimage = img.astype(np.float32) / 255.
      pimage = np.expand_dims(pimage, axis=0)
      return pimage
   def predict(self, image, shape):
      outs = self._yolo.predict(image)
      a1 = np.reshape(outs[0], (1, self.input_shape[0]//32, self.input_shape[1]//32, 3, 5+self.num_classes))
      a2 = np.reshape(outs[1], (1, self.input_shape[0]//16, self.input_shape[1]//16, 3, 5+self.num_classes))
      a3 = np.reshape(outs[2], (1, self.input_shape[0]//8, self.input_shape[1]//8, 3, 5+self.num_classes))
      boxes, scores, classes = self._yolo_out([a1, a2, a3], shape)
      return boxes, scores, classes
   def _sigmoid(self, x):
      return 1 / (1 + np.exp(-x))
   def _process_feats(self, out, anchors, mask):
      grid_h, grid_w, num_boxes = map(int, out.shape[1: 4])
      anchors = [anchors[i] for i in mask]
      anchors_tensor = np.array(anchors).reshape(1, 1, len(anchors), 2)
      out = out[0]
      box_xy = self._sigmoid(out[..., :2])
      box_wh = np.exp(out[..., 2:4])
      box_wh = box_wh * anchors_tensor
      box_confidence = self._sigmoid(out[..., 4])
      box_confidence = np.expand_dims(box_confidence, axis=-1)
      box_class_probs = self._sigmoid(out[..., 5:])
      col = np.tile(np.arange(0, grid_h), grid_w).reshape(-1, grid_w)
      row = np.tile(np.arange(0, grid_w).reshape(-1, 1), grid_h)
      col = col.reshape(grid_h, grid_w, 1, 1).repeat(3, axis=-2)
      row = row.reshape(grid_h, grid_w, 1, 1).repeat(3, axis=-2)
      grid = np.concatenate((col, row), axis=-1)
      box_xy += grid
      box_xy /= (grid_w, grid_h)
      box_wh /= self.input_shape
      box_xy -= (box_wh / 2.)
      boxes = np.concatenate((box_xy, box_wh), axis=-1)
      return boxes, box_confidence, box_class_probs
   def _filter_boxes(self, boxes, box_confidences, box_class_probs):
      box_scores = box_confidences * box_class_probs
      box_classes = np.argmax(box_scores, axis=-1)
      box_class_scores = np.max(box_scores, axis=-1)
      pos = np.where(box_class_scores >= self._t1)
      boxes = boxes[pos]
      classes = box_classes[pos]
      scores = box_class_scores[pos]
      return boxes, classes, scores
   def _nms_boxes(self, boxes, scores):
      x = boxes[:, 0]
      y = boxes[:, 1]
      w = boxes[:, 2]
      h = boxes[:, 3]
      areas = w * h
      order = scores.argsort()[::-1]
      keep = []
      while order.size > 0:
         i = order[0]
         keep.append(i)
         xx1 = np.maximum(x[i], x[order[1:]])
         yy1 = np.maximum(y[i], y[order[1:]])
         xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
         yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])
         w1 = np.maximum(0.0, xx2 - xx1 + 1)
         h1 = np.maximum(0.0, yy2 - yy1 + 1)
         inter = w1 * h1
         ovr = inter / (areas[i] + areas[order[1:]] - inter)
         inds = np.where(ovr <= self._t2)[0]
         order = order[inds + 1]
      keep = np.array(keep)
      return keep
   def _yolo_out(self, outs, shape):
      masks = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
      anchors = [[12, 16], [19, 36], [40, 28], [36, 75], [76, 55],
                [72, 146], [142, 110], [192, 243], [459, 401]]
      boxes, classes, scores = [], [], []
      for out, mask in zip(outs, masks):
         b, c, s = self._process_feats(out, anchors, mask)
         b, c, s = self._filter_boxes(b, c, s)
         boxes.append(b)
         classes.append(c)
         scores.append(s)
      boxes = np.concatenate(boxes)
      classes = np.concatenate(classes)
      scores = np.concatenate(scores)
      w, h = shape[1], shape[0]
      image_dims = [w, h, w, h]
      boxes = boxes * image_dims
      nboxes, nclasses, nscores = [], [], []
      for c in set(classes):
         inds = np.where(classes == c)
         b = boxes[inds]
         c = classes[inds]
         s = scores[inds]
         keep = self._nms_boxes(b, s)
         nboxes.append(b[keep])
         nclasses.append(c[keep])
         nscores.append(s[keep])
      if not nclasses and not nscores:
         return None, None, None
      boxes = np.concatenate(nboxes)
      classes = np.concatenate(nclasses)
      scores = np.concatenate(nscores)
      boxes[:, [2, 3]] = boxes[:, [0, 1]] + boxes[:, [2, 3]]
      return boxes, scores, classes
class Costmap:
   def __init__(self, image):
      color = rospy.get_param("map_maker/color")
      thresh = rospy.get_param("map_maker/color_thresh")
      self.image_map = self.get_image_map(image, color, thresh)
   def get_image_map(self, image, color, thresh):
      canvas_1 = np.zeros(image.shape[:], dtype="uint8")
      canvas_2 = np.zeros(image.shape[:], dtype="uint8")
      ret, canvas_1[:,:,0] = cv.threshold(image[:,:,0], color[0]-thresh, 255, cv.THRESH_BINARY)
      ret, canvas_2[:,:,0] = cv.threshold(image[:,:,0], color[0]+thresh, 255, cv.THRESH_BINARY_INV)
      ret, canvas_1[:,:,1] = cv.threshold(image[:,:,1], color[1]-thresh, 255, cv.THRESH_BINARY)
      ret, canvas_2[:,:,1] = cv.threshold(image[:,:,1], color[1]+thresh, 255, cv.THRESH_BINARY_INV)
      ret, canvas_1[:,:,2] = cv.threshold(image[:,:,2], color[2]-thresh, 255, cv.THRESH_BINARY)
      ret, canvas_2[:,:,2] = cv.threshold(image[:,:,2], color[2]+thresh, 255, cv.THRESH_BINARY_INV)
      image_map = cv.bitwise_and(cv.bitwise_and(canvas_1[:,:,0],canvas_2[:,:,0]),cv.bitwise_and(canvas_1[:,:,1],canvas_2[:,:,1]),cv.bitwise_and(canvas_1[:,:,2],canvas_2[:,:,2]))
      image_map = cv.bitwise_not(image_map)
      return image_map
   def func(self, x):
      return -x
   def get_vector_map(self, point):
      image_map = self.image_map
      weight = {0:0, 255:1000}
      visited = np.full(image_map.shape, 0, dtype='bool')
      score_map = np.full(image_map.shape, inf, dtype='float')
      col, row = point
      score_map[row, col] = 0
      queue = []
      queue.append([score_map[row, col], row, col])
      while len(queue)!=0:  
         node_points, node_row, node_col = min(queue)
         queue.remove(min(queue))
         if visited[node_row, node_col]==1:
            continue
         visited[node_row, node_col] = 1
         directions = [[0,1,1],[-1,0,1],[0,-1,1],[1,0,1],[1,1,2**0.5],[1,-1,2**0.5],[-1,1,2**0.5],[-1,-1,2**0.5]]
         #directions = [[0,1,1],[-1,0,1],[0,-1,1],[1,0,1]]
         for dir in directions:
            newnode_row, newnode_col = node_row+dir[0], node_col+dir[1]
            if newnode_row in range(image_map.shape[0]) and newnode_col in range(image_map.shape[1]):
               if score_map[newnode_row, newnode_col] > score_map[node_row, node_col] + weight[image_map[newnode_row, newnode_col]] + dir[-1]:
                     score_map[newnode_row, newnode_col] = score_map[node_row, node_col] + weight[image_map[newnode_row, newnode_col]] + dir[-1]
                     queue.append([score_map[newnode_row, newnode_col], newnode_row, newnode_col])
      dir_map = self.func(score_map)
      kernel_x = np.array([[-2**-0.5, 0, 2**-0.5],
                          [-1, 0, 1],
                          [-2**-0.5, 0, 2**-0.5]])
      kernel_y = np.array([[2**-0.5, 1, 2**-0.5],
                          [0, 0, 0],
                          [-2**-0.5, -1, -2**-0.5]])
      vector_map_x = cv.filter2D(dir_map, -1, kernel_x)
      vector_map_y = cv.filter2D(dir_map, -1, kernel_y)
      vector_map = np.arctan2(vector_map_y, vector_map_x)
      return vector_map
   def draw(self, obj, vector_map):
      x, y = locations[obj]
      vector_space = cv.cvtColor(self.image_map, cv.COLOR_GRAY2BGR)
      vector_space = cv.bitwise_not(vector_space)
      vector_space = cv.circle(vector_space, (x,y), min_dist, (0,255,0), 2)
      for row in range(0,vector_space.shape[0],6):
         for col in range(0,vector_space.shape[1],6):
            theta = vector_map[row,col]
            dis = 3
            n_row = int(row-dis*m.sin(theta))
            n_col = int(col+dis*m.cos(theta))
            vector_space = cv.arrowedLine(vector_space, (col,row), (n_col,n_row), (0,150,150), 1, tipLength = 0.3)
      return vector_space
class ROS_NODE:
   def __init__(self):
      rospy.init_node('map_maker', anonymous=True)
      self.bridge = CvBridge()
      self.pub1 = rospy.Publisher('vector_map', Image, queue_size=1)
      self.pub2 = rospy.Publisher('vector_space', Image, queue_size=1)
      self.pub3 = rospy.Publisher('condition', Bool, queue_size=1)
      rospy.Subscriber('state', state, self.state_related_cb)
      rospy.Subscriber('image', Image, self.img_related_cb)
      self.cond = Bool()
      self.cond.data = False
   def state_related_cb(self, data):
      try:
         x, y = locations[obj]
         dist = ((x-data.x)**2+(y-data.y)**2)**0.5
         if dist<=min_dist:
            self.cond = False
      except:
         pass
   def img_related_cb(self, data):
      self.img = self.bridge.imgmsg_to_cv2(data, desired_encoding="passthrough")

node = ROS_NODE()
priority_order = rospy.get_param("map_maker/priority_order")
default_locations = rospy.get_param("map_maker/default_locations")
locations = {key:data for key, data in zip(priority_order, default_locations)}
min_dist = rospy.get_param("map_maker/min_dist")
class_names = get_classes(r"/home/divyam/Desktop/ROS/ROS_WS/src/hardwired/ml_model data/hardwired_objects_classes.txt")
num_classes = len(class_names)     
anchors = get_anchors(r"/home/divyam/Desktop/ROS/ROS_WS/src/hardwired/ml_model data/yolo_anchors.txt")
num_anchors = len(anchors)
conf_thresh = rospy.get_param("map_maker/conf_thresh")
nms_thresh = rospy.get_param("map_maker/nms_thresh")
model = yolo_body(tf.keras.layers.Input(shape=INPUT_SHAPE+(3,)), num_anchors//3, num_classes)
model.load_weights(r"/home/divyam/Desktop/ROS/ROS_WS/src/hardwired/ml_model data/hardwired_objects_yolov3.h5")
decode = Decode(conf_thresh, nms_thresh, INPUT_SHAPE, model, class_names)
while not rospy.is_shutdown():
   try:
      boxes, scores, classes = decode.detect_image(node.img)
      for box, cl in zip(boxes, classes):
         x1, y1, x2, y2 = box
         x1, y1, x2, y2 = max(0,np.floor(x1+0.5).astype(int)), max(0,np.floor(y1+0.5).astype(int)), min(node.img.shape[1],np.floor(x2+0.5).astype(int)), min(node.img.shape[0],np.floor(y2+0.5).astype(int))
         location = [int(x1/2+x2/2),int(y1/2+y2/2)]
         locations[class_names[cl]] = location
      print("Objects detected")

   except:
      print("Objects not detected")
      #t.sleep(1)

costmap = Costmap(node.img)
for obj in priority_order:
   vector_map = costmap.get_vector_map(locations[obj])
   map_with_dir = costmap.draw(obj, vector_map)
   map_with_dir_msg = node.bridge.cv2_to_imgmsg(map_with_dir, encoding="passthrough")
   vector_map_msg = node.bridge.cv2_to_imgmsg(vector_map, encoding="passthrough")
   node.cond = True
   while not rospy.is_shutdown() and node.cond==True:
      node.pub1.publish(vector_map_msg)
      node.pub2.publish(map_with_dir_msg)
      node.pub3.publish(node.cond)