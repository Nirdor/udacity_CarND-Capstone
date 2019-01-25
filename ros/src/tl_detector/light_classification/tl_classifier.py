from styx_msgs.msg import TrafficLight

import numpy as np
import tensorflow as tf

class TLClassifier(object):
  def __init__(self, site = False):
    #TODO load classifier
    self.loaded = False
    if site:
      pass
    else:
      self.detection_graph = self.load_graph('frozen_inference_graph.pb')
    
    self.input_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
    self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
    self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
    self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
    self.sess = tf.Session(graph=self.detection_graph)
    self.sess.run([self.detection_boxes, self.detection_scores, self.detection_classes], feed_dict={self.input_tensor: np.zeros((1, 300, 400, 3), np.uint8)})
    self.loaded = True
  
  def load_graph(self, graph_file):
    """Loads a frozen inference graph"""
    graph = tf.Graph()
    with graph.as_default():
      od_graph_def = tf.GraphDef()
      with tf.gfile.GFile(graph_file, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')
    return graph

  def get_classification(self, image):
    """Determines the color of the traffic light in the image

    Args:
        image (cv::Mat): image containing the traffic light

    Returns:
        int: ID of traffic light color (specified in styx_msgs/TrafficLight)

    """
    #TODO implement light color prediction
    
    if not self.loaded:
      return TrafficLight.UNKNOWN
    ret = TrafficLight.UNKNOWN
    
    #with tf.Session(graph=self.detection_graph) as sess:
    (boxes, scores, classes) = self.sess.run([self.detection_boxes, self.detection_scores, self.detection_classes], feed_dict={self.input_tensor: image})
    
    boxes = np.squeeze(boxes)
    scores = np.squeeze(scores)
    classes = np.squeeze(classes)
    
    for i in range(len(classes)):
      if scores[i] > 0.5:
        if ret != TrafficLight.UNKNOWN and ret != classes[i] - 1:
          return TrafficLight.UNKNOWN
        ret = int(classes[i] - 1)
    
      return ret
