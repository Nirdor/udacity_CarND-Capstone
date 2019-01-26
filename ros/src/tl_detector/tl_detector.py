#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier
from scipy.spatial import KDTree
import tf
import cv2
import yaml
import math
import numpy as np

STATE_COUNT_THRESHOLD = 2
train = False
import time

class TLDetector(object):
    def __init__(self):
        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''
        rospy.init_node('tl_detector')

        self.loaded = False
        self.pose = None
        self.waypoints = None
        self.camera_image = None
        self.tree = None
        self.lights = []
        self.img = None
        self.has_image = False
        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_time = 0

        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        
        sub3 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)
        sub6 = rospy.Subscriber('/image_color', Image, self.image_cb)

        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)

        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)

        #self.bridge = CvBridge()
        self.light_classifier = TLClassifier(self.config['is_site'])
        #self.listener = tf.TransformListener()
        
        self.last_wp = -1
        self.state_count = 0
        self.loaded = True
        
        rospy.spin()

    def pose_cb(self, msg):
        self.pose = msg

    def waypoints_cb(self, waypoints):
        self.waypoints = waypoints.waypoints
        self.tree = KDTree([[p.pose.pose.position.x, p.pose.pose.position.y] for p in waypoints.waypoints])

    def traffic_cb(self, msg):
        self.lights = msg.lights

    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """
        self.has_image = True
        self.camera_image = msg
        if self.loaded and time.time() - self.last_time > 0.2:
            light_wp, state = self.process_traffic_lights()
        else:
            return
        self.last_time = time.time()
        if light_wp == None:
            return

        '''
        Publish upcoming red lights at camera frequency.
        Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
        of times till we start using it. Otherwise the previous stable state is
        used.
        '''
        if self.state != state:
            self.state_count = 0
            self.state = state
        elif self.state_count >= STATE_COUNT_THRESHOLD:
            self.last_state = self.state
            light_wp = light_wp if state == TrafficLight.RED or state == TrafficLight.YELLOW else -1
            self.last_wp = light_wp
            self.upcoming_red_light_pub.publish(Int32(light_wp))
        else:
            self.upcoming_red_light_pub.publish(Int32(self.last_wp))
        self.state_count += 1

    def get_closest_waypoint(self, pose):
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            pose (Pose): position to match a waypoint to

        Returns:
            int: index of the closest waypoint in self.waypoints
        """
        #TODO implement
        if self.tree is not None:
            x, y = pose.position.x, pose.position.y
            return self.tree.query([x, y])[1]
        return 0

    def get_light_state(self, light):
        """Determines the current color of the traffic light

        Args:
            light (TrafficLight): light to classify

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        if(not self.has_image or self.light_classifier is None):
            self.prev_light_loc = None
            return False
        
        #Get classification
        img = np.frombuffer(self.camera_image.data, np.uint8).reshape((self.camera_image.height, self.camera_image.width, 3))
        img = cv2.resize(img, None, fx = 0.5, fy = 0.5, interpolation = cv2.INTER_AREA)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        self.img = img.reshape((1, self.camera_image.height // 2, self.camera_image.width // 2, 3))
        return self.light_classifier.get_classification(self.img)

    def distance(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(wp1 + 1, wp2 + 1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist
    
    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        light = None

        
        if(self.pose and self.config):
            # List of positions that correspond to the line to stop in front of for a given intersection
            stop_line_positions = self.config['stop_line_positions']
                        
            car_position = self.get_closest_waypoint(self.pose.pose)
            
            #TODO find the closest visible traffic light (if one exists)
            if not train:
                for stop in stop_line_positions:
                    stop_pose = Pose()
                    stop_pose.position.x, stop_pose.position.y = stop
                    light_wp = self.get_closest_waypoint(stop_pose)
                    if light_wp > car_position and self.distance(self.waypoints, car_position, light_wp) < 70:
                        light = light_wp
                        break
            else:
                for l in self.lights:
                    light_wp = self.get_closest_waypoint(l.pose.pose)
                    #rospy.loginfo('Light:{}, Car:{}, max:{}'.format(light_wp, car_position, self.waypoints))
                    if light_wp > car_position and self.distance(self.waypoints, car_position, light_wp) < 100:
                        light = light_wp
                        break
                    
                
        if light:
            if not train:
                state = self.get_light_state(light)
            else:
                state = l.state
            return light, state
        return -1, TrafficLight.UNKNOWN

if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
