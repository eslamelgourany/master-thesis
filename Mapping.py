#!/usr/bin/python

#This script is particle filter localization.
#Input: ROS bag which should have VYCON NODE, CAMERA NODE, ROBOT MOVEMENT
#Output: TXT file which will have [X Y Z DESCRIPTOR]
# Author : Eslam Elgourany
# Contact : eslamelgourany@hotmail.com
# Thesis source code, CVUT, Prague, Czech Republic


#Import Libraries
#==================
from __future__ import print_function
import sys
import numpy as np
import time
import math
import cv2
import rospy
from pyquaternion import Quaternion
import roslib
import struct
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import String, Header
from sensor_msgs.msg import Image
import matplotlib.pyplot as plt
from geometry_msgs.msg import PointStamped, PoseStamped, Point, Pose, Quaternion, Transform
import tf.transformations as tr
from tf.transformations import euler_from_quaternion, quaternion_from_euler, quaternion_multiply,  quaternion_matrix
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.pyplot import plot, draw, show, ion



#LEGEND
#===========

# P = K R T
# K = camera_matrix
# R = Rotation Matrix
# T = Translation vector
# P = output_projection matrix


# Triangulate (Projection points, Projection Matricies)


class Position:

    def __init__(self):

        # Initialization of Variables
        #=============================

        self.translion_sub = rospy.Subscriber("/vrpn_client_node/turtle9/pose", PoseStamped, self.callback_position)
        self.image_sub = rospy.Subscriber("pylon_camera_node/image_raw", Image, self.callback_image)
        self.image = None
        self.bridge = CvBridge()
        self.detector = cv2.AKAZE_create()
        self.loop_rate = rospy.Rate(1)
        self.pub = rospy.Publisher("pylon_camera_node/image_akaze", Image, queue_size=10)
        self.previous_kp = None
        self.previous_des = None
        self.triangulate = None
        self.i = 0
        self.camera_matrix =  np.array ([[1087.5798, 0, 652.0836], [0, 1089.213, 484.02 ],[0, 0, 1]])
        self.distortion_vector = np.array([ -0.205646 , 0.039258 , -0.003634 , 0.001089 , 0.000000])

        # intiating previous x and y for translation calculations
        self.previous_x_position = float(0)
        self.previous_y_position = float(0)
        self.previous_z_position = float(0)
        # intiating previous x and y for Rotation calculations
        self.current_x_orientation = float(0)
        self.current_y_orientation = float(0)
        self.current_z_orientation = float(0)
        self.current_w_orientation = float(0)
        self.previous_x_orientation = float(0)
        self.previous_y_orientation = float(0)
        self.previous_z_orientation = float(0)
        self.previous_w_orientation = float(0)


        # FOR SAVING The map and descriptors, Uncomment below!
        # =====================================================

        # f = open("Triangulate_and_desc_Trajectory.txt", "w")
        # f.close()

    def callback_position(self,data):

        # -- Rotation_Matrix [Quaternion]
        #===============================================

        self.current_x_orientation = data.pose.orientation.x
        self.current_y_orientation = data.pose.orientation.y
        self.current_z_orientation = data.pose.orientation.z
        self.current_w_orientation = data.pose.orientation.w

        # -- Translation [VECTOR]
        #==========================================

        self.current_x_position = data.pose.position.x
        self.current_y_position = data.pose.position.y
        self.current_z_position = data.pose.position.z


    def callback_image(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        self.image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        (kps,descs) = self.detector.detectAndCompute(cv_image, 0)
        cv2.drawKeypoints(self.image, kps, self.image,(0,1000,0))
        rospy.loginfo("Timing images")
        if self.image is not None:
            self.pub.publish(self.bridge.cv2_to_imgmsg(self.image, "mono8"))


            # DISTANCE BETWEEN TWO FRAMES
            #=============================

            def compute_dist(x1, y1, x2, y2):
                dist = math.sqrt(((x2 - x1)**2) + ((y2 - y1)**2))
                return dist

            distance = compute_dist(self.previous_x_position, self.previous_y_position, self.current_x_position, self.current_y_position)

            if distance > 0.3:
                corrected_image = cv2.undistort(self.image, self.camera_matrix, self.distortion_vector)

                current_distance_kp, current_distance_des = self.detector.detectAndCompute(corrected_image, None)
                current_distance_array = [p.pt for p in current_distance_kp]
                current_distance_np_array = np.array(current_distance_array)
                current_distance_des_list_for_appending = list()

                if  self.previous_kp == None and self.previous_des == None:
                    self.previous_kp = current_distance_kp
                    self.previous_des = current_distance_des
                    self.previous_z_position = self.current_z_position
                    self.previous_x_position = self.current_x_position
                    self.previous_y_position = self.current_y_position
                    self.previous_x_orientation = self.current_x_orientation
                    self.previous_y_orientation =  self.current_y_orientation
                    self.previous_z_orientation = self.current_z_orientation
                    self.previous_w_orientation = self.current_w_orientation
                    return
                else:
                    print("Previous is full")


                # ====================== CALCULATIONS FOR PROJECTION MATRICIES ===================#
                orientation_list = [self.current_x_orientation, self.current_y_orientation, self.current_z_orientation, self.current_w_orientation]
                (roll, pitch, yaw) = euler_from_quaternion(orientation_list)
                fi = 0*yaw #0

                #print("FI current ", fi)        #DEBUG:
                x = 0*self.current_x_position
                y = 0*self.current_y_position
                c = np.cos(fi)
                s = np.sin(fi)
                d = 0*np.sqrt(x*x + y*y)
                alpha = np.arctan2(y,x)
                tvec = np.array([d * np.cos(alpha-fi), d*np.sin(alpha-fi), 0], dtype = float).reshape(3,1)
                #print("tvec is", tvec)          # DEBUG:


                R_matrix = np.array([
                [ c, -s, 0],
                [ s, c, 0],
                [ 0, 0, 1]
                ])

                R_t_matrix = np.concatenate((np.transpose(R_matrix), -tvec), axis =1)

                previous_orientation_list = [self.previous_x_orientation, self.previous_y_orientation, self.previous_z_orientation, self.previous_w_orientation]
                (previous_roll, previous_pitch, previous_yaw) = euler_from_quaternion(previous_orientation_list)
                fi_previous = previous_yaw - yaw
                #print("FI PREVIOUS ", fi_previous)    # DEBUG:
                x_previous = self.previous_x_position - self.current_x_position
                y_previous = self.previous_y_position - self.current_y_position

                #print("current fi ", fi)            # DEBUG:
                c_previous = np.cos(fi_previous)
                s_previous = np.sin(fi_previous)
                d_previous = np.sqrt(((x_previous)*(x_previous)) + ((y_previous)*(y_previous)))
                alpha_previous = np.arctan2(y_previous, x_previous)
                tvec_previous = np.array([d_previous*np.cos(alpha_previous-fi_previous), d_previous*np.sin(alpha_previous-fi_previous), 0], dtype = float).reshape(3,1)
                #print ("previous tvec ", tvec_previous)    # DEBUG:

                R_matrix_previous = np.array([
                [ c_previous, -s_previous, 0],
                [ s_previous, c_previous, 0],
                [ 0, 0, 1]
                ])

                R_t_matrix_previous = np.concatenate(((R_matrix_previous), tvec_previous), axis =1)
                projection_1_current =  np.dot(self.camera_matrix,R_t_matrix)
                projection_2_previous = np.dot(self.camera_matrix, R_t_matrix_previous)


                # #============== BRUTE FORCE MATCHING ===============#
                bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck= True)
                matches = bf.match(current_distance_des, self.previous_des)
                good = list()
                for m in matches :
                    #print("match ",m.queryIdx)                                   # DEBUG:
                    if m.distance < 30:                          # Distance between matches
                        good.append(m)
                        #print("good list is ", good)                               # DEBUG:
                min_match_count = 2
                matched_descriptors = list()
                if len(good) > min_match_count:
                    for goodm in good:

                        matched_descriptors.append(current_distance_des[goodm.queryIdx])        #nmatched_descriptors is Used for saving good desc
                    nmatched_descriptors = np.array(matched_descriptors)
                    #print("matched_descriptors", nmatched_descriptors.ndim)    # DEBUG:
                    projection_points_1 = np.float32([current_distance_kp[m.queryIdx].pt for m in good ]).T
                    projection_points_2 = np.float32([self.previous_kp[m.trainIdx].pt for m in good ]).T

                    previous_key_pts_array = [p.pt for p in self.previous_kp]
                    current_key_pts_array = [p.pt for p in current_distance_kp]

                    #print("PRJ1", projection_points_1)                         ##DEBUG:
                    #print("PRJ2", projection_points_2)                         # DEBUG:


                    #print("Projection matrix current", projection_1_current)      # DEBUG:
                    #print("Projection matrix previous", projection_2_previous)     # DEBUG:


                    #TRIANGULATE FUNCTION
                    #====================
                    self.triangulate = cv2.triangulatePoints(projection_1_current, projection_2_previous, projection_points_1, projection_points_2)


                    #Rotation and translation for triangulation points
                    #===================================================
                    c = np.swapaxes(self.triangulate, 0, 1)

                    for i in range(len(c)):
                        Xc = c[:,0]/c[:,3]
                        Yc = c[:,1]/c[:,3]
                        Zc = c[:,2]/c[:,3]
                        Wc = c[:,3]

                    #print("Xc is ", Xc)        # DEBUG:
                    #print("Yc is ", Yc)        # DEBUG:
                    #print("Zc is ", Zc)        # DEBUG:
                    #print("fi is ", fi)        # DEBUG:
                    #print ("x{} and  y{} of robot" .format(self.current_x_position, self.current_y_position))  # DEBUG:
                    x_features = ((Xc * np.cos(fi)) - (Yc * np.sin(fi))) + (self.current_x_position)
                    y_features = ((Yc * np.sin(fi)) + (Yc * np.cos(fi))) + (self.current_y_position)
                    z_features = Zc + self.current_z_position
                    w_features = Wc
                    print("x_features ", x_features)
                    print("y_features ", y_features)
                    print("z_features ", z_features)


                    final_features_coordinates = np.array([x_features, y_features, z_features, w_features]).T
                    #print("final_features_coordinates", final_features_coordinates)#/final_features_coordinates[3])    # DEBUG:
                    #print("final_features_coordinates", final_features_coordinates)                                    # DEBUG:
                    d_new = np.concatenate((final_features_coordinates, nmatched_descriptors), axis = 1)



                    # UNCOMMENT BELOW TO SAVE TRIANGULTE AND DESC IN A TXT FILE
                    #==============================================================
                    
                    # f = open("Triangulate_and_desc_Trajectory.txt", "a")
                    # for row in range(d_new.shape[0]):
                    #     line = (','.join(str(n) for n in d_new[row]) + '\n')
                    #     f.write(line)
                    # f.close()

                # SWAPPING OF VARIABLES
                #========================

                self.previous_kp = current_distance_kp
                self.previous_des = current_distance_des
                self.previous_z_position = self.current_z_position
                self.previous_x_position = self.current_x_position
                self.previous_y_position = self.current_y_position
                self.previous_x_orientation = self.current_x_orientation
                self.previous_y_orientation =  self.current_y_orientation
                self.previous_z_orientation = self.current_z_orientation
                self.previous_w_orientation = self.current_w_orientation

            else:
                pass


def main(args):
    ic = Position()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")





if __name__ == '__main__':
    rospy.init_node('mapping_script', anonymous=True)
    main(sys)
