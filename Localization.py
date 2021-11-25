#!/usr/bin/python

#This script is particle filter localization.
#Input: TXT file of map which have [X, Y ,Z, DESCRIPTOR]
#Output: Graphs of visualization for the robot position
# Author : Eslam Elgourany
# Contact : eslamelgourany@hotmail.com
# Thesis source code, CVUT, Prague, Czech Republic


#Import Libraries
#===================
import numpy as np
import rospy
import roslib; roslib.load_manifest('visualization_marker_tutorials')
import math
import itertools
from fractions import Fraction as frac
import cv2
from cv_bridge import CvBridge, CvBridgeError

# Third party libraries
#========================
from numpy.random import random
from std_msgs.msg import String, Header
from geometry_msgs.msg import Pose, PoseStamped, Quaternion, Point, Vector3
from filterpy.monte_carlo import systematic_resample
import matplotlib.pyplot as plt
from numpy.random import seed
from nav_msgs.msg import Odometry
import tf.transformations as tf
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from sensor_msgs.msg import Image
from geometry_msgs.msg import
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
from geometry_msgs.msg import
from std_msgs.msg import Header, ColorRGBA
from scipy.cluster.hierarchy import fcluster
from scipy.cluster.hierarchy import cophenet
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist
import sklearn.metrics as sm
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import matplotlib.colors as colors
import matplotlib.cm as cmx

#Initialize some Values
#========================

previous_orientation_x = 0
previous_orientation_y = 0
previous_orientation_z = 0
previous_orientation_w = 0
previous_orientation_z = float(0)
current_orientation_z = float(0)
previous_position_y = float(0)
previous_position_x = float(0)
gamma = float(0)
desc = None
image_width = float(1280)       # CAMERA IS 1.2 MP ... 1280 x 960
image_height = float(960)
camera_matrix =  np.array ([[1087.5798, 0, 652.0836], [0, 1089.213, 484.02 ],[0, 0, 1]])
num_of_particles = 2500
z_position_for_particle = 0.3
number_of_matches = 0

# LIST OF VARIABLES
#=====================
Xsnew = list()
Ysnew = list()
Zsnew = list()
Wsnew = list()
desc_features = list()
distances_of_matches = list()
features_in_range_particles = list()
all_particles_positions = list()
matched1= list()
particle_heading_list = list()


global weights
weights = np.ones(num_of_particles)



# OPEN THE MAP FILE
#======================

f = open('Triangulate_and_desc_Trajectory.txt', 'r').readlines()


# ORGANISE THE DATA IN THE TXT FILE !
#=====================================

for line in f:
    tmp = line.strip().split(",")
    values = [float(v) for v in tmp]
    #print("values", values)
    points4d = np.array(values).reshape((-1, 65))
    points3d = points4d[:, :65]         # 65 is number of elements per row
    #Normalization
    Xs = points3d[:, 0]/points4d[:,3]
    Ys = points3d[:, 1]/points4d[:,3]
    Zs = points3d[:, 2]/points4d[:,3]
    Ws = points3d[:, 3]/points4d[:,3]
    desc_features.append(values[4:])


    # FILTERATION OF THE FEATURES !!!
    #=================================
    output_list = []
    for i in range(len(Xs)):
        if 2.8 < Zs[i] < 3.6:
            output_list.append([Xs[i], Ys[i], Zs[i], Ws[i] , desc])

    for values in output_list:
        Xsnew.append(values[0])
        Ysnew.append(values[1])
        Zsnew.append(values[2])
        Wsnew.append(values[3])

# VICON CALLBACK
#===============

def callback_position_vicon(data):
    global current_x_position_real
    global current_y_position_real
    global current_z_position_real
    global current_x_orientation_real
    global current_y_orientation_real
    global current_z_orientation_real
    global current_w_orientation_real
    global real_position_of_the_robot
    global fi_robot
    current_x_position_real = data.pose.position.x
    current_y_position_real = data.pose.position.y
    current_z_position_real = data.pose.position.z
    current_x_orientation_real = data.pose.orientation.x
    current_y_orientation_real = data.pose.orientation.y
    current_z_orientation_real = data.pose.orientation.z
    current_w_orientation_real = data.pose.orientation.w
    current_orientation_list_odom = [current_x_orientation_real, current_y_orientation_real, current_z_orientation_real, current_w_orientation_real]
    (roll_robot, pitch_robot, yaw_robot) = euler_from_quaternion(current_orientation_list_odom)
    fi_robot = yaw_robot
    real_position_of_the_robot = [current_x_position_real, current_y_position_real]

#ODOM CALLBACK
#==============

def callback_position_odom(msg):
    global gamma
    global previous_position_y
    global previous_position_x
    global x_array_diff
    global y_array_diff
    global current_orientation_z
    global previous_orientation_x
    global previous_orientation_y
    global previous_orientation_z
    global previous_orientation_w
    global current_position_x
    global current_position_y
    global fi_previous
    global fi_current

    current_position_x = msg.pose.pose.position.x
    current_position_y = msg.pose.pose.position.y
    current_orientation_x = msg.pose.pose.orientation.x
    current_orientation_y = msg.pose.pose.orientation.y
    current_orientation_z = msg.pose.pose.orientation.z
    current_orientation_w = msg.pose.pose.orientation.w



    current_orientation_list_odom = [current_orientation_x, current_orientation_y, current_orientation_z, current_orientation_w]
    (roll_odom_current, pitch_odom_current, yaw_odom_current) = euler_from_quaternion(current_orientation_list_odom)
    fi_current = yaw_odom_current


    previous_orientation_list_odom = [previous_orientation_x, previous_orientation_y, previous_orientation_z, previous_orientation_w]
    (roll_odom_previous, pitch_odom_previous, yaw_odom_previous) = euler_from_quaternion(current_orientation_list_odom)
    fi_previous = yaw_odom_previous


    #print("current_orientantion_z", current_orientation_z)         ## DEBUG:
    #print("previous_orientation_z", previous_orientation_z)            ## DEBUG:
    #print("Current position x is ", current_position_x)             ## DEBUG:
    #print("Current position y is ", current_position_y)             ## DEBUG:

    y_array_diff = current_position_y - previous_position_y
    x_array_diff = current_position_x - previous_position_x
    gamma = np.arctan2(y_array_diff, x_array_diff)

    # MOTION MODEL IS HERE !!
    if math.sqrt(y_array_diff**2+x_array_diff**2) < 0.5 :
        motion_model(particles)

    # After calculating gamma SWAPPING OF VARIABLES !
    #===================================================
    previous_position_x = current_position_x
    previous_position_y = current_position_y
    previous_orientation_x = current_orientation_x
    previous_orientation_y = current_orientation_y
    previous_orientation_z = current_orientation_z
    previous_orientation_w = current_orientation_w


def minmax(val_list):
    '''
    DESCRIPTION
    Function to create range of x and y trajectories.
    It basically adds the smallest number in the range and the max as well to a list.
    '''
    min_val = min(val_list)
    max_val = max(val_list)
    return (min_val, max_val)



# PARTICLE FILTER STARTS HERE
#================================

# UNIFOM PARTICLES
#===================

def create_uniform_particles(x_range, y_range, hdg_range, num_of_particles):
    global particles
    global particle_heading_list


    '''
    Create Uniformly Distributed Particles

    PARAMETERS
     - x_range:             Interval of x values for particle locations
     - y_range:             Interval of y values for particle locations
     - hdg_range:           Interval of heading values for particles in radians
     - num_of_particles:    Number of particles

    DESCRIPTION
    Create N by 4 array to store x location, y location, and heading
    of each particle uniformly distributed. Take modulus of heading to ensure heading is
    in range (0, 2*pi).

    Returns particle locations and headings
    '''
    particles = np.empty((num_of_particles, 3))           #Return a new array of given shape and type, without initializing entries.
    particles[:, 0] = uniform(x_range[0], x_range[1], size = num_of_particles)
    particles[:, 1] = uniform(y_range[0], y_range[1], size = num_of_particles)
    particles[:, 2] = uniform(hdg_range[0], hdg_range[1], size = num_of_particles)
    #particles[:, 2] %= 2 * np.pi                                   # Conversion of heading to radians
    print("particles of X axis in create ", particles[:,0])
    #print("particles of Y axis in create ", particles[:,1])
    #print("particles of angle in create ", particles[:,2])



    return particles


# MOTION MODEL
# ===============
def motion_model(particles):
    global x_array_diff
    global y_array_diff
    global current_orientation_z
    global previous_orientation_z
    global fi_current
    global fi_previous
    alpha1 = 0.04
    alpha2 = 0.04
    alpha3 = 0.4
    alpha4 = 0.04

    '''
    Create Uniformly Distributed Particles

    PARAMETERS
     - Particles :                  Particles location

    DESCRIPTION

    MOTION MODEL ASSOCIATED With noise
    Returns particle new locations and headings
    '''



    num_of_particles = len(particles)
    delta_translation = math.sqrt((x_array_diff)**2 + (y_array_diff)**2)
    delta_rotation_1 = (gamma) - (fi_previous)
    delta_rotation_2 = fi_current - fi_previous - delta_rotation_1

    std1 = alpha1 * abs(delta_rotation_1) + alpha2 * abs(delta_translation)     #sa7
    std2 = (alpha3 * abs(delta_translation)) + (alpha4 * (abs(delta_rotation_1) + abs(delta_rotation_2)))
    std3 = (alpha1 * abs(delta_rotation_2)) + (alpha2 * abs(delta_translation))

    mu = 0
    s1 = np.random.normal(mu, std1, num_of_particles)
    s2 = np.random.normal(mu, std2, num_of_particles)
    s3 = np.random.normal(mu, std3, num_of_particles)

    delta_rotation_1_hat = delta_rotation_1 + s1
    delta_translation_hat = delta_translation + s2
    delta_rotation_2_hat = delta_rotation_2 + s3


    particles[:, 0] += delta_translation_hat * (np.cos(fi_previous + delta_rotation_1_hat))
    particles[:, 1] += delta_translation_hat * (np.sin(fi_previous + delta_rotation_1_hat))
    particles[:, 2] += fi_previous + delta_rotation_1_hat + delta_rotation_2_hat


#SENSOR MODEL - UPDATE WEIGHTS OF PARTICLES
#=============================================

def update(particles):
    global desc
    global number_of_matches
    global weights

    weights_of_current = []
    print ("num of particles beginning of sensor model", len(particles))
    for x, y, angle in zip(particles[:, 0], particles[:, 1], particles[:, 2]):
        #print("x in sensor model is ",x)
        d = np.sqrt(x*x+y*y)
        c = np.cos(angle)
        s = np.sin(angle)

        alpha = np.arctan2(y,x)
        tvec = np.array([d*np.cos(alpha-angle), d*np.sin(alpha-angle), 0], dtype = float).reshape(3,1)
        R_matrix = np.array([
        [ c, -s, 0],
        [ s, c, 0],
        [ 0, 0, 1]
        ])

        R_t_matrix = np.concatenate((np.transpose(R_matrix), 0*-tvec), axis =1)
        projection_matrix = np.dot(camera_matrix, R_t_matrix)
        sum_of_distances = 0
        good_matches = list()
        map_feature_count = 0
        sum_of_euclidean_distances = 0
        sum_of_manhattan_distances = 0

        for i in range(len(Xsnew)):
            x_features = Xsnew[i] - x
            y_features = Ysnew[i] - y
            z_features = Zsnew[i] - 0.3
            feature_coordinates = np.array([[x_features],[y_features],[z_features],[1]], dtype=float)
            position_of_features = np.dot(projection_matrix, feature_coordinates)
            u, v, w = position_of_features[0], position_of_features[1], position_of_features[2]
            u_normalized = u / w
            v_normalized = v / w
            if 0 < u_normalized < image_height and 0 < v_normalized < image_width:
                map_feature_count += 1
                #print("U normalized is {} and v normailzed is {} from particles in position x is {} y is {}  z is {}". format(u_normalized, v_normalized, x,y,angle))  #DEBUG:
                #print ("feature {}, {},{} and moved {},{},{}". format(x_features, y_features, z_features, Xsnew[i], Ysnew[i], Zsnew[i]))                                # DEBUG:
                virtual_camera_desc = desc_features[i]
                #print("X features that is seen is {} and virtual camera desc  is {}". format(x_features, virtual_camera_desc))             # DEBUG:
                virtual_camera_desc_array = np.array([virtual_camera_desc], dtype = np.uint8)
                bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)
                matches = bf.match(desc, virtual_camera_desc_array)
                for m in matches:
                    if m.distance < 25:
                        img1_idx = m.queryIdx
                        (x1, y1) = kps[img1_idx].pt
                        (x2, y2) = (u_normalized, v_normalized)
                        euclidean_distance = math.sqrt(((x1 - x2)**2) + ((y1 - y2)**2))
                        sum_of_euclidean_distances += euclidean_distance
                        #print("euclidean_distance", euclidean_distance)        # DEBUG:
                        #print("sum euclidean_distance", sum_of_euclidean_distances)        # DEBUG:

                        # Manhattan distance        # DEBUG:
                        #======================
                        # manhattan_distance = abs(x1 - x2) + abs(y1 - y2)
                        # sum_of_manhattan_distances += manhattan_distance

                        good_matches.append(m)
                        sum_of_distances += m.distance

        number_of_matches = len(good_matches)
        print ("num of matches ", number_of_matches)

        try:        # TO AVOID DIVISION OF ZEROS IN CASE IF NO MATCHES ARE FOUND!
            #weights_of_each_particle = 0.001+number_of_matches
            #weights_of_each_particle = (0.001 + number_of_matches**2/sum_of_euclidean_distances)
            weights_of_each_particle = (0.001 + number_of_matches/sum_of_euclidean_distances)
            weights_of_current.append(weights_of_each_particle)
        except ZeroDivisionError:
            weights_of_each_particle = (0.001 + number_of_matches)
            weights_of_current.append(weights_of_each_particle)

    weights_of_current = np.array(weights_of_current)
    weights = (weights_of_current) / sum(weights_of_current)





#RESAMPLING OF Particles - LOW VARIANCE METHOD
#=================================================

def resample(weights):


    N = len(weights)
    #print("N is in the resample before start ", N)                 # DEBUG:
    # make N subdivisions, and chose a random position within each one
    positions = (random(N) + range(N)) / N
    #print("positions are", positions)          # DEBUG:
    #print ("weights in resampling:", weights)  # DEBUG:
    indexes = np.zeros(N, 'i')
    cumulative_sum = np.cumsum(weights)
    #print("cumulative_sum array len is {} and positions is {}". format(len(cumulative_sum),len(positions)))    # DEBUG:

    i, j = 0, 0
    while i < N:
        #print ("pos {} @ {}  cumsum {} @ {}". format( positions[i] , i, cumulative_sum[j],j))  # DEBUG:
        if positions[i] < cumulative_sum[j]:
            indexes[i] = j
            #print ("I choose " , j)            # DEBUG:
            i += 1
        else:
            j += 1
    return indexes




def initialize_small_percentage(x_range, y_range, hdg_range, num_of_particles):
        global particles
        global particles_small_percentage
        # Take 5% of the particles which is inside particles and distribute them... This helps in increasing the effeciency of the localization
        percent_of_particles = num_of_particles - ((90 * num_of_particles)/100)
        particles_small_percentage = np.empty((percent_of_particles , 3))
        particles_small_percentage[:, 0] = uniform(x_range[0], x_range[1], size = percent_of_particles)
        particles_small_percentage[: ,1] = uniform(y_range[0], y_range[1], size = percent_of_particles)
        particles_small_percentage[:, 2] = uniform(hdg_range[0], hdg_range[1], size = percent_of_particles)
        particles_small_percentage[:, 2] %= 2 * np.pi


#=================================================== THE FILTER LOOP FUNCTIONS BELOW ======================================
# CREATE THE PARTICLES ---
#==========================
def intialize_particle_filter(num_of_particles, do_plot = True, plot_particles = False):
    global particles
    range_of_particles_in_x = (0,9)
    range_of_particles_in_y = (-6,6)
    hdg_range = (-3.18, 3.18)               # from 0 to 360 in radians
    particles = create_uniform_particles(range_of_particles_in_x, range_of_particles_in_y, hdg_range, num_of_particles)
    #print ("particles ", particles)                    # DEBUG:




# LOOP THE SENSOR MODEL AND RESAMPLING METHOD --- THIS LOOP IS CALLED IN THE CALL BACK IMAGE
#===========================================================================================

def particle_filter_loop():
    global particles
    global current_position_x
    global current_position_y
    global particles_small_percentage
    global current_x_position_real
    global current_y_position_real
    global current_z_position_real
    global current_x_orientation_real
    global current_y_orientation_real
    global current_z_orientation_real
    global current_w_orientation_real


    # SENSOR model
    #==============

    update(particles)


    # COLOUR map
    #=============

    max_weight = np.max(weights)
    jet = cm = plt.get_cmap('jet')
    cNorm  = colors.Normalize(vmin=0, vmax = max_weight)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
    colorVal = scalarMap.to_rgba(weights)
    #print ("weights", weights)              # DEBUG:
    #print ("color vla ", colorVal)          # DEBUG:




    plt.figure("Before Resampling")
    p1 = plt.scatter(particles[:, 0], particles[:, 1],  color = colorVal)
    p2 = plt.scatter(current_x_position_real, current_y_position_real, s = 50, color= 'red')
    MAX = 3
    for direction in (-1, 1):
        for point in np.diag(direction * MAX * np.array([1,1,1])):
            plt.plot([point[0]], [point[1]], [point[2]], 'w')
    p3 = plt.scatter(Xsnew, Ysnew, marker = "+" , color = "magenta")
    plt.legend([p1, p2, p3], ['Particles', 'Robot Position',"Features"], loc=4, numpoints=1)
    plt.show()



    # resampling method
    #==================
    indexes =  stratified_resample(weights)
    particles = particles[indexes,:]


    # initialize_small_percentage again for better results
    #======================================================
    #initialize_small_percentage(minmax(Xsnew), minmax(Ysnew), (-3.18, 3.18), num_of_particles= num_of_particles)       # DEBUG:


    plt.figure("After Resampling")
    p1 = plt.scatter(particles[:,0],particles[:,1], color ='blue')
    p2 = plt.scatter(current_x_position_real, current_y_position_real ,s = 50, color= 'red')
    MAX = 3
    for direction in (-1, 1):
        for point in np.diag(direction * MAX * np.array([1,1,1])):
            plt.plot([point[0]], [point[1]], [point[2]], 'w')
    p3 = plt.scatter(Xsnew, Ysnew, c= "magenta", marker = "+")
    plt.legend([p1, p2, p3], ['Particles', 'Robot Position',"Features"], loc=4, numpoints=1)
    plt.show()



intialize_particle_filter(num_of_particles = num_of_particles, do_plot = True, plot_particles = True)


def callback_image(msg):
    global desc
    global kps
    detector = cv2.AKAZE_create()
    bridge = CvBridge()
    cv_image = bridge.imgmsg_to_cv2(msg, "bgr8")
    image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    (kps,desc) = detector.detectAndCompute(image, 0)
    cv2.drawKeypoints(image, kps, image,(0,1000,0))
    rospy.loginfo("Timing images")
    particle_filter_loop()


def main():
    rospy.Subscriber("/odom", Odometry, callback_position_odom)
    rospy.Subscriber("/vrpn_client_node/turtle9/pose", PoseStamped, callback_position_vicon)
    rospy.Subscriber("/pylon_camera_node/image_akaze", Image, callback_image)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")

if __name__ == '__main__':
    rospy.init_node('particle_filter_script', anonymous=True)
    main()



    # ================Clusstering=========
    #=========================================



    # K means clusstering
    #=========================
    # num_of_clusters = 3
    # #num_of_clusters = 4
    # kmeans = KMeans( init = "random", n_clusters = num_of_clusters, n_init = 10, max_iter = 400, random_state = 42)
    # kmeans.fit(particles)
    # labels = kmeans.labels_
    # # In centroid it gives ndarray of shape (n_clusters, n_features) .. So first and second values are X AND Y coordinates
    #
    # centroids = kmeans.cluster_centers_
    # numbers = centroids.flatten().tolist()
    # numbers.extend(real_position_of_the_robot)
    # print("type of list of centroids ", centroids)
    # #print("centroids are ", centroids)
    # print("REAL robot position is ", real_position_of_the_robot)
    # print("centroids of number one is ", centroids[0,:])
    # print("centroids of number one is ", centroids[1,:])
    # print("centroids of number one is ", centroids[2,:])
    #
    # # SAVE CENTROID FILES
    # f1 = open("centroids_4_clusters.txt", "a")
    # f1.write(','.join(str(v) for v in numbers) + '\n')
    # f1.close()
    #
    # for i in range(num_of_clusters):
    #     ds = particles[np.where(labels == i)]
    #     plt.plot(ds[:, 0],ds[:, 1],'o')
    #     lines = plt.plot(centroids[i, 0], centroids[i, 1], 'kx')
    #     plt.setp(lines, ms = 15.0)
    #     plt.setp(lines, mew = 2.0)
    #     plt.scatter(current_x_position_real, current_y_position_real, s = 50, c='black')
    # plt.show()
