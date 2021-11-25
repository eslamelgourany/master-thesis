#!/usr/bin/python

#This script is to visualize the txt file "map" which is the output from the mapping script
#Input: TXT file which will have [X Y Z DESCRIPTOR]
#Output: 3D visualization or 2D visualization of the features
# Author : Eslam Elgourany
# Contact : eslamelgourany@hotmail.com
# Thesis source code, CVUT, Prague, Czech Republic

#Import Libraries
#==================

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import mpl_toolkits
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import Axes3D



f = open('map.txt', 'r').readlines()
Xsnew= list()
Ysnew= list()
Zsnew= list()
output_list = []

fig = plt.figure()
for line in f:
    tmp = line.strip().split(",")
    values = [float(v) for v in tmp]
    points4d = np.array(values).reshape((-1, 65))
    # Change below if you want to exclude W to 3
    points3d = points4d[:, :65]
    Xs = points3d[:, 0]/points4d[:,3]
    Ys = points3d[:, 1]/points4d[:,3]
    Zs = points3d[:, 2]/points4d[:,3]
    Ws = points3d[:, 3]/points4d[:,3]

    for i in range(len(Xs)):
        if 2.8 < Zs[i] < 3.6:
            output_list.append([Xs[i],Ys[i],Zs[i]])
        else:
            pass
    for values in output_list:
        Xsnew.append(values[0])
        Ysnew.append(values[1])
        Zsnew.append(values[2])
print("XNEW", Xsnew)
print("YNEW", Ysnew)
print("ZNEW", Zsnew)


#3D Visualization
#====================

ax = fig.add_subplot(111, projection = '3d')
ax.scatter(Xsnew, Ysnew, my_new_Z_list, c = 'r', marker = "o")
MAX = 3
for direction in (-1, 1):
    for point in np.diag(direction * MAX * np.array([1,1,1])):
        ax.plot([point[0]], [point[1]], [point[2]], 'w')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()




#2D Visualization
#==================

# ax = fig.add_subplot(111)
# ax.scatter(Xsnew, Ysnew, my_new_Z_list, c = 'r', marker = "o")
# MAX = 3
# for direction in (-1, 1):
#     for point in np.diag(direction * MAX * np.array([1,1,1])):
#         ax.plot([point[0]], [point[1]], [point[2]], 'w')
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# plt.show()
