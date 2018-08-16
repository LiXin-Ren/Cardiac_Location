import nibabel as nib
import numpy as np
from math import cos, sin
from PIL import Image
import scipy.misc
#import cv2
import random
from mpl_toolkits.mplot3d import Axes3D

file_dir = "training_axial_full_pat0.nii.gz"
data = nib.load(file_dir)

alpha = 20      #Z轴旋转
theta = 10      #Y轴旋转

rawData = data.get_fdata()                  #初始像素数组
Z = np.array([[cos(alpha), -sin(alpha), 0], [sin(alpha), cos(alpha), 0], [0, 0, 1]])
Y = np.array([[cos(theta), 0, sin(theta)], [0, 1, 0], [-sin(theta), 0, cos(theta)]])
transform = np.dot(Y, Z)                    #仿射矩阵
shapeArray = rawData.shape
#newArray = np.zeros((100, 100, 100, 3))        #待插值矩阵
"""
newArray = np.zeros((1000000, 3))
n = 0
for i in range(100):
    for j in range(100):
        for k in range(100):
            axis = np.dot(transform, np.array([i, j, k]))
            newArray[n,:] = axis
            n += 1
"""
from scipy.interpolate import griddata

points = np.zeros((shapeArray[0]*shapeArray[1]*shapeArray[2], 3))       #坐标
value = np.zeros_like(points)            #坐标对应的像素值
m = 0                                    #计数
for i in range(shapeArray[0]):
    for j in range(shapeArray[1]):
        for k in range(shapeArray[2]):
            points[m] = [i, j, k]
            value[m] = rawData[i,j,k]
            m += 1


print("points: ", points.shape)
print("rawData:  ", value.shape)
#print("newArray: ", newArray.shape)

#np.save('newArray',newArray)
newArray = np.load("newArray.npy")

print("newArray: ", newArray.shape)

grid_z0 = griddata(points, value, (newArray[100][0], newArray[100][1], newArray[100][2]), method='nearest')
print(grid_z0.shape)
print(grid_z0)
#np.save('1', grid_z0)
#print(grid_z0)
"""
#header
#data_header = data.header
# array = data.get_fdata() #像素数组文件
# print("data Shape:", array.shape)
#
# random.seed(1)


#scipy.misc.imsave('x.jpg', array[100, :, :])


#affineArray = data.affine
# M = affineArray[:3, :3]
# abc = affineArray[:3, 3]

# pyplot.imshow(array[20, :, :])

# def f(i, j, k):
#     return M.dot([i,j,k]) + abc

# originCenter = (np.array(data.shape) - 1)/2       #中心点的坐标
# print(originCenter)
# print(f(originCenter[0], originCenter[1], originCenter[2]))
# [ 63. 103.  70.]
# [-15.78507102  -2.37627125  32.94905376]

#
# print(array[1,1,1])
# print(f(1,1,1))
# 0.0
# [-77.19500011  52.51958179 123.26158619]

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
data = np.load("1.npy")
ax.scatter(data[0], data[1], data[2])
plt.show()
"""
