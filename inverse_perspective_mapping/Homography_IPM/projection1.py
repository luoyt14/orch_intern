
'''
给定内外参数矩阵，将点云投影到对应图像，鱼眼畸变，等距模型
'''
import cv2
import numpy as np
import math
import os


# D = np.array([-0.029934283651861, 0.066661510541701, 0.0, -0.0])

# R_radar = np.array([[0, 1, 0],
#                     [-1,  0, 0],
#                     [0,  0, 1]], dtype=np.float32)

# T_radar = np.array([[-30],
#               [15],
#               [100]], dtype=np.float32)

# R = np.array([
#    [-0.0463,   -1.0129,    0.0132],
#    [-0.1755,    0.0117,   -0.9597],
#     [0.9920,   -0.0098,   -0.1612],
# ])
# T = np.array([[104.3001],    [8.5644],   [65.5156]], dtype=np.float32)
# K = np.array([
#     [4.136942081318755e+02, 0.00000000000000000, 3.358328323628881e+02],
#     [0.00000000000000000, 4.141860769962941e+02, 1.924334993449346e+02],
#     [0.00000000000000000, 0.00000000000000000, 1.000000000000000000]
# ], dtype=np.float32)


K = np.array([
    [326.8655, 0.0000, 342.3895],
    [0.0000, 326.7865, 277.7755],
    [0.0000, 0.0000, 1.0000]
], dtype=np.float32)
D = np.array([-0.029934283651861, 0.066661510541701, 0.0, -0.0])

R_radar = np.array([[0, 1, 0],
                    [-1,  0, 0],
                    [0,  0, 1]], dtype=np.float32)

T_radar = np.array([[-40], [0], [110]], dtype=np.float32)

R = np.array([[-0.0372, -1.0420, -0.0003],
              [-0.1948, -0.0016, -0.9556],
              [0.9466, 0.0305, -0.1020]], dtype=np.float32)


T = np.array([[-78.6794],
              [80.4925],
              [-33.7904]], dtype=np.float32)


def radar_to_lidar(R, T, points):   # 重映射
    points = points.T
    points = points - T
    temp = np.dot(R, points)
    temp = np.array(temp, dtype=np.float32)
    return temp.T


def repro(camera_matrix, R, T, points):   # 重映射
    points1 = points.copy()
    points = points.T
    points = points - T
    temp = np.dot(R, points)
    temp = np.dot(camera_matrix, temp)

    temp /= temp[2]        # 相机坐标系下归一化坐标
    u = temp[0]
    v = temp[1]

    imgpoints = np.concatenate((np.expand_dims(u, axis=1),np.expand_dims(v, axis=1)), axis=1)
    imgpoints = np.array(imgpoints, dtype=np.int32)
    return imgpoints


def per_image(point_data, w, h, pointRotM, transferM, pflag=False):
    # 读取图片及对应点云数据
    data = point_data
    if len(data.shape) == 1:
        data = np.expand_dims(data, axis=0)
    data_shape = data.shape
    if len(data) == 0 or data_shape[1]==0:
        return [], []
    data = np.matmul(data, pointRotM)
    data = data[np.logical_not((data[:, 0]==0)*(data[:, 1]==0)*(data[:, 2]==0))]
    data += transferM
    data *= 1000

    ''' 投影点云到图像,自定义无畸变矫正 '''
    if pflag:
        lidar_points = radar_to_lidar(R_radar, T_radar, data)
    else:
        lidar_points = data
    imgpts = repro(K, R, T, lidar_points)
    if imgpts.size == 0:
        return [], []
    return imgpts, lidar_points




