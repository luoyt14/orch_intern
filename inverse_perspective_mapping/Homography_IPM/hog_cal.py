import numpy as np
import cv2
import os
import yaml
import time
import math

def calc_dist(M, u, v, ship_pitch, basic_pitch, K):
    """
    逆投影：计算点在水面上的位置
    输入：
    M： 单应性矩阵 numpy(3*3)
    u,v: 输入点像素位置
    ship_pitch: 船当前pitch值
    basic_pitch: 船基础pitch值
    K：相机内参矩阵
    输出: 
    地面上的坐标x，y
    """
    angle = np.radians(basic_pitch + ship_pitch)
    R1 = np.array([[1, 0, 0],
                [0, math.cos(angle), math.sin(angle)],
                    [0, -1 * math.sin(angle), math.cos(angle)]])
    K = np.matrix(K)
    H1 = K * R1 * K.I
    pos_1 = np.array([u,v,1]).reshape(3,1)
    pos_2 = np.dot(H1, pos_1)
    pos_2[:,0] /= pos_2[2,0]
    pos_2[2,0] = 1
    pos_g = np.mat(M).I * pos_2
    return pos_g[0]/pos_g[2], pos_g[1]/pos_g[2]

# 读取yaml文件
def get_yaml_data(yaml_file):
    # 打开yaml文件
    print("***获取yaml文件数据***")
    file = open(yaml_file, 'r', encoding="utf-8")
    file_data = file.read()
    file.close()
    # 读取yaml文件
    data = yaml.load(file_data)
    return data

homography_yaml_path = 'homography_matrix.yaml'  # yaml路径
yaml_data = get_yaml_data(homography_yaml_path)
hg_mat = np.array(yaml_data['homography_matrix'])

K_yaml_path = 'calibration_matrix.yaml'  # yaml路径
yaml_data = get_yaml_data(K_yaml_path)
K = np.array(yaml_data['camera_matrix'])