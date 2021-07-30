import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
from projection1 import per_image
import cv2
from hog_cal import *

#Root
txt_root = './data/camera_lidar_imu.txt'
lidar_root = './data/Lidar/' 
pic_root = './data/Picture/' # 原图路径
picres_root = './data/Picture_res/' # 分割结果路径
# Read data
lidar_data = []
camera_data = []
imu_data = []
time_data = []
with open(txt_root, "r") as f:
    for line in f.readlines():
        line = line.replace('\n','')
        data = line.split(" ")
        lidar_data.append(data[0])
        time_data.append(data[1])
        camera_data.append(data[2])
        imu_data.append([data[4],data[5]])

plt.ion()
fig=plt.figure(dpi=120)
# load yaml
homography_yaml_path = 'homography_matrix.yaml'  # yaml路径
yaml_data = get_yaml_data(homography_yaml_path)
hg_mat = np.array(yaml_data['homography_matrix'])

K_yaml_path = 'calibration_matrix.yaml'  # yaml路径
yaml_data = get_yaml_data(K_yaml_path)
K = np.array(yaml_data['camera_matrix'])

def read_lidar(lidarfile):
    with open(lidarfile, "r") as f:
        next(f)
        lidar_point = []
        for line in f.readlines():
            line = line.replace('\n','')
            data = line.split(" ")
            lidar_point.append([float(i) for i in data[:3]])
    lidar_point = np.array(lidar_point)
    return lidar_point[:,0:3]

def Fliter_lidar(lidar_point):
    #激光雷达滤波
    lidar_point = lidar_point[(lidar_point[:,0]>0) * (lidar_point[:,2]<0.1) * (lidar_point[:,1]<0) * (lidar_point[:,2]>0)]
    #lidar_point = lidar_point[(lidar_point[:,0]>0) * (lidar_point[:,2]<0.6) * (lidar_point[:,1]<0) * (lidar_point[:,2]>-0.2)]
    return lidar_point

basic_pitch = 1

for i in range(200,len(lidar_data),1): # 7295 8670
    #Lidar
    lidarfile = os.path.join(lidar_root, (str(lidar_data[i-200]) + '.txt'))
    lidar_point = read_lidar(lidarfile)
    lidar_point = Fliter_lidar(lidar_point)
    #Img
    img_root = os.path.join(pic_root, (str(int(camera_data[i])) + '.jpg'))
    img = cv2.imread(img_root)
    #IMU  正值为船抬头 
    ship_roll = imu_data[i][0]
    ship_pitch = imu_data[i][1]
    #Img_result
    imgres_root = os.path.join(picres_root, (str(int(camera_data[i])) + '.jpg'))
    img_gt = cv2.imread(imgres_root)
    _, img_gt = cv2.threshold(img_gt,127,255, cv2.THRESH_BINARY)
    img_gt = cv2.cvtColor(img_gt, cv2.COLOR_BGR2GRAY)
    h = img_gt.shape[0]
    w = img_gt.shape[1]
    img_0 = np.zeros(img_gt.shape)
    # Extracter bank_route 提取岸线
    contours, _ = cv2.findContours(img_gt, cv2.RETR_EXTERNAL,  cv2.CHAIN_APPROX_SIMPLE) 
    area = []
    for j in range(len(contours)):
        area.append(cv2.arcLength(contours[j], True))
    water_contours = contours[area.index(max(area))]
    water_contours = [point for point in water_contours  if (point[0][0]>=w/2 and point[0][0]<=(w-10)) and point[0][1]<=(h*2/3)]
    for j in range(len(water_contours)):
        img_0[water_contours[j][0][1],water_contours[j][0][0]] = 255
        cv2.circle(img, (water_contours[j][0][0],water_contours[j][0][1]), radius = 1, color = (255,0,0), thickness = -1)
    # IPM result  逆投影
    ipm_result = list()
    for j in range(len(water_contours)):
        ipmx, ipmy = calc_dist(hg_mat, water_contours[j][0][0], water_contours[j][0][1], float(ship_pitch), basic_pitch, K)
        ipm_result.append([ipmx/1000,ipmy/1000])
    ipm_result = np.array(ipm_result).reshape(-1,2)

    """
    ipm_result1 = list()
    for j in range(len(water_contours)):
        ipmx, ipmy = calc_dist_withroll(hg_mat, water_contours[j][0][0], water_contours[j][0][1], float(ship_pitch), basic_pitch, K, float(ship_roll))
        ipm_result1.append([ipmx/1000,ipmy/1000])
    ipm_result1 = np.array(ipm_result1).reshape(-1,2)
    """
    #print(ipm_result)
    
    #Plot
    plt.clf()
    plt.subplot(2,2,1)
    plt.imshow(img)
    plt.axis('off')

    

    plt.title('Roll:' + str(ship_roll) + '  ' + 'Pitch:' + str(ship_pitch))
    ax1 = plt.subplot(1,2,2)
    ax1.scatter(lidar_point[:,1]*(-1), lidar_point[:,0] ,c='b',marker='.',s = 10, label = 'lidar')
    #ax1.scatter(ipm_result1[:,0], ipm_result1[:,1] ,c='g',marker='.',s = 10, label ='camera')
    ax1.scatter(ipm_result[:,0], ipm_result[:,1] ,c='r',marker='.',s = 10, label = ' camera')

    ax1.legend()
    ax1.grid(True)
    ax1.set_xlim((-4, 4))
    ax1.set_ylim(( 0, 6))
    ax1.set_title(str(i))
    
    plt.subplot(2,2,3)
    plt.imshow(img_0)
    plt.axis('off')
    plt.title('Segmentation Result')
    plt.show()
    plt.pause(0.1)

