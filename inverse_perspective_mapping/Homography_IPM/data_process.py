import os
import numpy as np
import cv2
from sklearn.cluster import dbscan
import serial
import threading
import time
import binascii
from mpl_toolkits.mplot3d import Axes3D
from projection1 import per_image
import mmw_radar_config

lock = threading.Lock()
data_queue = []
com_cfg = 'com22'
com_dat = 'com21'
cfg = "./cfg/6843_3d_v2_15m.cfg"
cam = 0
pic_list = []

mmw_radar_config.mmw_radar_config(com_cfg, cfg)

# yaw = 0.5
# pitch = 1
# roll = -1

yaw = 5
pitch = 7
roll = 0

#011
# yaw = 8
# pitch = -2
# roll = 0

#003
# yaw = 2
# pitch = -7
# roll = 0



ryaw = np.radians(yaw)
rpitch = np.radians(pitch)
rroll = np.radians(roll)

pointRotM = np.matmul([[np.cos(ryaw), np.sin(ryaw), 0], [-np.sin(ryaw), np.cos(ryaw), 0], [0, 0, 1]],
                 [[1, 0, 0], [0, np.cos(rpitch), np.sin(rpitch)], [0, -np.sin(rpitch), np.cos(rpitch)]])
pointRotM = np.matmul(pointRotM, [[np.cos(rroll), 0, np.sin(rroll)], [0, 1, 0], [-np.sin(rroll), 0, np.cos(rroll)]])
# transferM = [-0.05, 0.05, 0]
transferM = [0, 0, 0]

def data_receive():
    global data_queue
    serial_front = serial.Serial(com_dat, 921600)
    if serial_front.isOpen():
        serial_front.close()
    serial_front.open()
    data = ""
    while True:
        time.sleep(0.01)
        numBytes = serial_front.inWaiting()
        # print(numBytes)
        if numBytes == 0:
            continue
        dataRaw = serial_front.read(numBytes)
        dataRaw = str(binascii.b2a_hex(dataRaw))[2:-1]
        data += dataRaw
        temp = data.split('0201040306050807')
        curr_time = time.time()
        i = 0 
        lock.acquire()
        while i < len(temp) - 1:
            data_queue.append({'data': temp[i], 'curr_time':curr_time - 0.1*(len(temp)-i-1)})
            if len(data_queue) > 3:
                data_queue.pop(0)
            i += 1
        lock.release()
        data = temp[-1]

def data_handle68(str0):
    str0 = '0201040306050807' + str0
    pointclouds = np.zeros((0, 6))
    if len(str0) < 80:
        return np.array(pointclouds)
    if len(str0) != int(myrightstr(str0[24:32]), 16) * 2:
        return np.array(pointclouds)
    if str0[:16] != '0201040306050807':
        return np.array(pointclouds)
    numDetectedObj=int(myrightstr(str0[56:64]),16)
    if numDetectedObj <= 0:
        return np.array(pointclouds)
    TLVlen = int(myrightstr(str0[88:96]), 16)
    str0 = str0[96:]
    TLVLen_sideInfo = int(myrightstr(str0[TLVlen*2 + 8:TLVlen*2 + 16]), 16)
    SideInfo = str0[TLVlen*2 + 16:TLVlen*2+16+TLVLen_sideInfo*2]
    data_array = np.array([int(myrightstr(str0[i*8:i*8+8]), 16) for i in range(numDetectedObj*4)])
    side_info = np.array([int(myrightstr1(SideInfo[i*4:i*4+4]), 16) for i in range(int(len(SideInfo)/4))])
    exp_num = data_array / 2 ** 23
    exp_num = exp_num.astype(np.int)
    exp_num[exp_num >= 256] -= 256
    exp_num -= 150
    exp_num = 2 ** exp_num.astype(np.float)
    num = data_array % (2 ** 23) + 2 ** 23
    floatnum = num * exp_num
    floatnum[data_array > 2 ** 31] = -floatnum[data_array > 2 ** 31]
    pointclouds = np.reshape(floatnum, (-1, 4))
    side_info = np.reshape(side_info, (-1, 2))
    pointclouds = np.concatenate((pointclouds, side_info), axis=1)
    return np.array(pointclouds)

def myrightstr(str0):
    str1=str0[6:8]+str0[4:6]+str0[2:4]+str0[0:2]
    return str1

def myrightstr1(str0):
    str1=str0[2:4]+str0[0:2]
    return str1

def direct_filter(data, pitch=-8):
    z = -0.3 - data[:, 1] * np.tan(np.radians(pitch))
    data[:, 2] = z
    data = data[data[:, 1] > 0.2]
    data = data[data[:, 1] < 2.1]
    if data.size == 0:
        data = np.zeros((0, 7))
        return data
    _, label = dbscan(data[:, :3], eps=0.5, min_samples=5)
    # data = data[label != -1]

    return data

def data_process(data1, data2, data3, w, h):
    '''
    数据预处理部分：
    包括点云解析，点云投影
    :params data1: 在当前时刻的前2个时刻的数据
    :params data2: 在当前时刻的前1个时刻的数据
    :params data3: 在当前时刻的数据
    :params w: 图片宽度
    :params h: 图片高度
    :return 投影点坐标，点云三维坐标，强度值，多普勒速度值
    '''
    data1 = data_handle68(data1)
    data2 = data_handle68(data2)
    data3 = data_handle68(data3)
    data1 = np.zeros((0, 6)) if data1.size == 0 else data1
    data2 = np.zeros((0, 6)) if data2.size == 0 else data2
    data3 = np.zeros((0, 6)) if data3.size == 0 else data3
    data1 = data1.reshape(-1, 6)
    data2 = data2.reshape(-1, 6)
    data3 = data3.reshape(-1, 6)
    merge_data = np.concatenate((data1, data2, data3), axis=0)
    x = np.expand_dims(np.arange(-0.75, 0.75, 0.01), axis=1)
    z = np.expand_dims(-0.3 * np.ones(len(x)), axis=1)
    y1 = np.expand_dims(1.5 * np.ones(len(x)), axis=1)
    y2 = np.expand_dims(1 * np.ones(len(x)), axis=1)
    y3 = np.expand_dims(2 * np.ones(len(x)), axis=1)
    a = np.expand_dims(-0.35 * np.ones(len(x)), axis=1)
    b = np.expand_dims(-0.35 * np.ones(len(x)), axis=1)
    c = np.expand_dims(-0.35 * np.ones(len(x)), axis=1)
    merge_data1=np.concatenate((x,y1,z, a, b, c),axis=1)
    merge_data2=np.concatenate((x,y2,z, a, b, c),axis=1)
    merge_data3=np.concatenate((x,y3,z, a, b, c),axis=1)
    merge_data_line = np.concatenate((merge_data1, merge_data2, merge_data3), axis=0)
    merge_data = direct_filter(merge_data)
    merge_data_line = direct_filter(merge_data_line)
    # plt.cla()
    # ax.scatter(merge_data[:, 0], merge_data[:, 1], merge_data[:, 2], c='b', s=2)
    # ax.set_xlim((-1, 1))
    # ax.set_ylim((0, 10))
    # ax.set_zlim((-1, 1))
    # plt.show()
    # plt.pause(0.001)
    # p为强度值，points为3维点，v为多普勒速度
    p = merge_data[:, 4] + merge_data[:, 5]
    dis = np.linalg.norm(merge_data[:, :2], axis=1)
    p += 4 * np.log(dis)
    v = merge_data[:, 3]
    if merge_data.size == 0:
        return [], [], [], [], []
    imgpts, points = per_image(merge_data[:, :3], w, h, pointRotM, transferM, True)
    imgpts_line, _ = per_image(merge_data_line[:, :3], w, h, pointRotM, transferM, True)
    max_kind = int(max(points[:, -1]))
    flag = (imgpts[:, 0] < w) * (imgpts[:, 1] < h)
    imgpts = imgpts[flag]
    points = points[flag]
    p = p[flag]
    v = v[flag]
    # print(imgpts.shape, points.shape, p.shape, v.shape)
    return imgpts, points, p, v, imgpts_line

def detect_img(name):
    data = np.loadtxt(name)
    if data.size == 0:
        data = np.zeros((0, 5))
    return data.reshape((-1, 5))

def capture_image():
    global pic_list
    cap = cv2.VideoCapture(cam + cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    try:
      while True:
        ret, img = cap.read()
        start_time = time.time()
        if img is not None:
            [first, second] = str(start_time).split('.')
            second = second.ljust(7, '0')
            name = first + '.' + second + '.jpg'
            data = {'name': name, 'image': img}
            pic_list.append(data)
            if len(pic_list) > 20:
                pic_list.pop(0)
            # print('curr pic time %f' % start_time)
        else:
            print('camera failure')
    except KeyboardInterrupt:
        return

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    ax = plt.subplot(111, projection='3d')
    threading.Thread(target=data_receive).start()
    threading.Thread(target=capture_image).start()

    w = 640
    h = 480
    plt.ion()
    while True:
        if not pic_list:
            time.sleep(0.1)
            continue
        tempimg = pic_list[-1]['image'].copy()
        lock.acquire()
        data_queue_cp = data_queue.copy()
        lock.release()
        # 此处已拿到同一时刻的点云和图片，数据分别为data[i], image
        imgpts, points, p, v, imgpts_line = data_process(data_queue_cp[0]['data'], data_queue_cp[1]['data'], data_queue_cp[2]['data'], w, h)
        for i in range(len(imgpts_line)):
            cv2.circle(tempimg, tuple(imgpts_line[i, :2]), 2, (0, 255, 0), -1)
        for i in range(len(imgpts)):
            cv2.circle(tempimg, tuple(imgpts[i, :2]), 2, (0, 0, 255), -1)
        cv2.imshow('img', tempimg)
        # cv2.waitKey(100)
        if cv2.waitKey(10) & 0xff == ord('a'):
            print('yaw1')
            yaw -= 1
        elif cv2.waitKey(10) & 0xff == ord('d'):
            print('yaw2')
            yaw += 1
        elif cv2.waitKey(10) & 0xff == ord('w'):
            print('pitch1')
            pitch -= 1
        elif cv2.waitKey(10) & 0xff == ord('s'):
            print('pitch2')
            pitch += 1
        elif cv2.waitKey(10) & 0xff == ord('q'):
            print('roll1')
            roll -= 1
        elif cv2.waitKey(10) & 0xff == ord('e'):
            print('roll2')
            roll += 1
        elif cv2.waitKey(10) & 0xff == ord('k'):
            print('back')
            transferM[1] -= 0.05
        elif cv2.waitKey(10) & 0xff == ord('i'):
            print('fornt')
            transferM[1] += 0.05
        elif cv2.waitKey(10) & 0xff == ord('j'):
            print('left')
            transferM[0] -= 0.05
        elif cv2.waitKey(10) & 0xff == ord('l'):
            print('right')
            transferM[0] += 0.05
        else:
            cv2.waitKey(10)
        ryaw = np.radians(yaw)
        rpitch = np.radians(pitch)
        rroll = np.radians(roll)
        print(yaw, pitch, roll, transferM)
        pointRotM = np.matmul([[np.cos(ryaw), np.sin(ryaw), 0], [-np.sin(ryaw), np.cos(ryaw), 0], [0, 0, 1]],
                        [[1, 0, 0], [0, np.cos(rpitch), np.sin(rpitch)], [0, -np.sin(rpitch), np.cos(rpitch)]])
        pointRotM = np.matmul(pointRotM, [[np.cos(rroll), 0, np.sin(rroll)], [0, 1, 0], [-np.sin(rroll), 0, np.cos(rroll)]])
        
