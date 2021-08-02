import cv2
from PIL import Image
import numpy as np
import json
import matplotlib.pyplot as plt
import os
import glob
import tqdm
from temp import removeDuplicateSimilar1, myUnique, getCurve, getYawAim
IMAGE_PATH = "Picture"
SEGPATH = "Picture_segformer_1"

# 如果某列有多个边缘点，则认为有圈
# max_count代表某列最多的边缘点-1
# max_num代表超过3个边缘点的列数
def findMaxConsecutiveZeros(nums):
    max_num, count, max_count = 0, 0, 0
    for num in nums:
        if num == 0:
            count += 1
        else:
            if count >= 2:
                max_num += 1
            count = 0
        max_count = max(count, max_count)
    return max_count, max_num

def isCurveAbnormal(points):
    res = removeDuplicateSimilar1(points, 20) # 筛掉一些点，否则曲率很大，这里筛选距离设为20，可修改
    tangentAngle = getYawAim(res)
    curveArr = getCurve(res, tangentAngle)
    if np.sum(np.abs(curveArr) >= 0.5) >= 2:
        return True
    else:
        return False


def process_seg(filename, dist_similar=10):
    # 读取分割图
    seg = cv2.imread(os.path.join(SEGPATH, filename))
    is_normal = True
    # 感觉高斯滤波的效果一般，中值滤波也一般，腐蚀膨胀也一般，都注释掉了
    # seg = cv2.GaussianBlur(seg, (3,3), 0)
    # seg = cv2.erode(seg, np.ones((3,3),np.uint8), iterations=3)
    # seg = cv2.dilate(seg, np.ones((3,3),np.uint8), iterations=3)
    h, w, _ = seg.shape
    # 提取分割图的边缘
    edges = cv2.Canny(seg, 100, 200)
    # 得到边缘的横纵坐标
    points = np.nonzero(edges.transpose(1,0))
    ww, hh = points
    # print(np.diff(ww))
    # 判断是否有内部圈，如果有写入文件
    max_count, max_num = findMaxConsecutiveZeros(np.diff(ww))
    if max_count >= 2 and max_num >= 10:
        is_normal = False
        with open(IMAGE_PATH + "_abnormal.txt", "a") as g:
            g.write(filename + "\n")
    
    # 得到边缘点列并去掉相同横坐标的点
    res = np.vstack((ww, hh)).transpose(1,0)
    _, ic = myUnique(res[:, 0], return_inverse=True, axis=0)
    del_flag = np.append(np.array([1]), np.diff(ic))  # 不删除队列第一个元素
    del_index = np.argwhere(del_flag == 0)
    res = np.delete(res, del_index, axis=0)
    # 计算每个点的曲率用于筛选
    # tangentAngle = getYawAim(res)
    # curveArr = getCurve(res, tangentAngle)
    if is_normal and isCurveAbnormal(res): # 之前没有因为内部圈异常写入到文件中，并且曲率异常
        with open(IMAGE_PATH + "_abnormal.txt", "a") as g:
            g.write(filename + "\n")

    res = removeDuplicateSimilar1(res, dist_similar)
    res = res.tolist()
    # 得到的列表添加上图像的边界点，这里假设水面在图像的下方，因此添加了图像右下角和左下角的点
    res.append([w-1, h-1])
    res.append([0, h-1])

    # 生成json格式的数据
    json_label = {}
    json_label["version"] = "4.5.9"
    json_label["flags"] = {}
    json_shape = {}
    json_shape["label"] = "lane"
    json_shape["points"] = res
    json_shape["group_id"] = None
    json_shape["shape_type"] = "polygon" # 多边形类型的标记
    json_shape["flags"] = {}
    json_label["shapes"] = [json_shape]
    json_label["imagePath"] = filename
    json_label["imageData"] = None
    json_label["imageHeight"] = h
    json_label["imageWidth"] = w
    jsonData = json.dumps(json_label, indent=4)
    with open(os.path.join(IMAGE_PATH, filename[:-3]+"json"), "w") as f:
        f.write(jsonData)

if __name__ == '__main__':
    for filename in tqdm.tqdm(os.listdir(SEGPATH), ascii=" \uf307", colour='MAGENTA'):
        process_seg(filename, 50)
        # break
