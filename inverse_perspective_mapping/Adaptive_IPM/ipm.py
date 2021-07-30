import os
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt


M, N = 480., 640.
TAN_ALPHA_R = 64. / 128.5
TAN_ALPHA_C = 126. / 128.5
H = 0.31 # 摄像头距湖面的高度，单位m
THETA0 = 7. * math.pi / 180 # 摄像头朝下的角度
# THETAP = -1.89 * math.pi / 180
THETAP = 0


def projectX(v, thetap=0):
    fenzi = 1. + np.tan(THETA0) * (1. - 2. * ((v - 1) / (M - 1))) * TAN_ALPHA_R
    fenmu = np.tan(THETA0) - (1. - 2. * ((v - 1) / (M - 1))) * TAN_ALPHA_R
    return H * fenzi / fenmu

def projectY(u, v, thetap=0):
    X = projectX(v, thetap)
    return (2. * (u - 1) / (N - 1) - 1) * TAN_ALPHA_C * X


if __name__ == '__main__':
    img = plt.imread("test2.jpg")
    img_gt = plt.imread("test_seg2-2.jpg")
    # img_gt = cv2.resize(img_gt, (640, 480))
    _, img_gt = cv2.threshold(img_gt,127,255,cv2.THRESH_BINARY)
    edges = cv2.Canny(img_gt, 0, 1)
    v, u = np.nonzero(edges)
    img_edge = img.copy()
    for i in range(len(u)):
        img_edge = cv2.circle(img_edge, (u[i],v[i]), 5, (0,255,0), -1)
    u = u[-300:]
    v = v[-300:]
    # print(u[:20])
    # print(v[:20])
    # print(u[-1:-10:-1], v[-1:-10:-1])
    # print(v.shape)
    X, Y = projectX(v, THETAP), projectY(u, v, THETAP)
    print(X[-1], Y[-1])
    # print(Y[:10])
    # print(X[:10])
    plt.figure(1)
    plt.subplot2grid((2,2),(0,0))
    plt.imshow(img)
    plt.axis('off')
    ax1 = plt.subplot2grid((2,2),(0,1),rowspan=2)
    ax1.scatter(Y, X, c='b',marker='.',s = 10)
    ax1.set_xlim((-6, 6))
    ax1.set_ylim((0, 6))
    plt.subplot2grid((2,2),(1,0))
    plt.imshow(img_edge)
    plt.axis('off')
    plt.show()
    # plt.savefig("test_pred2")
    # plt.figure(2)
    # Y = projectY(u[-1], v)
    # plt.scatter(v, Y)
    # plt.show()