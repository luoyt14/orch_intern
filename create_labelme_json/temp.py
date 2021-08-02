# coding: utf-8
import math
import os
from math import asin, acos, pow, sqrt, sin, cos, pi, atan2
import numpy as np

''' *****************************计算曲率******************************* '''
def getYawAim(NorthEast):
    """
    计算每点处期望控制角
    输入:
    NorthEast*2维np.array数组
    输出:
    tangent_angle是n*1维np.array数组,表示每点处的曲线切向角 0~360度角对应从一四三二象限
    """
    # 计算切线方向角
    tangent_angle = np.rad2deg(np.arctan2(
        np.diff(NorthEast[:, 1]), np.diff(NorthEast[:, 0])))
    # 角度限幅至0~360
    tangent_angle[tangent_angle < 0] += 360
    tangent_angle = np.append(tangent_angle, tangent_angle[-1])  # 补充位数
    return tangent_angle

def getCurve(sidePoints, tangentAngle, predPoints=2):
    """返回曲线平均曲率，
    tangentAngle:曲线切线角，
    index:当前索引值,
    presPoints表示前后检索点个数
    """
    errAngle = angleMatDiff(tangentAngle, np.append(tangentAngle[1:], tangentAngle[-1]))
    NE = sidePoints
    dist = np.linalg.norm(np.diff(NE, axis=0), ord=2, axis=1)
    dist = np.append(dist,dist[-1])
    curveArr = np.convolve( errAngle, [1]*predPoints, mode='same') / np.convolve(dist, [1]*predPoints, mode='same')
    curveArr = np.around(curveArr, 1)
    curveArr = smooth(curveArr)
    return curveArr

def angleMatDiff(angT, ang):
    '''比较两个角度差，带符号'''
    angle_diff = []
    for i in range(len(angT)):

        if angT[i] - ang[i] > 180:
            angle_diff.append(angT[i] - (ang[i] + 360))
        elif angT[i] - ang[i] < -180:
            angle_diff.append(angT[i] - (ang[i] - 360))
        else:
            angle_diff.append(angT[i] - ang[i])
    return np.array(angle_diff)

def smooth(data, depth=2):
    """
    :param data: 存放平滑列表数据
    :param deepth: 平滑指数，deepth=0,表示平滑度为1，即不做平滑处理
    :return: 平滑后的数据
    """
    return np.convolve(data, [1/depth]*depth, mode='same')


''' *****************************降采样******************************* '''
def removeDuplicateSimilar1(sidePoints, dist_similar=1):
    """
    删除点队列中两两相邻的重复点和相近点
    输入:
    sidePoints:点队列,n*2维的np.array数组
    dist_similar:距离阈值,删去距离小于dist_similar的点
    输出:
    sidePoints:处理后的点队列
    """
    latlng = sidePoints[:, 0:2]
    # _, ic = myUnique(latlng, return_inverse=True, axis=0)
    _, ic = myUnique(latlng, return_inverse=True, axis=0)
    del_flag = np.append(np.array([1]), np.diff(ic))  # 不删除队列第一个元素
    del_index = np.argwhere(del_flag == 0)
    latlng = np.delete(latlng, del_index, axis=0)
    sidePoints = np.delete(sidePoints, del_index, axis=0)
    # 去除相近数据
    lat0, lng0 = latlng[0, 0], latlng[0, 1]
    NE = sidePoints[:, 0:2]
    dist = np.linalg.norm(np.diff(NE, axis=0), ord=2, axis=1)
    while np.sum(dist[0:-1] < dist_similar):
        del_index = []
        dist_temp = dist[0]
        for ii in range(1, len(dist)):
            if dist_temp < dist_similar:
                del_index.append(ii)
                dist_temp += dist[ii]
            else:
                dist_temp = dist[ii]
        latlng = np.delete(latlng, del_index, axis=0)
        sidePoints = np.delete(sidePoints, del_index, axis=0)
        NE = sidePoints[:, 0:2]
        dist = np.linalg.norm(np.diff(NE, axis=0), ord=2, axis=1)
    return sidePoints



def myUnique(ar, return_index=False, return_inverse=False, return_counts=False, axis=None):
    """
    低版本numpy适用的去重,在numpy v1.11.0上测试ok,输入参数与numpy.unique相同
    """
    ar = np.asanyarray(ar)
    if axis is None:
        ret = _unique1d(ar, return_index, return_inverse, return_counts)
        return _unpack_tuple(ret)
    # axis was specified and not None
    ar = np.swapaxes(ar, axis, 0)
    orig_shape, orig_dtype = ar.shape, ar.dtype

    def reshape_uniq(uniq):
        uniq = uniq.view(orig_dtype)
        uniq = uniq.reshape(-1, *orig_shape[1:])
        uniq = np.swapaxes(uniq, 0, axis)
        return uniq
    ar = ar.reshape(orig_shape[0], -1)
    ar = np.ascontiguousarray(ar)
    dtype = [('f{i}'.format(i=i), ar.dtype) for i in range(ar.shape[1])]
    consolidated = ar.view(dtype)
    output = _unique1d(consolidated, return_index=False, return_inverse=True,
                       return_counts=False)
    output = (reshape_uniq(output[0]),) + output[1:]
    return _unpack_tuple(output)

def _unpack_tuple(x):
    """ Unpacks one-element tuples for use as return values """
    if len(x) == 1:
        return x[0]
    else:
        return x

def _unique1d(ar, return_index=False, return_inverse=False, return_counts=False):
    """
    Find the unique elements of an array, ignoring shape.
    """
    ar = np.asanyarray(ar).flatten()

    optional_indices = return_index or return_inverse

    if optional_indices:
        perm = ar.argsort(kind='mergesort' if return_index else 'quicksort')
        aux = ar[perm]
    else:
        ar.sort()
        aux = ar
    mask = np.empty(aux.shape, dtype=np.bool_)
    mask[:1] = True
    mask[1:] = aux[1:] != aux[:-1]

    ret = (aux[mask],)
    if return_index:
        ret += (perm[mask],)
    if return_inverse:
        imask = np.cumsum(mask) - 1
        inv_idx = np.empty(mask.shape, dtype=np.intp)
        inv_idx[perm] = imask
        ret += (inv_idx,)
    if return_counts:
        idx = np.concatenate(np.nonzero(mask) + ([mask.size],))
        ret += (np.diff(idx),)
    return ret