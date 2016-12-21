#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import sys


# 计算某点的中心坐标
def cal_center(points):
    x = (points[0][0] + points[1][0] + points[2][0] + points[3][0]) / 4
    y = (points[0][1] + points[1][1] + points[2][1] + points[3][1]) / 4
    return np.array([np.float32(x), np.float32(y)], np.float32)


# 判断某点是否在给定端点围成的多边形内
def pnpoly(roi_points, test_point):
    result = False

    min_x = sys.maxint
    min_y = sys.maxint
    max_x = 0
    max_y = 0

    for point in roi_points:
        if point[0] < min_x:
            min_x = point[0]
        if point[0] > max_x:
            max_x = point[0]
        if point[1] < min_y:
            min_y = point[1]
        if point[1] > max_y:
            max_y = point[1]

    if test_point[0] < min_x or test_point[0] > max_x or test_point[1] < min_y or test_point[1] > max_y:
        return False

    j = len(roi_points) - 1
    for i in range(len(roi_points)):
        if (((roi_points[i][1] > test_point[1]) != (roi_points[j][1] > test_point[1])) and (
                    test_point[0] < (roi_points[j][1] - roi_points[i][0]) * (test_point[1] - roi_points[i][1]) / (
                            roi_points[j][1] - roi_points[i][1]) + roi_points[i][0])):
            result = not result

    return result
