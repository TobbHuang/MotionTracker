#!/usr/bin/python
# -*- coding: utf-8 -*-
import cv2
import numpy as np
import utils
import time


class Target:
    def __init__(self, id, frame, track_window, roi_points):
        # 设置跟踪roi
        self.id = int(id)
        x, y, w, h = track_window
        self.track_window = track_window
        self.roi = cv2.cvtColor(frame[y:y + h, x:x + w], cv2.COLOR_BGR2HSV)
        roi_hist = cv2.calcHist([self.roi], [0], None, [16], [0, 180])
        self.roi_hist = cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

        # 设置kalman
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        self.kalman.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                                               np.float32) * 0.03
        self.measurement = np.array((2, 1), np.float32)
        self.prediction = np.zeros((2, 1), np.float32)
        self.term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
        self.center = None
        self.frame_num = 0
        self.silent_time = 0
        self.roi_points = roi_points
        self.should_remove = False
        self.update(frame)

        self.start_time = time.time()
        self.end_time = ""
        print "创建目标:", id, " 时间:", time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.start_time)), "位置:", track_window

    def update(self, frame):
        self.frame_num += 1

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        back_project = cv2.calcBackProject([hsv], [0], self.roi_hist, [0, 180], 1)

        # meanShift
        ret, self.track_window = cv2.meanShift(back_project, self.track_window, self.term_crit)
        x, y, w, h = self.track_window
        latest_center = utils.cal_center([[x, y], [x + w, y], [x, y + h], [x + w, y + h]])
        # 记录silent time，如果达到某一阈值则视为跟踪结束
        if (self.frame_num > 1 and abs(self.center[0] - latest_center[0]) + abs(
                    self.center[1] - latest_center[1]) < 5):
            self.silent_time += 1
        else:
            self.silent_time = 0
        self.center = latest_center
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
        if self.frame_num % 5 == 0:
            print "meanShift算法数据 ID:", self.id, " 位置:", (x, y, w, h), " 中心点:", self.center

        self.kalman.correct(self.center)
        prediction = self.kalman.predict()
        prediction_width = int(prediction[0])
        prediction_height = int(prediction[1])
        # print "kalman预测 ID:", self.id, " ", prediction_width, prediction_height
        # kalman预测结果决定该目标是否跟踪结束
        if (self.frame_num > 10 and not utils.pnpoly(self.roi_points,
                                                     [prediction_width, prediction_height])) or self.silent_time >= 10:
            self.should_remove = True

    def remove_self(self):
        duration_second = int(time.time() - self.start_time)
        duration_minute = 0
        if duration_second >= 60:
            duration_minute = duration_second / 60
            duration_second -= 60 * duration_minute
        print "目标 ", self.id, " 离开跟踪区域，持续时间:", duration_minute, "m", duration_second, "s"
