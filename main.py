#!/usr/bin/python
# -*- coding: utf-8 -*-
import cv2
import target
import utils

__name__ = "__main__"

font = cv2.FONT_HERSHEY_SIMPLEX
roi_points = []
targets = []


def remove_target(target_item):
    for i in range(0, len(targets)):
        if target_item.id == targets[i].id:
            del targets[i]
            break
    target_item.remove_self()
    return


def main():
    # camera = cv2.VideoCapture(0)
    camera = cv2.VideoCapture("movie.mpg")
    # I can't refer values through 'cv.XXX', so use Integer directly
    resolution = (camera.get(3), camera.get(4))

    print "目标窗口分辨率为 " + str(resolution)
    print "请输入ROI顶点坐标，输入0终止"
    print "输入格式sample："
    print "100,100\n200,200\n300,300\n0"
    print "请输入："

    # 接收并解析roi各顶点坐标
    # while True:
        # line = sys.stdin.readline().replace("\n", "")
        # if line == "0":
        #     break
        # point = line.split(",")
        # point[0] = int(point[0])
        # point[1] = int(point[1])
        # roi_points.append(point)

    roi_points.append([250, 200])
    roi_points.append([50, 200])
    roi_points.append([50, 50])
    roi_points.append([250, 50])

    history = 0

    # KNN background subtractor
    bs = cv2.createBackgroundSubtractorKNN(detectShadows=True)
    bs.setHistory(history)

    cv2.namedWindow("移动目标监测")
    frames = 0
    counter = 0

    while True:
        grabbed, frame = camera.read()
        if grabbed is False:
            print "无法获取下一帧图像 ", frames
            break

        fgmask = bs.apply(frame)

        # background subtractor 构建历史帧
        if frames < history:
            frames += 1
            continue

        th = cv2.threshold(fgmask.copy(), 127, 255, cv2.THRESH_BINARY)[1]
        th = cv2.erode(th, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=2)
        dilated = cv2.dilate(th, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 3)), iterations=2)
        image, contours, hier = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in contours:
            if cv2.contourArea(c) > 500:
                (x, y, w, h) = cv2.boundingRect(c)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
                cen = utils.cal_center([[x, y], [x + w, y], [x, y + h], [x + w, y + h]])
                is_new = True
                for t in targets:
                    # 如果target已存在于数组中，则跳过
                    if abs(cen[0] - t.center[0]) + abs(cen[1] - t.center[1]) < 150:
                        is_new = False
                        break

                # 如果target是新的且在roi内
                if is_new and utils.pnpoly(roi_points, cen):
                    targets.append(target.Target(counter, frame, (x, y, w, h), roi_points))
                    counter += 1

        frames += 1

        for t in targets:
            t.update(frame)

        # 更新数组
        i = 0
        while i < len(targets):
            if targets[i].should_remove:
                remove_target(targets[i])
            else:
                i += 1

        # 画roi的范围
        for i in range(len(roi_points)):
            j = (i + 1) % len(roi_points)
            cv2.line(frame, tuple(roi_points[i]), tuple(roi_points[j]), (255, 0, 0))

        cv2.imshow("移动目标监测", frame)
        if cv2.waitKey(110) & 0xff == 27:
            break

    camera.release()


if __name__ == "__main__":
    main()
