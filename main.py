#!/usr/bin/python
# -*- coding: utf-8 -*-
import cv2
import target
import utils

__name__ = "__main__"

font = cv2.FONT_HERSHEY_SIMPLEX
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

    history = 0

    # KNN background subtractor
    bs = cv2.createBackgroundSubtractorKNN(detectShadows=True)
    bs.setHistory(history)

    cv2.namedWindow("motion tracker")
    frames = 0
    counter = 0

    while True:
        grabbed, frame = camera.read()
        if grabbed is False:
            print "failed to grab frame. " + str(frames)
            break

        fgmask = bs.apply(frame)

        # this is just to let the background subtractor build a bit of history
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
                # only create targets in the first frame, then just follow the ones you have
                cen = utils.cal_center([[x, y], [x + w, y], [x, y + h], [x + w, y + h]])
                is_new = True
                for t in targets:
                    # skip if already exist
                    if abs(cen[0] - t.center[0]) + abs(cen[1] - t.center[1]) < 150:
                        is_new = False
                        break

                if is_new:
                    print "创建新实例 ID:", counter, " pos:", (x, y, w, h), " center:", cen
                    targets.append(target.Target(counter, frame, (x, y, w, h), resolution))
                    counter += 1

        frames += 1

        for t in targets:
            t.update(frame)

        # refresh array
        i = 0
        while i < len(targets):
            if targets[i].should_remove:
                remove_target(targets[i])
            else:
                i += 1

        cv2.imshow("motion tracker", frame)
        if cv2.waitKey(110) & 0xff == 27:
            break

    camera.release()


if __name__ == "__main__":
    main()
