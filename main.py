import cv2
import numpy as np

__name__ = "__main__"

font = cv2.FONT_HERSHEY_SIMPLEX


def center(points):
    x = (points[0][0] + points[1][0] + points[2][0] + points[3][0]) / 4
    y = (points[0][1] + points[1][1] + points[2][1] + points[3][1]) / 4
    return np.array([np.float32(x), np.float32(y)], np.float32)


class Target:
    def __init__(self, id, frame, track_window):
        # set up the roi
        self.id = int(id)
        x, y, w, h = track_window
        self.track_window = track_window
        self.roi = cv2.cvtColor(frame[y:y + h, x:x + w], cv2.COLOR_BGR2HSV)
        roi_hist = cv2.calcHist([self.roi], [0], None, [16], [0, 180])
        self.roi_hist = cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

        # set up the kalman
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        self.kalman.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                                               np.float32) * 0.03
        self.measurement = np.array((2, 1), np.float32)
        self.prediction = np.zeros((2, 1), np.float32)
        self.term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
        self.center = None
        self.update(frame)

    def __del__(self):
        print "Target %d destroyed" % self.id

    def update(self, frame, show_frame=None):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        back_project = cv2.calcBackProject([hsv], [0], self.roi_hist, [0, 180], 1)

        ret, self.track_window = cv2.meanShift(back_project, self.track_window, self.term_crit)
        x, y, w, h = self.track_window
        self.center = center([[x, y], [x + w, y], [x, y + h], [x + w, y + h]])
        # cv2.rectangle(frame, (x,y), (x+w, y+h), (255, 255, 0), 2)

        self.kalman.correct(self.center)
        # prediction = self.kalman.predict()
        # cv2.circle(frame, (int(prediction[0]), int(prediction[1])), 4, (255, 0, 0), -1)
        # fake shadow
        if show_frame is not None:
            cv2.putText(show_frame, "ID: %d -> %s" % (self.id, self.center), (11, (self.id + 1) * 25 + 1),
                        font, 0.6,
                        (0, 0, 0),
                        1,
                        cv2.LINE_AA)
            # actual info
            cv2.putText(show_frame, "ID: %d -> %s" % (self.id, self.center), (10, (self.id + 1) * 25),
                        font, 0.6,
                        (0, 255, 0),
                        1,
                        cv2.LINE_AA)


def main():
    # camera = cv2.VideoCapture("path"))
    camera = cv2.VideoCapture(0)

    history = 20

    # KNN background subtractor
    bs = cv2.createBackgroundSubtractorKNN()

    cv2.namedWindow("motion tracker")
    targets = {}
    first_frame = True
    frames = 0
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))

    while True:
        grabbed, frame = camera.read()
        if grabbed is False:
            print "failed to grab frame."
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

        counter = 0
        for c in contours:
            if cv2.contourArea(c) > 500:
                (x, y, w, h) = cv2.boundingRect(c)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
                # only create targets in the first frame, then just follow the ones you have
                if first_frame is True:
                    targets[counter] = Target(counter, frame, (x, y, w, h))
                counter += 1

        first_frame = False
        frames += 1

        # show_frame = frame
        # mirror
        show_frame = np.fliplr(frame).copy()

        for i, p in targets.iteritems():
            p.update(frame, show_frame)

        cv2.imshow("motion tracker", show_frame)
        # out.write(frame)
        if cv2.waitKey(110) & 0xff == 27:
            break
    # out.release()
    camera.release()


if __name__ == "__main__":
    main()
