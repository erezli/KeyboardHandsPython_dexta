import cv2
from keyboard_in_VR.KeyboardInVR_OOP.detected_object import ObjectFrame
import numpy as np


class Fingers(ObjectFrame):
    max_area = 900
    min_area = 600
    finger_num = 0

    def __init__(self, hsv_l, hsv_u):
        super().__init__(hsv_l, hsv_u)
        self._roi_hists = np.zeros((180, 1))
        self.track_window = []
        self.position = []  # (x, y)
        self.areas = []
        # self.finger_num = finger_num

    def update_property(self, frame):
        """
        using the usv boundary to filter the frame. return the new position
        :param frame:
        :return:
        """
        # mask out the background
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        l_b = np.array(self.hsv_l)
        u_b = np.array(self.hsv_u)
        mask = cv2.inRange(hsv, l_b, u_b)
        res = cv2.bitwise_and(frame, frame, mask=mask)
        # detect the object using contour
        imgray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
        blur = cv2.bilateralFilter(imgray, 9, 75, 75)
        trans = cv2.dilate(blur, None, iterations=2)
        contours, hierarchy = cv2.findContours(trans, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        x_list = []
        y_list = []
        w_list = []
        h_list = []
        track_win = []
        contour_list = []

        for contour in contours:
            (x, y, w, h) = cv2.boundingRect(contour)

            # set restriction on contours area to filter false input
            if cv2.contourArea(contour) < self.min_area:
                continue
            if cv2.contourArea(contour) > self.max_area:
                continue
            x_list.append(x)
            y_list.append(y)
            w_list.append(w)
            h_list.append(h)
            track_win.append((x, y, w, h))
            contour_list.append(contour)
            # res2 = frame.copy()
            # cv2.drawContours(res2, contours, -1, (255, 255, 0), 3)
            # cv2.circle(res2, (int(x + w / 2), int(y + h / 2)), 20, (255, 34, 34), -1)
        # if len(x_list) == 0:
            # cv2.putText(res2, 'No Finger Detected - hold on', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 255), 1)
        self.finger_num = len(x_list)
        self.track_window = track_win
        self.position = [(xx + ww / 2, yy + hh / 2) for xx, ww, yy, hh in zip(x_list, w_list, y_list, h_list)]
        self.areas = contour_list
        # return res2

    def update_roi_hists(self, frame):
        hists = []
        for i in range(self.finger_num):
            (x, y, w, h) = self.track_window[i]
            # set up the ROI for tracking
            roi = frame[y:y + h, x:x + w]
            hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv_roi, np.array(self.hsv_l), np.array(self.hsv_u))
            roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
            cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
            hists.append(roi_hist)
        self.roi_hists = hists

    def tracking_position(self, frame):
        # setup the termination criteria, either 10 iteration or move by at least 1 pt
        term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
        for i in range(self.finger_num):
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            dst = cv2.calcBackProject([hsv], [0], self.roi_hists[i], [0, 180], 1)
            ret, track_window = cv2.CamShift(dst, self.track_window[i], term_crit)  # ret has value x, y, w, h, rot
            pts = cv2.boxPoints(ret)
            pts = np.int0(pts)
            final_frame = cv2.polylines(frame, [pts], True, (255, 255, 9), 2)
            self.track_window[i] = track_window
        return final_frame

    def translucent_fingers(self, frame, first_frame, transparency=4):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        l_b = np.array(self.hsv_l)
        u_b = np.array(self.hsv_u)
        mask = cv2.inRange(hsv, l_b, u_b)
        res = cv2.bitwise_and(frame, frame, mask=mask)
        if transparency == 4:
            a = 0.7
            b = 0.3
        elif transparency == 3:
            a = 0.6
            b = 0.4
        elif transparency == 2:
            a = 0.5
            b = 0.5
        else:
            a = 0.4
            b = 0.6
        # fgmask = cv2.cvtColor(fgmask, cv2.COLOR_GRAY2BGR)
        hands = cv2.bitwise_and(frame, res)
        translucent_fingers = cv2.addWeighted(first_frame, a, hands, b, 0)
        return translucent_fingers

    @staticmethod
    def detect_hand_haar(frame):
        hand_detection = cv2.CascadeClassifier('../haarcascades/Hand.Cascade.1.xml')
        hand_detection_2 = cv2.CascadeClassifier('../haarcascades/hand.xml')
        fist_detection = cv2.CascadeClassifier('../haarcascades/fist.xml')
        palm_detection = cv2.CascadeClassifier('../haarcascades/fist.xml')

        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        box = []
        hand_rectangle1 = hand_detection.detectMultiScale(grey, 1.3, 5)
        for (x, y, w, h) in hand_rectangle1:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 10)
            box.append((x, y, w, h))
        hand_rectangle2 = hand_detection_2.detectMultiScale(grey, 1.3, 5)
        for (x, y, w, h) in hand_rectangle2:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 10)
            box.append((x, y, w, h))
        fist_rectangle = fist_detection.detectMultiScale(grey, 1.3, 5)
        for (x, y, w, h) in fist_rectangle:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 10)
            box.append((x, y, w, h))
        palm_rectangle = palm_detection.detectMultiScale(grey, 1.3, 5)
        for (x, y, w, h) in palm_rectangle:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 10)
            box.append((x, y, w, h))
        return box
