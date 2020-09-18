import cv2
from keyboard_in_VR.KeyboardInVR_OOP.detected_object import ObjectFrame
import numpy as np


class Fingers(ObjectFrame):
    max_area = 900
    min_area = 600
    finger_num = 0

    def __init__(self, hsv_l, hsv_u, ycrcb_l, ycrcb_u):
        super().__init__(hsv_l, hsv_u)
        self.hsvBoundary = [hsv_l, hsv_u]
        self.YCrCbBoundary = [ycrcb_l, ycrcb_u]
        self._roi_hists = np.zeros((180, 1))
        self.track_window = []
        self.position = []  # (x, y)
        self.areas = []

        # self.finger_num = finger_num

    def update_property(self, frame):
        """
        using the usv boundary to filter the frame. return the new position
        This function is originally for fingers wearing blue gloves.
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
        """
        This function is originally for fingers wearing blue gloves.
        :param frame:
        :return:
        """
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
        """
        This function is originally for fingers wearing blue gloves.
        :param frame:
        :return:
        """
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
        """
        This function is beaten by SkinDetect and TranslucentPile. Because this function uses the first frame ONLY for
        background of translucent effect.
        This function is originally for fingers wearing blue gloves.
        :param frame:
        :param first_frame:
        :param transparency:
        :return:
        """
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
    def HandFiltering(frame):
        """
        Uses adaptive threshold to find hands on keyboard and contours to filter out 'small' noise.
        :param frame:
        :return:
        """
        frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(frameGray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 799, -20)
        mask = np.zeros(frame.shape, dtype=frame.dtype)
        contors, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        for c in contors:
            if cv2.contourArea(c) > 1000:
                cv2.drawContours(mask, [c], -1, (255, 255, 255), -1)
        # mask = cv2.bitwise_not(mask)
        mask = cv2.bitwise_or(frame, mask)

        return mask

    def SkinDetect(self, img):
        """
        Use HSV combined with YCrCb to attempt better filtering result to get hands.
        :param img:
        :return: masks to mask out hands
        """
        img = self.HandFiltering(img)
        # converting from gbr to hsv color space
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # skin color range for hsv color space
        hsv_mask = cv2.inRange(img_hsv, self.hsvBoundary[0], self.hsvBoundary[1])
        hsv_mask = cv2.morphologyEx(hsv_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

        # converting from gbr to YCbCr color space
        img_y_cr_cb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        # skin color range for hsv color space
        y_cr_cb_mask = cv2.inRange(img_y_cr_cb, self.YCrCbBoundary[0], self.YCrCbBoundary[1])
        y_cr_cb_mask = cv2.morphologyEx(y_cr_cb_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

        # merge skin detection (YCbCr and hsv)
        global_mask = cv2.bitwise_and(y_cr_cb_mask, hsv_mask)
        global_mask = cv2.medianBlur(global_mask, 3)
        global_mask = cv2.morphologyEx(global_mask, cv2.MORPH_OPEN, np.ones((4, 4), np.uint8))

        hsv_result = cv2.bitwise_not(hsv_mask)
        y_cr_cb_result = cv2.bitwise_not(y_cr_cb_mask)
        global_result = cv2.bitwise_not(global_mask)
        return global_result, hsv_result, y_cr_cb_result

    def TranslucentPile(self, frame, count):
        """
        Makes hands translucent on the frame
        :param frame:
        :param count:
        :return:
        """
        global added
        nohand, hsv, ycrcb = self.SkinDetect(frame)
        nohand = cv2.erode(nohand, np.ones((3, 3), np.uint8), iterations=3)
        nohand = cv2.bitwise_and(frame, frame, mask=nohand)
        if count == 1:
            added = nohand
        else:
            added = cv2.addWeighted(nohand, 1 / (count + 1), added, 1 - (1 / (count + 1)), 0)
        show = cv2.addWeighted(added, .4, frame, .6, 0)
        return show

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
