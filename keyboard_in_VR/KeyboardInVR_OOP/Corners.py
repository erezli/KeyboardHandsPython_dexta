from keyboard_in_VR.KeyboardInVR_OOP.detected_object import ObjectFrame
import cv2
import numpy as np


class Corners(ObjectFrame):
    min_area = 100
    max_area = 1000
    lk_params = dict(winSize=(10, 10), maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    def __init__(self, hsv_l, hsv_u):
        super().__init__(hsv_l, hsv_u)

        self.positions = []     # positions of four corners - vertices of keyboard
        self.cnt = []
        self.track_win = []
        self._roi_hists = [np.zeros((180, 1)) for i in range(4)]

    def update_property(self, frame):
        """
        Using HSV colour to detect orange corner and update its location (sorted)
        :param frame:
        :return:
        """
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

        track_win = []
        contour_list = []

        for contour in contours:
            if cv2.contourArea(contour) < self.min_area:
                continue
            if cv2.contourArea(contour) > self.max_area:
                continue
            (x, y, w, h) = cv2.boundingRect(contour)
            track_win.append((x, y, w, h))
            contour_list.append(contour)
        if len(track_win) == 4:
            track_win.sort()
            l_w = track_win[:2]
            r_w = track_win[-2:]
            l_w.sort(key=lambda element: element[1])
            r_w.sort(key=lambda element: element[1])
            # give 4 vertices - top left - top right - bottom left - bottom right
            true_vertices = [(l_w[0][0], l_w[0][1]), (r_w[0][0] + r_w[0][2], r_w[0][1]),
                             (l_w[1][0], l_w[1][1] + l_w[1][3]), (r_w[1][0] + r_w[1][2], r_w[1][1] + r_w[1][3])]
            self.positions = true_vertices
            self.cnt = contour_list
            self.track_win = [l_w[0], r_w[0], l_w[1], r_w[1]]

    def optical_flow_track(self, old_frame_gray, frame):
        """
        Use optical flow to track corners with given previous location (sorted).
        :param old_frame_gray:
        :param frame:
        :return:
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        new_wins = []
        for wins in self.track_win:
            old_point1 = np.array([[wins[0], wins[1]]], dtype=np.float32)
            old_point2 = np.array([[wins[0]+wins[2], wins[1]+wins[3]]], dtype=np.float32)
            new_1, status, error = cv2.calcOpticalFlowPyrLK(old_frame_gray, gray, old_point1,
                                                            None, **self.lk_params)
            new_4, status, error = cv2.calcOpticalFlowPyrLK(old_frame_gray, gray, old_point2,
                                                            None, **self.lk_params)
            new_wins.append((new_1.ravel()[0], new_1.ravel()[1],
                             new_4.ravel()[0]-new_1.ravel()[0], new_4.ravel()[1]-new_1.ravel()[1]))
        new_wins.sort()
        l_w = new_wins[:2]
        r_w = new_wins[-2:]
        l_w.sort(key=lambda element: element[1])
        r_w.sort(key=lambda element: element[1])
        # give 4 vertices - top left - top right - bottom left - bottom right
        true_vertices = [(l_w[0][0], l_w[0][1]), (r_w[0][0] + r_w[0][2], r_w[0][1]),
                         (l_w[1][0], l_w[1][1] + l_w[1][3]), (r_w[1][0] + r_w[1][2], r_w[1][1] + r_w[1][3])]
        self.positions = true_vertices
        self.track_win = new_wins
        pts = [self.positions[0], self.positions[1], self.positions[3], self.positions[2]]
        cv2.polylines(frame, np.int0([pts]),
                      True, (0, 255, 145), 2)
        return gray.copy()

    def update_roi_hists(self, frame):
        """
        This function updates the ROI's histogram - use before using camshift tracking.
        :param frame:
        :return:
        """
        hists = []
        for i in range(4):
            (x, y, w, h) = self.track_win[i]
            # set up the ROI for tracking
            roi = frame[y:y + h, x:x + w]
            hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv_roi, np.array(self.hsv_l), np.array(self.hsv_u))
            roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
            cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
            hists.append(roi_hist)
        self.roi_hists = hists

    def cam_shift_track(self, frame):
        """
        Camshift tracking of 4 corners.
        :param frame:
        :return:
        """
        term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
        for i in range(4):
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            dst = cv2.calcBackProject([hsv], [0], self.roi_hists[i], [0, 180], 1)
            ret, track_window = cv2.CamShift(dst, self.track_win[i], term_crit)  # ret has value x, y, w, h, rot
            pts = cv2.boxPoints(ret)
            pts = np.int0(pts)
            final_frame = cv2.polylines(frame, [pts], True, (255, 255, 9), 2)
            self.track_win[i] = track_window
        true_vertices = [(self.track_win[0][0], self.track_win[0][1]),
                         (self.track_win[1][0] + self.track_win[1][2], self.track_win[1][1]),
                         (self.track_win[2][0], self.track_win[2][1] + self.track_win[2][3]),
                         (self.track_win[3][0] + self.track_win[3][2], self.track_win[3][1] + self.track_win[3][3])]
        self.positions = true_vertices
        pts = [self.positions[0], self.positions[1], self.positions[3], self.positions[2]]
        cv2.polylines(frame, np.int0([pts]),
                      True, (0, 255, 145), 2)
        return final_frame
