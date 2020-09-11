import numpy as np
import cv2


def calc_roi_hist(frame, track_window, l_b, u_b):
    (x, y, w, h) = track_window
    # set up the ROI for tracking
    roi = frame[y:y + h, x:x + w]
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_roi, np.array((l_b[0], l_b[1], l_b[2])), np.array((u_b[0], u_b[1], u_b[2])))
    roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
    cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
    return roi_hist


def cam_shift(frame, track_window, roi_hist):
    """
    tracks a detected object
    :param roi_hist:
    :param track_window: initial position values of detected object (x, y, w, h)
    :param frame:
    :return:
    """
    # setup the termination criteria, either 10 iteration or move by at least 1 pt
    term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
    # cv2.imshow('roi', roi)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
    ret, track_window = cv2.CamShift(dst, track_window, term_crit)  # ret has value x, y, w, h, rot
    pts = cv2.boxPoints(ret)
    pts = np.int0(pts)
    final_frame = cv2.polylines(frame, [pts], True, (255, 255, 9), 2)
    # ret, track_window = cv2.meanShift(dst, track_window, term_crit)
    # x, y, w, h = track_window
    # final_frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)
    cv2.imshow('dst', dst)
    # cv2.imshow('final', final_img)
    # cv2.imshow('frame', frame)

    return final_frame, track_window
