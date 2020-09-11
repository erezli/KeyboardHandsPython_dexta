import cv2
import numpy as np


def detect_finger_by_hsv(frame, l_hsv, u_hsv):
    """
    detect the color set by the arguments l_hsv and u_hsv.
    this function mask out the color outside the range and return the filtered image
    :param frame:
    :param l_hsv: list of lower HSV values
    :param u_hsv: list of upper HSV values
    :return: a masked image that only shows color in the set range
    """

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    l_b = np.array([l_hsv[0], l_hsv[1], l_hsv[2]])
    u_b = np.array([u_hsv[0], u_hsv[1], u_hsv[2]])

    mask = cv2.inRange(hsv, l_b, u_b)

    res = cv2.bitwise_and(frame, frame, mask=mask)

    # cv2.imshow("frame", frame)
    # cv2.imshow("mask", mask)
    # cv2.imshow("res", res)
    return res, mask
