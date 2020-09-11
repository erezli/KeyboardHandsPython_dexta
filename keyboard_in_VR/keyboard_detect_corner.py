import cv2
import numpy as np
from keyboard_in_VR import finger_detection_color


def detect_keyboard_corner(img):
    filtered, mask = finger_detection_color.detect_finger_by_hsv(img, [70, 38, 47],
                                                                 [120, 160, 160])  # [80, 25, 11], [255, 255, 217]
    blur = cv2.bilateralFilter(filtered, 9, 50, 100)
    opening = cv2.morphologyEx(blur, cv2.MORPH_OPEN, None)
    dilation = cv2.dilate(opening, None, iterations=2)
    grey = cv2.cvtColor(dilation, cv2.COLOR_BGR2GRAY)
    cv2.imshow('grey', grey)

    corners = cv2.goodFeaturesToTrack(grey, 4, 0.9, 100)  # max number of corners(from strongest); quality level of
    # corner; min distance
    x_list = []
    y_list = []
    if corners is not None:
        corners = np.int0(corners)

        for i in corners:
            x, y, = i.ravel()
            x_list.append(x)
            y_list.append(y)
            cv2.circle(img, (x, y), 9, 255, -1)

    return img, x_list, y_list
