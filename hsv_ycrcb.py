import numpy as np
import cv2


def SkinDetect(img, hsvBoundary, YCrCbBoundary):
    # converting from gbr to hsv color space
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # skin color range for hsv color space
    hsv_mask = cv2.inRange(img_hsv, hsvBoundary[0], hsvBoundary[1])
    hsv_mask = cv2.morphologyEx(hsv_mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    # converting from gbr to YCbCr color space
    img_y_cr_cb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    # skin color range for hsv color space
    y_cr_cb_mask = cv2.inRange(img_y_cr_cb, YCrCbBoundary[0], YCrCbBoundary[1])
    y_cr_cb_mask = cv2.morphologyEx(y_cr_cb_mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    # merge skin detection (YCbCr and hsv)
    global_mask = cv2.bitwise_and(y_cr_cb_mask, hsv_mask)
    global_mask = cv2.medianBlur(global_mask, 3)
    global_mask = cv2.morphologyEx(global_mask, cv2.MORPH_CLOSE, np.ones((4, 4), np.uint8))

    hsv_result = cv2.bitwise_not(hsv_mask)
    y_cr_cb_result = cv2.bitwise_not(y_cr_cb_mask)
    global_result = cv2.bitwise_not(global_mask)
    return global_result, hsv_result, y_cr_cb_result
