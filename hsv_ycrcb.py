import numpy as np
import cv2


def SkinDetect(img, hsvBoundary, YCrCbBoundary):
    # converting from gbr to hsv color space
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # skin color range for hsv color space
    hsv_mask = cv2.inRange(img_hsv, hsvBoundary[0], hsvBoundary[1])
    hsv_mask = cv2.morphologyEx(hsv_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    # converting from gbr to YCbCr color space
    img_YCrCb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    # skin color range for hsv color space
    YCrCb_mask = cv2.inRange(img_YCrCb, YCrCbBoundary[0], YCrCbBoundary[1])
    YCrCb_mask = cv2.morphologyEx(YCrCb_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    # merge skin detection (YCbCr and hsv)
    global_mask = cv2.bitwise_and(YCrCb_mask, hsv_mask)
    global_mask = cv2.medianBlur(global_mask, 3)
    global_mask = cv2.morphologyEx(global_mask, cv2.MORPH_OPEN, np.ones((4, 4), np.uint8))

    HSV_result = cv2.bitwise_not(hsv_mask)
    YCrCb_result = cv2.bitwise_not(YCrCb_mask)
    global_result = cv2.bitwise_not(global_mask)
    return global_result, HSV_result, YCrCb_result
