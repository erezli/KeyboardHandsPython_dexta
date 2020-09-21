import numpy as np
import cv2
import adaptive_contour


def SkinDetect(img, hsvBoundary, YCrCbBoundary):
    """
    using HSV and YCrCb after applying adaptive threshold to detect hands/skin
    - the colour boundary may differ between people.
    :param img:
    :param hsvBoundary: list: [(lower boundary tuple), (upper boundary tuple)]
    :param YCrCbBoundary: list: [(lower boundary tuple), (upper boundary tuple)]
    :return: masks
    """
    imgF, imgM = adaptive_contour.HandFiltering(img)
    imgM = cv2.cvtColor(imgM, cv2.COLOR_BGR2GRAY)
    # converting from gbr to hsv color space
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # skin color range for hsv color space
    hsv_mask = cv2.inRange(img_hsv, hsvBoundary[0], hsvBoundary[1])
    hsv_mask = cv2.morphologyEx(hsv_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    # converting from gbr to YCbCr color space
    img_y_cr_cb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    # skin color range for hsv color space
    y_cr_cb_mask = cv2.inRange(img_y_cr_cb, YCrCbBoundary[0], YCrCbBoundary[1])
    y_cr_cb_mask = cv2.morphologyEx(y_cr_cb_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    # merge skin detection (YCbCr and hsv)
    global_mask = cv2.bitwise_and(y_cr_cb_mask, hsv_mask)
    global_mask = cv2.bitwise_or(global_mask, imgM)
    global_mask = cv2.medianBlur(global_mask, 3)
    global_mask = cv2.morphologyEx(global_mask, cv2.MORPH_OPEN, np.ones((4, 4), np.uint8))

    hsv_result = cv2.bitwise_not(hsv_mask)
    y_cr_cb_result = cv2.bitwise_not(y_cr_cb_mask)
    global_result = cv2.bitwise_not(global_mask)
    return global_result, hsv_result, y_cr_cb_result


def nothing(x):
    return None


if __name__ == '__main__':
    cv2.namedWindow("TrackBarHSV")
    cv2.createTrackbar("LH", "TrackBarHSV", 0, 255, nothing)
    cv2.createTrackbar("LS", "TrackBarHSV", 0, 255, nothing)
    cv2.createTrackbar("LV", "TrackBarHSV", 0, 255, nothing)
    cv2.createTrackbar("UH", "TrackBarHSV", 255, 255, nothing)
    cv2.createTrackbar("US", "TrackBarHSV", 255, 255, nothing)
    cv2.createTrackbar("UV", "TrackBarHSV", 255, 255, nothing)

    cv2.namedWindow("TrackBarYCrCb")
    cv2.createTrackbar("LY", "TrackBarYCrCb", 0, 255, nothing)
    cv2.createTrackbar("Lr", "TrackBarYCrCb", 0, 255, nothing)
    cv2.createTrackbar("Lb", "TrackBarYCrCb", 0, 255, nothing)
    cv2.createTrackbar("UY", "TrackBarYCrCb", 255, 255, nothing)
    cv2.createTrackbar("Ur", "TrackBarYCrCb", 255, 255, nothing)
    cv2.createTrackbar("Ub", "TrackBarYCrCb", 255, 255, nothing)

    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        _, frame = cap.read()
        hl = cv2.getTrackbarPos("LH", "TrackBarHSV")
        sl = cv2.getTrackbarPos("LS", "TrackBarHSV")
        vl = cv2.getTrackbarPos("LV", "TrackBarHSV")
        hu = cv2.getTrackbarPos("UH", "TrackBarHSV")
        su = cv2.getTrackbarPos("US", "TrackBarHSV")
        vu = cv2.getTrackbarPos("UV", "TrackBarHSV")
        hsvB = [(hl, sl, vl), (hu, su, vu)]
        yl = cv2.getTrackbarPos("LY", "TrackBarYCrCb")
        rl = cv2.getTrackbarPos("Lr", "TrackBarYCrCb")
        bl = cv2.getTrackbarPos("Lb", "TrackBarYCrCb")
        yu = cv2.getTrackbarPos("UY", "TrackBarYCrCb")
        ru = cv2.getTrackbarPos("Ur", "TrackBarYCrCb")
        bu = cv2.getTrackbarPos("Ub", "TrackBarYCrCb")
        ycrcbB = [(yl, rl, bl), (yu, ru, bu)]

        handOnly, hsv, ycrcb = SkinDetect(frame, hsvB, ycrcbB)
        cv2.imshow("hsvOnly", hsv)
        cv2.imshow("ycrcbOnly", ycrcb)
        cv2.imshow("hands", handOnly)

        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
