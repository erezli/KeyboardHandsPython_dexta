import cv2
import numpy as np
from keyboard_in_VR import finger_detection_color


def locate_keyboard(img):
    """
    this function is used to locate the keyboard using HSV color detection and contours shape detection
    :param img:
    :return: ROI - frame that only contain the keyboard
    """
    # apply hsv filter
    filtered, mask = finger_detection_color.detect_finger_by_hsv(img, [70, 38, 47], [120, 255, 255])    #[80, 25, 11], [255, 255, 217]
    blur = cv2.bilateralFilter(filtered, 9, 50, 100)
    opening = cv2.morphologyEx(blur, cv2.MORPH_OPEN, None)
    dilation = cv2.dilate(opening, None, iterations=2)
    grey = cv2.cvtColor(dilation, cv2.COLOR_BGR2GRAY)
    cv2.imshow('grey', grey)

    # detect keyboard
    # contours, _ = cv2.findContours(grey, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contours, _ = cv2.findContours(grey, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    x, y, w, h = 0, 0, 0, 0
    approx_zero = np.zeros((4, 2))

    # identify which contour is the one we need
    for contour in contours:
        count = 0
        approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
        # x = approx.ravel()[0]
        # y = approx.ravel()[1]
        if len(approx) == 4:
            if cv2.contourArea(contour) >= 15000:
                count += 1
                if count >= 2:
                    # print('seems like two keyboards are being detected...')
                    cv2.putText(img, 'seems like two keyboards are being detected...', (10, 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    return x, y, w, h, approx_zero, True
                else:
                    cv2.putText(img, 'found keyboard', (10, 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    x, y, w, h = cv2.boundingRect(approx)
                    if img.shape[1] - 5 < x+w <= img.shape[1] or 0 <= x < 5 or img.shape[0] - 5 < y+h <= img.shape[0] or 0 <= y < 5:
                        cv2.putText(img, 'bring the keyboard to centre', (img.shape[0], 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, .3, (0, 0, 255), 1)
                        return x, y, w, h, approx_zero, True
                    else:
                        cv2.drawContours(img, [approx], 0, (255, 76, 186), 5)
                        print((x, y, w, h))
                        print(approx)
                        return x, y, w, h, approx, False
            elif 50000 > cv2.contourArea(contour) > 1500:
                # print('maybe bring the camera closer')
                cv2.putText(img, 'maybe bring the camera closer', (10, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                return x, y, w, h, approx_zero, True
            else:
                # print('no keyboard detected, try harder')
                cv2.putText(img, 'no keyboard detected, try harder', (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
                return x, y, w, h, approx_zero, True
        else:
            continue
    return x, y, w, h, approx_zero, True


def crop_to_roi(x, y, w, h, approx, img):
    mask = np.zeros_like(img)
    channel_count = img.shape[2]
    # print(channel_count)
    match_mask_color = (255,) * channel_count   # should give (255, 255, 255)
    # print(match_mask_color)
    vertices = [(x, y), (x+w, y), (x+w, y+h), (x, y+h)]
    cv2.fillPoly(mask, [approx], match_mask_color)
    # cv2.fillPoly(mask, np.array([vertices]), match_mask_color)
    # cv2.imshow('mask', mask)
    masked_img = cv2.bitwise_and(img, mask)
    return masked_img
