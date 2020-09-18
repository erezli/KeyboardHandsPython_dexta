import cv2
import numpy as np


def HandFiltering(frame):
    # 自适应阈值
    frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(frameGray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 799, -20)
    # 只保留一定面积以上的mask
    mask = np.zeros(frame.shape, dtype=frame.dtype)
    contors, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    for c in contors:
        if cv2.contourArea(c) > 1000:
            cv2.drawContours(mask, [c], -1, (255, 255, 255), -1)
    # mask = cv2.bitwise_not(mask)
    mask = cv2.bitwise_or(frame, mask)

    return mask


if __name__ == '__init__':
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        _, frame = cap.read()
        res = HandFiltering(frame)
        cv2.imshow('res', res)

        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
