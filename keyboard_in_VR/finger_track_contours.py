import cv2
import numpy as np


def finger_track_by_contours(frame1, frame2):
    """
    track any movement saw by the camera, using the difference between a frame and the frame after.
    In a stable environment where finger is the only thing that is moving.
    :param frame1:
    :param frame2:
    :return: 4 lists of x, y, w, h values for all the objects detected
    """

    diff = cv2.absdiff(frame1, frame2)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)   # convert to gray
    blur = cv2.GaussianBlur(gray, (5, 5), 0)    # blur
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)    # apply threshold
    # thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 9, 1)
    erosion = cv2.erode(thresh, None, iterations=1)
    dilation = cv2.dilate(erosion, None, iterations=1)
    contours, _ = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    x_list = []
    y_list = []
    w_list = []
    h_list = []

    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        x_list.append(x)
        y_list.append(y)
        w_list.append(w)
        h_list.append(h)
        if cv2.contourArea(contour) < 1000:
            continue
        cv2.rectangle(frame1, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame1, "Status: {}".format('Movement'), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    # cv2.drawContours(frame1, contours, -1, (0, 255, 0), 2)
    cv2.imshow('feed', frame1)

    '''
    frame1 = frame2
    ret, frame2 = cap.read()
    '''
    return x_list, y_list, w_list, h_list
