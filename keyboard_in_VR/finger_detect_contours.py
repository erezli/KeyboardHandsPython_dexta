import cv2


def detect_finger_by_contours(frame):
    """
    use after applying detect_finger_by_hsv.
    this function draws contours around all the object detected - fingers in this case.
    :param frame:
    :return: 4 lists of values for x, y, w, h of the finger detected
    """
    imgray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # blur = cv2.GaussianBlur(imgray, (5, 5), 0)  # blur
    blur = cv2.bilateralFilter(imgray, 9, 75, 75)
    # _, thresh = cv2.threshold(blur, 100, 255, cv2.THRESH_BINARY)
    trans = cv2.dilate(blur, None, iterations=2)
    contours, hierarchy = cv2.findContours(trans, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    x_list = []
    y_list = []
    w_list = []
    h_list = []

    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)

        # set restriction on contours area to filter false input
        if cv2.contourArea(contour) < 600:
            continue
        if cv2.contourArea(contour) > 900:
            continue
        x_list.append(x)
        y_list.append(y)
        w_list.append(w)
        h_list.append(h)
        cv2.drawContours(frame, contours, -1, (255, 255, 0), 3)
        cv2.circle(frame, (int(x+w/2), int(y+h/2)), 20, (255, 34, 34), -1)
    if len(contours) == 0:
        cv2.putText(frame, 'No Finger Detected - hold on', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 255), 1)
    cv2.imshow('feed2', frame)

    return x_list, y_list, w_list, h_list
