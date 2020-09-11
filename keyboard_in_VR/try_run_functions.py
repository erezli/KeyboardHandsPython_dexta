import cv2
from keyboard_in_VR import finger_detection_color
from keyboard_in_VR import finger_track_contours
from keyboard_in_VR import finger_detect_contours


cap = cv2.VideoCapture(0)

while cap.isOpened():

    # motion track
    ret, frame1 = cap.read()
    ret, frame2 = cap.read()
    res1, mask1 = finger_detection_color.detect_finger_by_hsv(frame1, [70, 100, 200], [180, 255, 255])
    res2, mask2 = finger_detection_color.detect_finger_by_hsv(frame2, [70, 100, 200], [180, 255, 255])
    # -- used to be [70, 26, 220], [255, 255, 255]

    ret, frame = cap.read()
    # res = finger_detection_color.detect_finger_by_hsv(frame, [70, 26, 200], [255, 255, 255])
    # -- this is for white gloves
    res, mask = finger_detection_color.detect_finger_by_hsv(frame, [88, 130, 200], [255, 255, 255])

    x, y, w, h = finger_track_contours.finger_track_by_contours(res1, res2)
    x2, y2, w2, h2 = finger_detect_contours.detect_finger_by_contours(res)

    print(x2, y2, w2, h2)
    # print(len(x))

    if cv2.waitKey(9) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
