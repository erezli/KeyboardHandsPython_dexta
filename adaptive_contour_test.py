import cv2
from tests import adaptive_contour

cap = cv2.VideoCapture(0)

while cap.isOpened():
    _, frame = cap.read()
    res = adaptive_contour.HandFiltering(frame)
    cv2.imshow('res', res)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
