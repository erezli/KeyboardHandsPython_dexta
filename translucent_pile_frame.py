import cv2
import hsv_ycrcb
import numpy as np

fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
hsvB = [(0, 0, 160), (206, 28, 255)]
ycrcbB = [(186, 106, 113), (255, 141, 139)]


def movementFilter(frame):
    fg = fgbg.apply(frame)
    msk = cv2.bitwise_and(frame, frame, mask=fg)
    sub = cv2.absdiff(frame, msk)
    return sub, msk


def colourFilter(frame):
    nohand, hsv, ycrcb = hsv_ycrcb.SkinDetect(frame, hsvB, ycrcbB)
    nohand = cv2.erode(nohand, np.ones((2, 2), np.uint8), iterations=3)
    nohand = cv2.bitwise_and(frame, frame, mask=nohand)

    cv2.imshow("hsvOnly", hsv)
    cv2.imshow("ycrcbOnly", ycrcb)
    cv2.imshow("hands", nohand)
    return nohand, hsv, ycrcb


cap = cv2.VideoCapture(0)
count = 0

while cap.isOpened():
    global added
    count += 1
    _, frame = cap.read()
    bg, hsvbg, ycrcbbg = colourFilter(frame)
    if count == 1:
        added = bg
    else:
        added = cv2.addWeighted(bg, 1/(count+1), added, 1-(1/(count+1)), 0)

    show = cv2.addWeighted(added, .4, frame, .6, 0)
    cv2.imshow("Frame", show)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
