############
# use this file to find suitable HSV and YCrCb boundary.
############
import cv2
import hsv_ycrcb

fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()


def nothing(x):
    return None


def backgroundOnly(frame):
    fg = fgbg.apply(frame)
    msk = cv2.bitwise_and(frame, frame, mask=fg)
    sub = cv2.absdiff(frame, msk)
    return sub, msk


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
    _, frame1 = cap.read()
    '''
    subtract1, mask1 = backgroundOnly(frame1)
    # cv2.waitKey(10)
    _, frame2 = cap.read()
    subtract2, mask2 = backgroundOnly(frame2)
    # cv2.waitKey(10)
    _, frame3 = cap.read()
    subtract3, mask3 = backgroundOnly(frame3)
    # cv2.waitKey(10)
    _, frame4 = cap.read()
    subtract4, mask4 = backgroundOnly(frame4)
    # cv2.waitKey(10)
    _, frame5 = cap.read()
    subtract5, mask5 = backgroundOnly(frame5)
    _, frame6 = cap.read()
    subtract6, mask6 = backgroundOnly(frame6)
    _, frame7 = cap.read()
    subtract7, mask7 = backgroundOnly(frame7)

    cv2.imshow("Frame", frame1)

    #cv2.imshow("s1", subtract1)
    #cv2.imshow("s2", subtract2)
    #cv2.imshow("s3", subtract3)
    #cv2.imshow("s4", subtract4)
    #cv2.imshow("s5", subtract5)
    #cv2.imshow("s6", subtract6)
    #cv2.imshow("s7", subtract7)

    added = cv2.addWeighted(subtract1, .5, mask1, .5, 0)
    added = cv2.addWeighted(subtract2, .334, added, .666, 0)
    added = cv2.addWeighted(subtract3, .25, added, .75, 0)
    added = cv2.addWeighted(subtract4, .2, added, .8, 0)
    added = cv2.addWeighted(subtract5, .1667, added, .8223, 0)
    added = cv2.addWeighted(subtract6, .143, added, .857, 0)
    added = cv2.addWeighted(subtract7, .125, added, .875, 0)
    cv2.imshow("added", added)
    '''

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

    handOnly, hsv, ycrcb = hsv_ycrcb.SkinDetect(frame1, hsvB, ycrcbB)
    cv2.imshow("hsvOnly", hsv)
    cv2.imshow("ycrcbOnly", ycrcb)
    cv2.imshow("hands", handOnly)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
