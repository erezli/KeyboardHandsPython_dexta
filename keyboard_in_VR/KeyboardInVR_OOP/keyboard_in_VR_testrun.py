# in this test keyboard detection, perspective transform and translucent hands are applied.

import cv2
from keyboard_in_VR.KeyboardInVR_OOP.Keyboard import Keyboard
from keyboard_in_VR.KeyboardInVR_OOP.Finger import Fingers


cap = cv2.VideoCapture(0)
layout = {}
keyboard = Keyboard([70, 38, 47], [120, 255, 255], layout)
hands = Fingers((0, 0, 160), (206, 28, 255), (186, 106, 113), (255, 141, 139))

while cap.isOpened():
    _, frame = cap.read()
    keyboard.get_position_contour(frame)
    # cv2.imshow('frame', frame)
    if keyboard.track_window != [0, 0, 0, 0]:
        print("found")
        break

print(keyboard.approx)
keyboard.vertices = keyboard.approx
print(keyboard.approx)
_, first_frame = cap.read()
count = 0
while cap.isOpened():
    _, frame = cap.read()
    perspective = keyboard.perspective_transformation(frame)
    print(type(frame))
    print(type(perspective))
    count += 1
    # first_per = keyboard.perspective_transformation(first_frame)
    # res = cv2.addWeighted(first_per, 0.5, perspective, 0.5, 0)
    show = hands.TranslucentPile(perspective, count)
    cv2.imshow('perspective transformation', perspective)
    # cv2.imshow('first', first_per)
    # cv2.imshow('result', res)
    cv2.imshow('show', show)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()

# while cap.isOpened():
#     _, frame = cap.read()
#     # get position of vertices
#     # keyboard.get_position_4_corners(frame)
#     keyboard.vertices = keyboard.approx
#     ######
#     # add a track method to keyboard
#     ######
#     keyboard_frame = keyboard.perspective_transformation(frame)
#     w, h, c = keyboard_frame.shape
#     keyboard_layout = {(range(0, w/2), range(0, h/2)): 'A',
#                        (range(w/2, w), range(0, h/2)): 'B',
#                        (range(0, w/2), range(h/2, h)): 'C',
#                        (range(w/2, w), range(h/2, h)): 'D'}
#     keyboard.layout = keyboard_layout
