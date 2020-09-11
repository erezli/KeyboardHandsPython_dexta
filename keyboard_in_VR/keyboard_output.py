import cv2
import numpy as np


def perspective_transform(frame, vertices):
    """

    :param frame:
    :param vertices: vertices needs to be in the structure of [[x1, y1], [x2, y2] ...]
    :return:
    """
    pts1 = np.float32(vertices)
    pts2 = np.float32([[0, 0], [880, 0], [0, 260], [880, 260]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    result = cv2.warpPerspective(frame, matrix, (880, 260))
    return result


def the_covered_key(frame, finger_positions):
    """

    :param frame: after perspective_transform
    :param finger_positions: this is in the cropped frame
    :return:
    """
    keyboard_layout = {(range(0, 440), range(0, 130)): 'A',
                       (range(0, 440), range(131, 260)): 'B',
                       (range(441, 880), range(0, 130)): 'C',
                       (range(441, 880), range(131, 260)): 'D'}

    return_list = []
    if finger_positions is not None:
        (x_f, y_f, w_f, h_f) = finger_positions
        for x, y, w, h in zip(x_f, y_f, w_f, h_f):
            for position_range in keyboard_layout.keys():
                if x+w/2 in position_range[0] and y+h/2 in position_range[1]:
                    print(keyboard_layout[position_range])
                    cv2.putText(frame, keyboard_layout[position_range],
                                (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    if keyboard_layout[position_range] not in return_list:
                        return_list.append(keyboard_layout[position_range])

    return return_list
