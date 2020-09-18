import cv2
import numpy as np
from keyboard_in_VR.KeyboardInVR_OOP.detected_object import ObjectFrame


class Keyboard(ObjectFrame):

    max_area = 15000
    min_area = 5000

    def __init__(self, hsv_l, hsv_u, layout):
        super().__init__(hsv_l, hsv_u)

        self.layout = layout
        self.track_window = [0, 0, 0, 0]
        self.approx = np.zeros((4, 2))
        self.vertices = [(0, 0), (0, 0), (0, 0), (0, 0)]
        self.output = []
        self.contour = None

    def get_position_contour(self, frame):
        """
        updates the position of keyboard using HSV colour filter and contour detection
        :param frame:
        :return:
        """
        # mask out the background
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        l_b = np.array(self.hsv_l)
        u_b = np.array(self.hsv_u)
        mask = cv2.inRange(hsv, l_b, u_b)
        res = cv2.bitwise_and(frame, frame, mask=mask)
        blur = cv2.bilateralFilter(res, 9, 50, 100)
        opening = cv2.morphologyEx(blur, cv2.MORPH_OPEN, None)
        dilation = cv2.dilate(opening, None, iterations=2)
        grey = cv2.cvtColor(dilation, cv2.COLOR_BGR2GRAY)

        contours, _ = cv2.findContours(grey, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        count = 0
        for contour in contours:

            approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
            if len(approx) == 4:
                if cv2.contourArea(contour) >= 15000:
                    count += 1
                    if count >= 2:
                        cv2.putText(frame, 'seems like two keyboards are being detected...', (10, 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                        pass
                    else:
                        x, y, w, h = cv2.boundingRect(approx)
                        if frame.shape[1] - 5 < x + w <= frame.shape[1] or 0 <= x < 5 or frame.shape[0] - 5 < y + h <= \
                                frame.shape[0] or 0 <= y < 5:
                            cv2.putText(frame, 'bring the keyboard to centre', (frame.shape[0], 20),
                                        cv2.FONT_HERSHEY_SIMPLEX, .3, (0, 0, 255), 1)
                            pass
                        else:
                            cv2.putText(frame, 'found keyboard', (10, 20),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                            # cv2.drawContours(frame, [approx], 0, (255, 76, 186), 5)
                            # print((x, y, w, h))
                            # print(approx)
                            self.track_window = (x, y, w, h)
                            self.approx = approx
                            self.contour = contour
            else:
                cv2.putText(frame, 'no keyboard detected, try harder', (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
        # cv2.imshow("frame", frame)

    def get_position_4_corners(self, frame):
        # see Corners class
        pass

    def sort_vertices(self):
        """
        Sort 4 vertices into the order top left, bottom left, bottom right, top right
        :return:
        """
        pt1 = self.vertices[0]
        pt2 = self.vertices[1]
        pt3 = self.vertices[2]
        pt4 = self.vertices[3]
        if pt1[0] < pt2[0] and pt1[0] < pt3[0] or pt1[0] < pt2[0] and pt1[0] < pt4[0] or pt1[0] < pt3[0] and pt1[0] < pt4[0]:
            if pt1[1] < pt2[1] and pt1[1] < pt3[1] or pt1[1] < pt4[1] and pt1[1] < pt3[1] or pt1[1] < pt2[1] and pt1[1] < pt4[1]:
                if pt2[0] < pt3[0] and pt2[0] < pt4[0]:
                    if pt3[1] < pt4[1]:
                        self.vertices = [pt1, pt2, pt4, pt3]
                    else:
                        self.vertices = [pt1, pt2, pt3, pt4]
                elif pt3[0] < pt4[0] and pt3[0] < pt2[0]:
                    if pt2[1] < pt4[1]:
                        self.vertices = [pt1, pt3, pt4, pt2]
                    else:
                        self.vertices = [pt1, pt3, pt2, pt4]
                else:
                    if pt2[1] < pt3[1]:
                        self.vertices = [pt1, pt4, pt3, pt2]
                    else:
                        self.vertices = [pt1, pt4, pt2, pt3]
            else:
                if pt2[0] < pt3[0] and pt2[0] < pt4[0]:
                    if pt3[1] < pt4[1]:
                        self.vertices = [pt2, pt1, pt4, pt3]
                    else:
                        self.vertices = [pt2, pt1, pt3, pt4]
                elif pt3[0] < pt4[0] and pt3[0] < pt2[0]:
                    if pt2[1] < pt4[1]:
                        self.vertices = [pt3, pt1, pt4, pt2]
                    else:
                        self.vertices = [pt3, pt1, pt2, pt4]
                else:
                    if pt2[1] < pt3[1]:
                        self.vertices = [pt4, pt1, pt3, pt2]
                    else:
                        self.vertices = [pt4, pt1, pt2, pt3]
        elif pt2[0] < pt1[0] and pt2[0] < pt3[0] or pt2[0] < pt1[0] and pt2[0] < pt4[0] or pt2[0] < pt3[0] and pt2[0] < pt4[0]:
            if pt2[1] < pt1[1] and pt2[1] < pt3[1] or pt2[1] < pt4[1] and pt2[1] < pt3[1] or pt2[1] < pt1[1] and pt2[1] < pt4[1]:
                if pt1[0] < pt3[0] and pt1[0] < pt4[0]:
                    if pt3[1] < pt4[1]:
                        self.vertices = [pt2, pt1, pt4, pt3]
                    else:
                        self.vertices = [pt2, pt1, pt3, pt4]
                elif pt3[0] < pt4[0] and pt3[0] < pt1[0]:
                    if pt1[1] < pt4[1]:
                        self.vertices = [pt2, pt3, pt4, pt1]
                    else:
                        self.vertices = [pt2, pt3, pt1, pt4]
                else:
                    if pt1[1] < pt3[1]:
                        self.vertices = [pt2, pt4, pt3, pt1]
                    else:
                        self.vertices = [pt2, pt4, pt1, pt3]
            else:
                if pt1[0] < pt3[0] and pt1[0] < pt4[0]:
                    if pt3[1] < pt4[1]:
                        self.vertices = [pt1, pt2, pt4, pt3]
                    else:
                        self.vertices = [pt1, pt2, pt3, pt4]
                elif pt3[0] < pt4[0] and pt3[0] < pt1[0]:
                    if pt1[1] < pt4[1]:
                        self.vertices = [pt3, pt2, pt4, pt1]
                    else:
                        self.vertices = [pt3, pt2, pt1, pt4]
                else:
                    if pt1[1] < pt3[1]:
                        self.vertices = [pt4, pt2, pt3, pt1]
                    else:
                        self.vertices = [pt4, pt2, pt1, pt3]
        elif pt3[0] < pt2[0] and pt3[0] < pt1[0] or pt3[0] < pt2[0] and pt3[0] < pt4[0] or pt3[0] < pt1[0] and pt3[0] < pt4[0]:
            if pt3[1] < pt2[1] and pt3[1] < pt1[1] or pt3[1] < pt4[1] and pt3[1] < pt1[1] or pt3[1] < pt2[1] and pt3[1] < pt4[1]:
                if pt2[0] < pt1[0] and pt2[0] < pt4[0]:
                    if pt1[1] < pt4[1]:
                        self.vertices = [pt3, pt2, pt4, pt1]
                    else:
                        self.vertices = [pt3, pt2, pt1, pt4]
                elif pt1[0] < pt4[0] and pt1[0] < pt2[0]:
                    if pt2[1] < pt4[1]:
                        self.vertices = [pt3, pt1, pt4, pt2]
                    else:
                        self.vertices = [pt3, pt1, pt2, pt2]
                else:
                    if pt2[1] < pt1[1]:
                        self.vertices = [pt3, pt4, pt1, pt2]
                    else:
                        self.vertices = [pt3, pt4, pt2, pt1]
            else:
                if pt2[0] < pt1[0] and pt2[0] < pt4[0]:
                    if pt1[1] < pt4[1]:
                        self.vertices = [pt2, pt3, pt4, pt1]
                    else:
                        self.vertices = [pt2, pt3, pt1, pt4]
                elif pt1[0] < pt4[0] and pt1[0] < pt2[0]:
                    if pt2[1] < pt4[1]:
                        self.vertices = [pt1, pt3, pt4, pt2]
                    else:
                        self.vertices = [pt1, pt3, pt2, pt4]
                else:
                    if pt2[1] < pt1[1]:
                        self.vertices = [pt4, pt3, pt1, pt2]
                    else:
                        self.vertices = [pt4, pt3, pt2, pt1]
        else:
            if pt4[1] < pt2[1] and pt4[1] < pt3[1] or pt4[1] < pt1[1] and pt4[1] < pt3[1] or pt4[1] < pt2[1] and pt4[1] < pt1[1]:
                if pt2[0] < pt3[0] and pt2[0] < pt1[0]:
                    if pt3[1] < pt1[1]:
                        self.vertices = [pt4, pt2, pt1, pt3]
                    else:
                        self.vertices = [pt4, pt2, pt3, pt1]
                elif pt3[0] < pt1[0] and pt3[0] < pt2[0]:
                    if pt2[1] < pt1[1]:
                        self.vertices = [pt4, pt3, pt1, pt2]
                    else:
                        self.vertices = [pt4, pt3, pt2, pt1]
                else:
                    if pt2[1] < pt3[1]:
                        self.vertices = [pt4, pt1, pt3, pt2]
                    else:
                        self.vertices = [pt4, pt1, pt2, pt3]
            else:
                if pt2[0] < pt3[0] and pt2[0] < pt1[0]:
                    if pt3[1] < pt1[1]:
                        self.vertices = [pt2, pt4, pt1, pt3]
                    else:
                        self.vertices = [pt2, pt4, pt3, pt1]
                elif pt3[0] < pt1[0] and pt3[0] < pt2[0]:
                    if pt2[1] < pt1[1]:
                        self.vertices = [pt3, pt4, pt1, pt2]
                    else:
                        self.vertices = [pt3, pt4, pt2, pt1]
                else:
                    if pt2[1] < pt3[1]:
                        self.vertices = [pt1, pt4, pt3, pt2]
                    else:
                        self.vertices = [pt1, pt4, pt2, pt3]

    def perspective_transformation(self, frame):
        """
        crop the frame to display detected keyboard only.
        :param frame:
        :return:
        """
        self.sort_vertices()
        pts1 = np.float32(self.vertices)
        pts2 = np.float32([[0, 0], [0, 260], [880, 260], [880, 0]])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        result = cv2.warpPerspective(frame, matrix, (880, 260))
        return result

    def keyboard_output(self, finger_positions):    # rewrite this
        return_list = []
        if finger_positions is not None:
            '''
            (x_f, y_f, w_f, h_f) = finger_positions     # this is the track window of the Finger object
            for x, y, w, h in zip(x_f, y_f, w_f, h_f):
                for position_range in self.layout.keys():
                    if x + w / 2 in position_range[0] and y + h / 2 in position_range[1]:
                        print(self.layout[position_range])
                        # cv2.putText(frame, self.layout[position_range], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0,
                        # 255), 3) 
                        if self.layout[position_range] not in return_list:
                            return_list.append(self.layout[position_range])
            '''
            # or use position rather than track window
            for (x, y) in finger_positions:
                for position_range in self.layout.keys():
                    if x in position_range[0] and y in position_range[1]:
                        print(self.layout[position_range])
                        # cv2.putText(frame, self.layout[position_range], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0,
                        # 255), 3)
                        if self.layout[position_range] not in return_list:
                            return_list.append(self.layout[position_range])

        self.output = return_list

