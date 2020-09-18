# 好像没必要将所有对象都归到一个parent class里面， 但写都写了
class ObjectFrame:
    # keyboard and fingertips

    def __init__(self, hsv_l, hsv_u):
        """
        For detected and on-track objects. Stores their position values and hsv boundary for filtering.
        :param hsv_l: [h, s, v]
        :param hsv_u: [h, s, v]
        """
        # self.position = position
        self.hsv_l = hsv_l
        self.hsv_u = hsv_u

    # @property
    # def position(self):
    #     return self.position

    @property
    def hsv_b(self):
        return self.hsv_l, self.hsv_u

    # @position.setter
    # def position(self, new_position):
    #     self.position = new_position

    @hsv_b.setter
    def hsv_b(self, new_hsv):
        self.hsv_l = new_hsv[0]
        self.hsv_u = new_hsv[1]


