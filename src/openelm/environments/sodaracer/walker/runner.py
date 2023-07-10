import math

from openelm.environments.sodaracer.walker import query_cppn, walker_creator


def make_walker(p_scale=1):  # acrylic of current (m)
    wc = walker_creator()

    def connect(x1, y1, x2, y2):
        if -2 * x1 + x2 * 2 > 2:
            return True
        return x1 <= abs(y1 - y2)

    def amp(x, y, x2, y2):
        return abs(x - x2) + abs(y - y2)

    def phase(x1, y1, x2, y2):
        return -x1 / 2 - math.cos(math.pi / 9)

    joints = query_cppn(wc, 5, 7 + p_scale, 2, connect, amp, phase)
    return wc.get_walker()
