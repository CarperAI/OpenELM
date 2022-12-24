from openelm.environments.sodaracer.walker.walk_creator import walker_creator


def query_cppn(wc, xgrid, ygrid, scale, connect_func, amp_func, phase_func):
    """Create a grid of points and functionally connect them."""
    joints = {}
    for x in range(xgrid):
        for y in range(ygrid):
            joints[(x, y)] = wc.add_joint(x * scale, y * scale)
    for x1 in range(xgrid):
        for y1 in range(ygrid):
            for x2 in range(x1, xgrid):
                for y2 in range(y1, ygrid):
                    if x1 == y1 and x2 == y2:
                        continue
                    if connect_func(x1, y1, x2, y2):
                        amp = amp_func(x1, y1, x2, y2)
                        phase = phase_func(x1, y1, x2, y2)
                        wc.add_muscle(joints[(x1, y1)], joints[(x2, y2)], amp, phase)
    return joints


def make_walker():
    wc = walker_creator()

    def connect(x1, y1, x2, y2):
        if ((x1 - x2) ** 2 + (y1 - y2) ** 2) > 4.5:
            return False
        return True

    def amp(x1, y1, x2, y2):
        return max(abs(x1 - x2), abs(y1 - y2))

    def phase(x1, y1, x2, y2):
        return x1 if x1 % 2 == 1 else -x1

    _ = query_cppn(wc, 8, 3, 1.5, connect, amp, phase)

    return wc.get_walker()
