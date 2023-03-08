from openelm.environments.sodaracer.simulator import IESoRWorld, SodaraceSimulator
from openelm.environments.sodaracer.walker import Walker

CIRCLE = """
def make_circle(wc, cx, cy, radius, num_points):
    \"\"\"Approximate a circle with center (cx,cy) square with num_points points.\"\"\"
    joints = []

    tot_ang = 3.14 * 2.0

    for idx in range(num_points):
        ang = (tot_ang / num_points) * idx
        x = math.cos(ang) * radius + cx
        y = math.sin(ang) * radius + cy
        joints.append(wc.add_joint(x, y))

    return joints

"""

RADIAL = """
def make_walker():
    \"\"\"Create a radial walker.\"\"\"
    wc = walker_creator()

    num_points = 8
    rad = 5.0
    cx, cy = (5, 5)
    # the main body is a square
    points = make_circle(wc, cx, cy, rad, num_points)
    center = wc.add_joint(cx, cy)

    for k in range(num_points):
        wc.add_muscle(points[k], points[(k + 1) % num_points])
        wc.add_muscle(points[k], center, float(k) / num_points, float(k) / num_points)

    return wc.get_walker()

"""

WHEEL = """
def make_walker():
    \"\"\"Create a wheel walker.\"\"\"
    wc = walker_creator()
    num_points = 8
    rad = 3.0
    cx, cy = (11, 5)
    points = make_circle(wc, 0.6, -0.5, rad / 2, num_points)
    center = wc.add_joint(cx + 1, cy + 1)
    for j in range(num_points):
        for i in range(num_points - 5):
            wc.add_muscle(points[j], points[(i + j) % num_points],
                          0.0, 1.0, (j + 1) / num_points)
        wc.add_muscle(points[j], center, 3, (j + 1) / num_points)
    return wc.get_walker()

"""

SQUARE_PREREQ = """
def make_square(wc, x0, y0, x1, y1):
    \"\"\"Make a square with top left x0,y0 and top right x1,y1.\"\"\"
    j0 = wc.add_joint(x0, y0)
    j1 = wc.add_joint(x0, y1)
    j2 = wc.add_joint(x1, y1)
    j3 = wc.add_joint(x1, y0)
    return j0, j1, j2, j3

"""

SQUARE = """
def make_walker():
    \"\"\"Create a square walker.\"\"\"
    wc = walker_creator()

    # the main body is a square
    sides = make_square(wc, 0, 0, 10, 10)
    center = wc.add_joint(5, 5)

    # connect the square with distance muscles
    for k in range(len(sides) - 1):
        wc.add_muscle(sides[k], sides[k + 1])
    wc.add_muscle(sides[3], sides[0])

    # one prong of the square is a distance muscle
    wc.add_muscle(sides[3], center)

    # the other prongs from the center of the square are active
    wc.add_muscle(sides[0], center, 5.0, 0.0)
    wc.add_muscle(sides[1], center, 10.0, 0.0)
    wc.add_muscle(sides[2], center, 2.0, 0.0)

    return wc.get_walker()

"""

GALLOPER_PREREQ = """
def make_sensor(wc, x0, y0, x1, y1, d):
    return (
        wc.add_joint(x0, y0),
        wc.add_joint(x1, y1),
        wc.add_joint(x1, y0),
        wc.add_joint(x0, y1),
        wc.add_joint(d, 0.5),
        wc.add_joint(x1, 0.5),
    )

"""

GALLOPER = """
def make_walker(
    dx=0.0,
    dy=0.0,
    ddr=0,
    ddc=1.6,
):
    wc = walker_creator()
    ends = [
        make_sensor(wc, 5 + dx, -1 + dy, ddr, ddc, 4.5),
        make_sensor(wc, 0, -0.1, sid, 9.5, 0.03),
        make_sensor(wc, 5.5, -0.001, 5.0, 4.86 + 0.8, 0.07),
        make_sensor(wc, 5.5, -3.0, 6.0, 4.86 + 0.8, 0.07),
        make_sensor(wc, 0, dx, ddr, ddc, 1.0),
    ]

    sides = ends[0] + ends[1] + ends[2] + ends[-1] + ends[-2] + ends[-3]

    center = wc.add_joint(dx, dy)

    # connect the square with distance muscles
    for k in range(len(sides) - 6):
        wc.add_muscle(sides[k], sides[k + 1], 30, 0.5)
    wc.add_muscle(sides[2], sides[4], 4.0, 0.8)
    for k in range(len(sides) - 2):
        wc.add_muscle(sides[k], sides[k + 2], 18.0, 60.0 / 5.5)

    for k in reversed(range(len(sides) - 6)):
        wc.add_muscle(sides[k], sides[k + 5], 4.0, 20.0 / 9.0)

    wc.add_muscle(center, sides[7], 2, 0, 90.0 / 9.0)
    return wc.get_walker()

"""

QUERY_CPPN = """
def query_cppn(wc, xgrid, ygrid, scale, connect_func, amp_func, phase_func):
    \"\"\"Create a grid of points and functionally connect them.\"\"\"
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

"""

CPPN_FIXED = """
def make_walker():
    wc = walker_creator()

    def connect(x1, y1, x2, y2):
        if ((x1 - x2) ** 2 + (y1 - y2) ** 2) > 4.5:
            return False
        return True

    def amp(x1, y1, x2, y2):
        return max(abs(x1 - x2), abs(y1 - y2))

    def phase(x1, y1, x2, y2):
        return np.sign(x1)

    _ = query_cppn(wc, 8, 3, 1.5, connect, amp, phase)

    return wc.get_walker()

"""

CPPN_MUTABLE = """
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

"""

RUNNER = """
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

"""

IMPORTS = """
from openelm.environments.sodaracer.walker import walker_creator
import math

"""

SEEDS_DICT = {
    "wheel": WHEEL,
    "radial": RADIAL,
    "square": SQUARE,
    "cppn_fixed": CPPN_FIXED,
    "cppn_mutable": CPPN_MUTABLE,
    "galloper": GALLOPER,
    "runner": RUNNER,
}

INSTRUCTIONS = {
    0: "",
    1: "def make_walker():\n",
    2: "#Create a new walker by modifying the starting function above.\ndef make_walker():\n",
    3: "#Combine the ,starting programs above to make a new program.\ndef make_walker():\n",
}

__all__ = [
    "IESoRWorld",
    "SodaraceSimulator",
    "Walker",
    "IMPORTS",
    "CIRCLE",
    "RADIAL",
    "WHEEL",
    "SQUARE_PREREQ",
    "SQUARE",
    "GALLOPER_PREREQ",
    "GALLOPER",
    "QUERY_CPPN",
    "CPPN_FIXED",
    "CPPN_MUTABLE",
    "RUNNER",
    "SEEDS_DICT",
]
