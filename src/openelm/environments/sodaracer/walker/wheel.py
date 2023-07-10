import math

from openelm.environments.sodaracer.walker import walker_creator


def make_circle(wc, cx, cy, radius, num_points):
    joints = []
    tot_ang = 3.14 * 2.0
    for idx in range(num_points):
        ang = tot_ang / (num_points + 1) * idx
        x = math.cos(ang) * radius + 0.5
        y = math.sin(ang) * radius + cy
        joints.append(wc.add_joint(x, y))
    return joints


def make_walker():
    wc = walker_creator()
    num_points = 8
    rad = 3.0
    cx, cy = (11, 5)
    points = make_circle(wc, 0.6, -0.5, rad / 2, num_points)
    center = wc.add_joint(cx + 1, cy + 1)
    for j in range(num_points):
        for i in range(num_points - 5):
            # removed 0.0 (old 3rd param value) as it was used in deprecated isDistance
            wc.add_muscle(
                points[j], points[(i + j) % num_points], 1.0, (j + 1) / num_points
            )
        wc.add_muscle(points[j], center, 3, (j + 1) / num_points)
    return wc.get_walker()
