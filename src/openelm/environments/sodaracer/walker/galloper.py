from openelm.environments.sodaracer.walker import walker_creator


def make_sensor(wc, x0, y0, x1, y1, d):
    return (
        wc.add_joint(x0, y0),
        wc.add_joint(x1, y1),
        wc.add_joint(x1, y0),
        wc.add_joint(x0, y1),
        wc.add_joint(d, 0.5),
        wc.add_joint(x1, 0.5),
    )


def make_walker(
    dx=0.0,
    dy=0.0,
    ddr=0,
    ddc=1.6,
):
    wc = walker_creator()
    ends = [
        make_sensor(wc, 5 + dx, -1 + dy, ddr, ddc, 4.5),
        make_sensor(wc, 0, -0.1, 8.0, 9.5, 0.03),  # replace "sid" with 8.0 (from paper)
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

    wc.add_muscle(center, sides[7], 2.0, 90.0 / 9.0)
    return wc.get_walker()
