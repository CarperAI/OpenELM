from walk_creator import walker_creator
import math

def make_radial(wc, c, radius=1, spokes=7):
    """ Make a radial walker"""
    for spoke in range(spokes):
        x = radius * math.cos(2*math.pi/spokes * spoke)
        y = radius * math.sin(2*math.pi/spokes * spoke)
        wc.add_joint(x, y)

def make_walker():
    wc = walker_creator()

    # the main body is a radial
    center = wc.add_joint(5, 5)
    wheel = make_radial(wc, center, radius=1, spokes=4)

    return wc.get_walker()