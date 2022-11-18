from elm.environments.sodaracer.helpers import (
    BodyInformation,
    Bone,
    DistanceAccessor,
    Entity,
    EntityType,
    Muscle,
    PhysicsId,
    staticBodyCount,
)
from elm.environments.sodaracer.simulator import IESoRWorld, SodaraceSimulator
from elm.environments.sodaracer.walker import Walker

SQUARE_SEED = {
    "program_str": """from elm.environments.sodaracer.walker import walker_creator


def make_square(wc, x0, y0, x1, y1):
    \"\"\"Make a square with top left x0,y0 and top right x1,y1.\"\"\"
    j0 = wc.add_joint(x0, y0)
    j1 = wc.add_joint(x0, y1)
    j2 = wc.add_joint(x1, y1)
    j3 = wc.add_joint(x1, y0)
    return j0, j1, j2, j3


def make_walker():
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
    wc.add_muscle(sides[0], center, False, 5.0, 0.0)
    wc.add_muscle(sides[1], center, False, 10.0, 0.0)
    wc.add_muscle(sides[2], center, False, 2.0, 0.0)

    return wc.get_walker()
""",
    "result_dict": {
        "joints": [(0, 0), (0, 10), (10, 10), (10, 0), (5, 5)],
        "muscles": [
            [0, 1, {"type": "distance", "amplitude": 0.0, "phase": 0.0}],
            [1, 2, {"type": "distance", "amplitude": 0.0, "phase": 0.0}],
            [2, 3, {"type": "distance", "amplitude": 0.0, "phase": 0.0}],
            [3, 0, {"type": "distance", "amplitude": 0.0, "phase": 0.0}],
            [3, 4, {"type": "distance", "amplitude": 0.0, "phase": 0.0}],
            [0, 4, {"type": "muscle", "amplitude": 5.0, "phase": 0.0}],
            [1, 4, {"type": "muscle", "amplitude": 10.0, "phase": 0.0}],
            [2, 4, {"type": "muscle", "amplitude": 2.0, "phase": 0.0}],
        ],
    },
}
