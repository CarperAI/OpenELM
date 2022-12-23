import numpy as np

from .environments import (
    BaseEnvironment,
    FunctionOptim,
    Genotype,
    ImageOptim,
    MatchString,
    Sodarace,
)

# ----- Generate sample seeds and init args for environments -----
# They are simple template arguments to initialize several environments.
# Sample usage:
#   from openelm.environment import sodarace_init_args
#   sodarace = Sodarace(**sodarace_init_args, run_name="test")


IMAGE_SEED = {
    "program_str": """import numpy as np
def draw_blue_rectangle() -> np.ndarray:
\tpic = np.zeros((32, 32, 3))
\tfor x in range(2, 30):
\t\tfor y in range(2, 30):
\t\t\tpic[x, y] = np.array([0, 0, 255])
\treturn pic
""",
    "result_obj": None,
    "error_code": 0,
}
exec(IMAGE_SEED["program_str"], globals())
IMAGE_SEED["result_obj"] = globals()["draw_blue_rectangle"]()
target = np.zeros((32, 32, 3))
for y in range(32):
    for x in range(32):
        if (y - 16) ** 2 + (x - 16) ** 2 <= 100:  # a radius-10 circle
            target[y, x] = np.array([1, 1, 0])


SQUARE_SEED = {
    "program_str": """from openelm.environments.sodaracer.walker.walk_creator import walker_creator


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
    wc.add_muscle(sides[0], center, 5.0, 0.0)
    wc.add_muscle(sides[1], center, 10.0, 0.0)
    wc.add_muscle(sides[2], center, 2.0, 0.0)

    return wc.get_walker()

""",
    "result_obj": {
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
    "error_code": 0,
}

# A sample init args for ImageOptim
image_init_args = {
    "seed": IMAGE_SEED,
    "config": "openelm/config/elm_image_cfg.yaml",
    "target_img": target,
    "diff_model": None,
    "behavior_mode": "3-channel",
}

# A sample init args for Sodarace
sodarace_init_args = {
    "seed": SQUARE_SEED,
    "config": "openelm/config/elm_sodarace_cfg.yaml",
    "diff_model": None,
    "eval_steps": 1000,
}

# ----- (Sample init args end) -----

__all__ = [
    "Genotype",
    "BaseEnvironment",
    "FunctionOptim",
    "ImageOptim",
    "MatchString",
    "Sodarace",
    "IMAGE_SEED",
    "image_init_args",
    "SQUARE_SEED",
    "sodarace_init_args",
]
