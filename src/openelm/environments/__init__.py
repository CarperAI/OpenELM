import numpy as np

from openelm.environments.environments import (
    BaseEnvironment,
    FunctionOptim,
    Genotype,
    ImageOptim,
    MatchString,
    Sodarace,
)
from openelm.environments.sodaracer import IMPORTS, SQUARE, SQUARE_PREREQ

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
    "program_str": IMPORTS + SQUARE_PREREQ + SQUARE,
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
