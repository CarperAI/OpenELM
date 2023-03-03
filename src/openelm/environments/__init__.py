from typing import Any

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
from openelm.mutation_model import (
    ImagePromptModel,
    SodaraceDiffModel,
    SodaracePromptModel,
)

# ----- Generate sample seeds and init args for environments -----
# They are simple template arguments to initialize several environments.


IMAGE_SEED = {
    "program_str": """import numpy as np
def draw_blue_rectangle() -> np.ndarray:
\tpic = np.zeros((32, 32, 3))
\tfor x in range(2, 30):
\t\tfor y in range(2, 30):
\t\t\tpic[x, y] = np.array([0, 0, 255])
\treturn pic
""",
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
}

image_init_args = {
    "seed": IMAGE_SEED,
    "target_img": target,
    "diff_model": None,
    "prompt_model": ImagePromptModel,
    "behavior_mode": "3-channel",
}

MODELS_DICT: dict[str, dict[str, Any]] = {
    "sodarace": {
        "prompt_model": SodaracePromptModel,
        "diff_model": SodaraceDiffModel,
    },
    "image_evolution": {
        "prompt_model": ImagePromptModel,
    },
}

ENVS_DICT: dict[str, Any] = {"sodarace": Sodarace, "image_evolution": ImageOptim}

__all__ = [
    "Genotype",
    "BaseEnvironment",
    "FunctionOptim",
    "ImageOptim",
    "MatchString",
    "Sodarace",
    "IMAGE_SEED",
    "SQUARE_SEED",
    "image_init_args",
    "sodarace_init_args",
]
