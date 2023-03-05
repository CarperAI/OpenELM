from typing import Any

from openelm.environments.environments import (
    BaseEnvironment,
    FunctionOptim,
    Genotype,
    ImageOptim,
    MatchString,
    Sodarace,
)

ENVS_DICT: dict[str, Any] = {"sodarace": Sodarace, "image_evolution": ImageOptim}

__all__ = [
    "Genotype",
    "BaseEnvironment",
    "FunctionOptim",
    "ImageOptim",
    "MatchString",
    "Sodarace",
    "ENVS_DICT",
]
