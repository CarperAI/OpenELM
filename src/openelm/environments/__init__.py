from typing import Any

from openelm.environments.environments import (
    BaseEnvironment,
    FunctionOptim,
    Genotype,
    ImageOptim,
    MatchString,
    P3Problem,
    PromptEvolution,
    P3ProbSol,
    Sodarace,
)
from openelm.map_elites import CVTMAPElites, MAPElites

ENVS_DICT: dict[str, Any] = {
    "sodarace": Sodarace,
    "image_evolution": ImageOptim,
    "p3": P3ProbSol,
    "prompt_evolution": PromptEvolution,
}

QD_DICT: dict[str, Any] = {
    "mapelites": MAPElites,
    "cvtmapelites": CVTMAPElites,
}

__all__ = [
    "Genotype",
    "BaseEnvironment",
    "FunctionOptim",
    "ImageOptim",
    "MatchString",
    "Sodarace",
    "ENVS_DICT",
    "QD_DICT",
    "P3Problem",
    "P3ProbSol",
]
