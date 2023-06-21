from typing import Any

from openelm.environments.environments import (
    BaseEnvironment,
    FunctionOptim,
    Genotype,
    ImageOptim,
    MatchString,
    P3Problem,
    P3ProbSol,
    PoetryEvolution,
    PromptEvolution,
    Sodarace,
)
from openelm.map_elites import CVTMAPElites, MAPElites

ENVS_DICT: dict[str, Any] = {
    "sodarace": Sodarace,
    "image_evolution": ImageOptim,
    "p3_probsol": P3ProbSol,
    "p3_problem": P3Problem,
    "prompt_evolution": PromptEvolution,
    "qdaif": PoetryEvolution,
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
