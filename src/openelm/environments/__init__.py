from typing import Any

from openelm.algorithms.map_elites import CVTMAPElites, MAPElites
from openelm.environments.base import BaseEnvironment, Genotype


def load_env(env_name: str) -> Any:
    if env_name == "sodarace":
        from openelm.environments.sodaracer.sodarace import Sodarace

        return Sodarace
    elif env_name == "image_evolution":
        from openelm.environments.base import ImageOptim

        return ImageOptim
    elif env_name == "match_string":
        from openelm.environments.base import MatchString

        return MatchString
    elif env_name == "function_optim":
        from openelm.environments.base import FunctionOptim

        return FunctionOptim
    elif env_name == "p3_probsol":
        from openelm.environments.p3.p3 import P3ProbSol

        return P3ProbSol
    elif env_name == "p3_problem":
        from openelm.environments.p3.p3 import P3Problem

        return P3Problem
    elif env_name == "prompt_evolution":
        from openelm.environments.prompt.prompt import PromptEvolution

        return PromptEvolution
    elif env_name == "qdaif":
        from openelm.environments.poetry import PoetryEvolution

        return PoetryEvolution
    else:
        raise ValueError(f"Unknown environment {env_name}")


def load_algorithm(algorithm_name: str) -> Any:
    if algorithm_name == "mapelites":
        return MAPElites
    elif algorithm_name == "cvtmapelites":
        return CVTMAPElites


__all__ = [
    "Genotype",
    "BaseEnvironment",
    "load_algorithm",
    "load_env",
]
