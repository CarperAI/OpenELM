from typing import Any

from openelm.environments.environments import BaseEnvironment, Genotype
from openelm.map_elites import CVTMAPElites, MAPElites


def load_env(env_name: str) -> Any:
    if env_name == "sodarace":
        from openelm.environments.sodarace_env import Sodarace

        return Sodarace
    elif env_name == "image_evolution":
        from openelm.environments.environments import ImageOptim

        return ImageOptim
    elif env_name == "match_string":
        from openelm.environments.environments import MatchString

        return MatchString
    elif env_name == "function_optim":
        from openelm.environments.environments import FunctionOptim

        return FunctionOptim
    elif env_name == "p3_probsol":
        from openelm.environments.p3_env import P3ProbSol

        return P3ProbSol
    elif env_name == "p3_problem":
        from openelm.environments.p3_env import P3Problem

        return P3Problem
    elif env_name == "prompt_evolution":
        from openelm.environments.prompt_env import PromptEvolution

        return PromptEvolution
    elif env_name == "qdaif":
        from openelm.environments.environments import PoetryEvolution

        return PoetryEvolution
    else:
        raise ValueError(f"Unknown environment {env_name}")


QD_DICT: dict[str, Any] = {
    "mapelites": MAPElites,
    "cvtmapelites": CVTMAPElites,
}

__all__ = [
    "Genotype",
    "BaseEnvironment",
    "QD_DICT",
    "load_env",
]
