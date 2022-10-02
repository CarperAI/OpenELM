import math
import string
from abc import ABC, abstractmethod

import numpy as np
from numpy import array

from map_elites.map_elites import Genotype, Phenotype

from ..sodaracer_env import simulator


def ackley(x: np.ndarray) -> np.ndarray:
    d = x.shape[-1]
    a = 5
    b = 0.1

    o1 = -a * np.exp(-b * np.sqrt(np.sum(x ** 2, axis=1) / d))
    o2 = -np.exp(np.sum(np.cos(math.tau * x) / d, axis=1))

    return -(a + math.exp(1) + o1 + o2)


# class Genotype(ABC):


class BaseEnvironment(ABC):

    @abstractmethod
    def random(self) -> Genotype:
        raise NotImplementedError

    @abstractmethod
    def mutate(self, x: Genotype) -> Genotype:
        raise NotImplementedError

    @abstractmethod
    def fitness(self, x: Genotype) -> float:
        raise NotImplementedError

    @abstractmethod
    def to_behaviour_space(self, x: Genotype) -> Phenotype:
        raise NotImplementedError

    @abstractmethod
    def to_string(self, x: Genotype) -> str:
        raise NotImplementedError


# find all local maxima of a multimodal function
class FunctionOptim(BaseEnvironment):
    def __init__(self, ndim=2):
        self.genotype_ndim = ndim
        self.genotype_space = np.repeat([[-4, 4]], self.genotype_ndim, axis=0).T

    def random(self) -> Genotype:
        return np.random.uniform(*self.genotype_space)

    def mutate(self, x: Genotype) -> Genotype:
        x = x.copy()
        ix = np.random.randint(self.genotype_ndim)
        x[ix] = x[ix] + np.random.uniform(-1, 1)
        return x

    def fitness(self, x: Genotype) -> float:
        return ackley(x[None])[0]

    def to_behaviour_space(self, x: Genotype) -> Phenotype:
        return x

    def to_string(self, x: Genotype) -> str:
        return f'({", ".join(map(str, x))})'

    @property
    def max_fitness(self):
        return 0

    @property
    # [starts, endings) of search intervals
    def behaviour_space(self):
        return self.genotype_space

    @property
    def behaviour_ndim(self):
        return self.behaviour_space.shape[1]


# find a string by mutating one character at a time
class MatchString(BaseEnvironment):
    def __init__(self, target: str):
        self.alphabet = string.ascii_letters

        self.target = array([self.alphabet.index(ch) for ch in target])
        self.genotype_ndim = self.target.shape[0]
        self.genotype_space = np.repeat([[0, len(self.alphabet)]], self.genotype_ndim, axis=0).T

    def random(self) -> Genotype:
        return np.random.uniform(*self.genotype_space)

    def mutate(self, x: Genotype) -> Genotype:
        x = x.copy()
        ix = np.random.randint(self.genotype_ndim)
        x[ix] = x[ix] + np.random.uniform(-5, 5)
        return x

    def fitness(self, x: Genotype) -> float:
        return -np.abs(x - self.target).sum()

    def to_behaviour_space(self, x: Genotype) -> Phenotype:
        return x

    def to_string(self, x: Genotype) -> str:
        return ''.join(self.alphabet[ix] for ix in np.clip(np.round(x).astype(int), 0, len(self.alphabet) - 1))

    @property
    def max_fitness(self):
        return 0

    @property
    # [starts, endings) of search intervals
    def behaviour_space(self):
        return self.genotype_space

    @property
    def behaviour_ndim(self):
        return self.behaviour_space.shape[1]


# class Sodaracer(Genotype):
#     def __init__(self, program_str: str):
#         self.program_str = program_str


class Sodarace(BaseEnvironment):
    def __init__(self, seed: dict, diff_model, max_height: int = 100, max_width: int = 100, max_mass: int = 100,
                 ndim: int = 3) -> None:
        self.seed = seed
        self.diff_model = diff_model
        self.genotype_ndim = ndim
        self.genotype_space = np.array([[0, max_height], [0, max_width], [0, max_mass]]).T

        self.simulator = simulator.SodaraceSimulator(body=self.seed["sodaracer"])

    def generate_program(self, x: str) -> Genotype:
        # Call LM to generate a new program and run it, retuning a dict containing the program string
        # and the dict from running
        return self.diff_model.generate_program(seed=x)

    def fitness(self, x: Genotype) -> float:
        # Call Sodaracers environment to get the fitness.
        return self.simulator.evaluate(x)

    def random(self) -> Genotype:
        program_dict = self.generate_program(self.seed["program_str"])
        # TODO: consider storing morphology dict inside genotype?
        self.simulator = simulator.SodaraceSimulator(body=program_dict["sodaracer"])
        return program_dict

    def mutate(self, x: Genotype) -> Genotype:
        # TODO: maybe create proper Genotype class.
        program_dict = self.generate_program(x["program_str"])
        self.simulator = simulator.SodaraceSimulator(body=program_dict["sodaracer"])
        return program_dict

    def to_behaviour_space(self, x: Genotype) -> Phenotype:
        # Map from floats of h,w,m to behaviour space grid cells.
        # TODO: Implement this.
        morphology = self.simulator.morphology
        return np.array([morphology['height'], morphology['width'], morphology['mass']]).astype(int)

    def to_string(self, x: Genotype) -> str:
        return str(x)

    @property
    def max_fitness(self):
        return 0

    @property
    # [starts, endings) of search intervals
    def behaviour_space(self):
        return self.genotype_space

    @property
    def behaviour_ndim(self):
        return self.behaviour_space.shape[1]
