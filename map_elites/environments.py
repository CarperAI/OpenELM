import math
import string

import numpy as np
from numpy import array

from map_elites.map_elites import Genotype, Phenotype


def ackley(x: np.ndarray) -> np.ndarray:
    d = x.shape[-1]
    a = 5
    b = 0.1

    o1 = -a * np.exp(-b * np.sqrt(np.sum(x ** 2, axis=1) / d))
    o2 = -np.exp(np.sum(np.cos(math.tau * x) / d, axis=1))

    return -(a + math.exp(1) + o1 + o2)


# TODO: write base class
# find all local maxima of a multimodal function
class FunctionOptim:
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
class MatchString:
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


class Sodarace:
    def __init__(self, max_height: int = 10, max_width: int = 10, max_mass: int = 10,
                 ndim: int = 3, seed: str = None) -> None:
        self.seed = seed
        self.genotype_ndim = ndim
        self.genotype_space = np.array([[0, max_height], [0, max_width], [0, max_mass]]).T

    def generate_program(self, x: Genotype) -> Genotype:
        # Call LM to generate a new program.
        return NotImplementedError

    def get_characteristics(self, x: Genotype) -> list(float):
        # Call Sodaracers environment to get behaviour characteristics.
        raise NotImplementedError

    def fitness(self, x: Genotype) -> float:
        # Call Sodaracers environment to get the fitness.
        raise NotImplementedError

    def random(self) -> Genotype:
        return self.generate_program(self.seed)

    def mutate(self, x: Genotype) -> Genotype:
        return self.generate_program(x)

    def to_behaviour_space(self, x: Genotype) -> Phenotype:
        # Map from floats of h,w,m to behaviour space grid cells.
        return np.array(self.get_characteristics(x)).astype(int)

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

