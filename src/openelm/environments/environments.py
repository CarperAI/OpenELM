import json
import math
import string
from abc import ABC, abstractmethod
from typing import Generic, Optional, TypeVar, Union

import numpy as np
import requests

from openelm.configs import EnvConfig, ImageEnvConfig, SodaraceEnvConfig
from openelm.environments.sodaracer import (
    CIRCLE,
    GALLOPER_PREREQ,
    IMPORTS,
    INSTRUCTIONS,
    QUERY_CPPN,
    SEEDS_DICT,
    SQUARE_PREREQ,
    SodaraceSimulator,
    Walker,
)
from openelm.mutation_model import MutationModel
from openelm.utils.code_eval import pool_exec_processes

Phenotype = Optional[np.ndarray]


def ackley(x: np.ndarray) -> np.ndarray:
    d = x.shape[-1]
    a = 5
    b = 0.1

    o1 = -a * np.exp(-b * np.sqrt(np.sum(x**2, axis=1) / d))
    o2 = -np.exp(np.sum(np.cos(math.tau * x) / d, axis=1))

    return -(a + math.exp(1) + o1 + o2)


class Genotype(ABC):
    def __str__(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def to_phenotype(self) -> Optional[Phenotype]:
        raise NotImplementedError


GenoType = TypeVar("GenoType", bound=Genotype)


class BaseEnvironment(ABC, Generic[GenoType]):
    def __init__(self) -> None:
        self.genotype_space: np.ndarray
        self.batch_size: int
        self.config: EnvConfig

    @abstractmethod
    def random(self) -> list[GenoType]:
        raise NotImplementedError

    @abstractmethod
    def mutate(self, x: list[GenoType]) -> list[GenoType]:
        raise NotImplementedError

    @abstractmethod
    def fitness(self, x: GenoType) -> float:
        raise NotImplementedError

    @property
    def max_fitness(self) -> int:
        return 0

    @property
    # [starts, endings) of search intervals
    def behavior_space(self) -> np.ndarray:
        return self.genotype_space

    @property
    def behavior_ndim(self) -> int:
        return self.behavior_space.shape[1]


class ArrayGenotype(Genotype, np.ndarray):
    def __new__(cls, input_array):
        obj = np.asarray(input_array).view(cls)
        return obj

    def __str__(self) -> str:
        return f'({", ".join(map(str, np.asarray(self)))})'

    def to_phenotype(self) -> Phenotype:
        return np.asarray(self)


# find all local maxima of a multimodal function
class FunctionOptim(BaseEnvironment[ArrayGenotype]):
    def __init__(self, ndim=2):
        self.genotype_ndim = ndim
        self.genotype_space = np.repeat([[-4, 4]], self.genotype_ndim, axis=0).T
        self.batch_size: int = 1

    def random(self) -> list[ArrayGenotype]:
        return [
            ArrayGenotype(np.random.uniform(*self.genotype_space))
            for _ in range(self.batch_size)
        ]

    def mutate(self, x: list[ArrayGenotype]) -> list[ArrayGenotype]:
        for i in range(self.batch_size):
            ix = np.random.randint(self.genotype_ndim)
            x[i][ix] = x[i][ix] + np.random.uniform(-1, 1)
        return x

    def fitness(self, x: ArrayGenotype) -> float:
        return ackley(x[None])[0]


class StringArrayGenotype(ArrayGenotype):
    def __str__(self) -> str:
        x: np.ndarray = np.round(self)
        return "".join(
            string.ascii_letters[ix]
            for ix in np.clip(x.astype(int), 0, len(string.ascii_letters) - 1)
        )

    def to_phenotype(self) -> Phenotype:
        return np.asarray(self)


class MatchString(BaseEnvironment[StringArrayGenotype]):
    # find a string by mutating one character at a time

    def __init__(self, target: str):
        self.alphabet = string.ascii_letters

        self.target = np.array([self.alphabet.index(ch) for ch in target])
        self.genotype_ndim = self.target.shape[0]
        self.genotype_space = np.repeat(
            [[0, len(self.alphabet)]], self.genotype_ndim, axis=0
        ).T

    def random(self) -> list[StringArrayGenotype]:
        return [
            StringArrayGenotype(np.random.uniform(*self.genotype_space))
            for _ in range(self.batch_size)
        ]

    def mutate(self, x: list[StringArrayGenotype]) -> list[StringArrayGenotype]:
        for i in range(self.batch_size):
            ix = np.random.randint(self.genotype_ndim)
            x[i][ix] = x[i][ix] + np.random.uniform(-1, 1)
        return x

    def fitness(self, x: StringArrayGenotype) -> float:
        return -np.abs(x - self.target).sum()


class ImageGeneration(Genotype):
    """Genotype for generated images."""

    def __init__(self, program_str: str, result_obj: np.ndarray):
        self.program_str = program_str
        self.result_obj = result_obj
        self.valid = self.validate()

    def __str__(self) -> str:
        if self.valid:
            return str(self.result_obj.reshape((-1, 3)).mean(axis=0).astype(int))
        else:
            return ""

    def validate(self) -> bool:
        return len(self.result_obj.shape) == 3 and self.result_obj.shape[2] == 3

    def to_phenotype(self, mode: str = "3-channel-avg") -> Optional[Phenotype]:
        if not self.valid:
            return None
        if mode == "3-channel-avg":
            # Average RGB channels.
            # Assume the input is of shape (height, width, channel), and we
            # average each channel to get (channel,)
            return np.average(self.result_obj.reshape((-1, 3)), axis=0)
        else:
            return None


class ImageOptim(BaseEnvironment[ImageGeneration]):
    """
    Mutate programs that return images.

    Fitness is simply the absolute difference between the returning
    image and the target image. To map into the behavior space,
    if behavior_mode=="3-channel", the image will be divided into blocks
    (specified in `block_size`), and average
    values of RGB channels in each block will be put together as a point in the
    behavior space (average-pooling).
    """

    # Record different definitions of behavior spaces in a dict. Feel free to add.
    behavior_mode_spec = {"3-channel-avg": {"genotype_ndim": 3}}

    def __init__(
        self,
        seed: Union[dict, ImageGeneration],
        target_img: np.ndarray,
        config: ImageEnvConfig,
        mutation_model: MutationModel,
        behavior_mode: str = "3-channel",
    ):
        """
        Mutate programs that return images.

        Fitness is simply the absolute difference between the returning
        image and the target image. To map into the behavior space,
        if behavior_mode=="3-channel", the image will be divided into blocks
        (specified in `block_size`), and average values of RGB channels in each
        block will be put together as a point in the behavior space (average-pooling).

        Args:
            seed: the seed dict.
            config: the config dict.
            target_img: the target image.
            mutation_model: the diff model (or alternatives).
            behavior_mode: (Optional) a string indicating the way an individual
            is mapped into behavior space.
        """
        if isinstance(seed, dict):
            self.seed = ImageGeneration(**seed)
        elif not isinstance(seed, ImageGeneration):
            raise TypeError

        self.target_img = target_img
        self.shape = target_img.shape

        self.mutation_model: MutationModel = mutation_model

        self.behavior_mode = behavior_mode
        self.genotype_ndim: int = self.behavior_mode_spec[self.behavior_mode][
            "genotype_ndim"
        ]
        self.genotype_space = np.repeat([[0, 255]], self.genotype_ndim, axis=0).T

    def generate_programs(self, code_batch: list[str]) -> list[ImageGeneration]:
        """
        Call LM to generate a new program and run it.

        Returns:
            An ImageGeneration object containing the code, the resulting image
            and the error code.
        """
        generated_programs = self.mutation_model.generate_programs(code_batch)
        return [ImageGeneration(**p) for p in generated_programs]

    def random(self) -> list[ImageGeneration]:
        """
        Randomly generate a batch of codes and evaluate their outputs.

        Returns:
            a tuple of the code string and the returning result (None if there
            is error).
        """
        program_str_list = [self.seed.program_str] * self.batch_size
        new_images = self.generate_programs(program_str_list)
        return new_images

    def mutate(self, images_list: list[ImageGeneration]) -> list[ImageGeneration]:
        """
        Randomly mutate a batch of codes and evaluate their outputs.

        Args:
            x: the individual to be mutated.

        Returns:
            a tuple of the code string and the returning result (None if there
            is an error).
        """
        program_str_list = [sr.program_str for sr in images_list]
        new_images = self.generate_programs(program_str_list)
        return new_images

    def fitness(self, x: ImageGeneration) -> float:
        if not x.valid or x.result_obj.shape != self.shape:
            return -np.inf
        return -np.abs(x.result_obj - self.target_img).sum()


class Sodaracer(Genotype):
    def __init__(self, program_str: str, result_obj: dict):
        """
        The Sodaracer genotype.

        Args:
            program_str: the string for the original code.
            result_obj: the dict of sodaracer.
        """
        self.program_str: str = program_str
        self.result_obj: dict = result_obj
        # self._fitness: Optional[float] = None

        # Check whether the Sodaracer is valid.
        try:
            # Test the Sodaracer by instantiating a simulation.
            self.simulator = SodaraceSimulator(body=self.result_obj)
            self.morphology = self.simulator.morphology
            self.evaluate(0)
            self.valid = True
        except Exception:
            self.valid = False

    def evaluate(self, eval_ms: int) -> float:
        self._fitness = self.simulator.evaluate(eval_ms)
        # if self._fitness is None:
        return self._fitness

    def __str__(self) -> str:
        return self.program_str

    def to_phenotype(self) -> Optional[Phenotype]:
        if self.valid:
            return np.array(
                [
                    self.morphology["height"],
                    self.morphology["width"],
                    self.morphology["mass"],
                ]
            ).astype(int)
        else:
            return None

    @property
    def fitness(self) -> Optional[float]:
        return self._fitness


class Sodarace(BaseEnvironment[Sodaracer]):
    def __init__(
        self,
        seeds: list[Union[dict, Sodaracer]],
        config: SodaraceEnvConfig,
        mutation_model: MutationModel,
    ) -> None:
        """
        Sodarace environment.

        Args:
            seeds: the seed dict.
            config: the environment config.
            mutation_model: the mutation model.
        """
        if isinstance(seeds, dict):
            self.seeds: Sodaracer = Sodaracer(**seeds)
        elif not isinstance(seeds, Sodaracer):
            raise TypeError
        self.config: SodaraceEnvConfig = config
        self.mutation_model: MutationModel = mutation_model

        self.genotype_space = np.array(self.config.behavior_space).T
        self.genotype_ndim = self.genotype_space.shape[1]

        self.seed_strs: list[str] = self.config.starting_seeds

    def construct_prompt(
        self, code_batch: Optional[Union[list[str], str]] = None
    ) -> dict[str, str]:
        prompt_str: str = IMPORTS
        if "square" in self.seed_strs:
            prompt_str += SQUARE_PREREQ
        if "galloper" in self.seed_strs:
            prompt_str += GALLOPER_PREREQ
        if "radial" in self.seed_strs or "wheel" in self.seed_strs:
            prompt_str += CIRCLE
        if (
            "cppn_fixed" in self.seed_strs
            or "cppn_mutable" in self.seed_strs
            or "runner" in self.seed_strs
        ):
            prompt_str += QUERY_CPPN
        # For crossover:
        # If init steps, combine seeds and prereqs, and use instruction 3 code below.
        # For all other steps, prepend all prereqs and ignore instruction 3 code.
        # For non-crossover
        # Always preprend prereq, and len(code_batch) == 1
        import_str: str = prompt_str
        if code_batch is None:
            # Initialization steps
            seeds = [SEEDS_DICT[seed] for seed in self.seed_strs]
            if not self.config.crossover:
                # TODO: Sample from seeds randomly
                prompt_str += seeds[0]
            elif self.config.crossover:
                if self.config.instruction == 3:
                    instruction_str: str = INSTRUCTIONS[self.config.instruction].split(
                        ","
                    )[0]
                for seed in seeds:
                    prompt_str += seed
                    if self.config.instruction == 3:
                        reverse_seeds: dict[str, str] = {
                            v: k for k, v in SEEDS_DICT.items()
                        }
                        instruction_str += reverse_seeds[seed] + ", "
                if self.config.instruction == 3:
                    instruction_str += INSTRUCTIONS[self.config.instruction].split(",")[
                        1
                    ]
                raise NotImplementedError
        else:
            # Evolution steps
            if not self.config.crossover:
                if isinstance(code_batch, list):
                    prompt_str += code_batch[0]
                elif isinstance(code_batch, str):
                    prompt_str += code_batch
            elif self.config.crossover:
                # Crossover
                raise NotImplementedError
        instruction_str = INSTRUCTIONS[self.config.instruction]
        import_str += instruction_str
        prompt_str += instruction_str
        return {"prompt": prompt_str, "template": import_str}

    def generate_programs(self, code_batch: list[dict[str, str]]) -> list[Sodaracer]:
        """Generate new programs with a mutation model and evaluate them."""
        local_scope_exec: bool = self.config.instruction != 0
        generated_programs = self.mutation_model.generate_programs(
            code_batch, local_scope_exec
        )
        if self.config.sandbox:
            results = []
            for code in generated_programs:
                resp = requests.post(
                    f"{self.config.sandbox_server}/gen_racer",
                    json={"code": code, "timeout": self.config.timeout},
                    timeout=self.config.timeout,
                )
                if resp.status_code == 200:
                    return_dict = json.loads(resp.text)
                    results.append(return_dict)
            return [Sodaracer(**p) for p in results]
        else:
            results = pool_exec_processes(
                generated_programs,
                func_name="make_walker",
                timeout=self.config.timeout,
                processes=self.config.processes,
                debug=self.config.debug,
            )
            result_list: list = []
            for i, result in enumerate(results):
                try:
                    if isinstance(result, Walker) and result.validate():
                        result_list.append(
                            {
                                "program_str": generated_programs[i],
                                "result_obj": result.to_dict(),
                            }
                        )
                    else:
                        if self.config.debug:
                            print("Failed execution, type:", result)
                            print(generated_programs[i])
                except Exception as e:
                    if self.config.debug:
                        print(type(e), e)
            return [Sodaracer(**p) for p in result_list]

    def random(self) -> list[Sodaracer]:
        program_list = [self.construct_prompt() for _ in range(self.config.batch_size)]
        new_sodaracers = self.generate_programs(program_list)
        return new_sodaracers

    def mutate(self, sodaracer_list: list[Sodaracer]) -> list[Sodaracer]:
        program_str_list = [sr.program_str for sr in sodaracer_list]
        new_sodaracers = self.generate_programs(program_str_list)
        return new_sodaracers

    def fitness(self, x: Sodaracer) -> float:
        # Call Sodaracers environment to get the fitness.
        if x.valid:
            return x.evaluate(self.config.eval_ms)
        else:
            return -np.inf
