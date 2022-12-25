import math
import string
from abc import ABC, abstractmethod
from typing import Generic, Optional, TypeVar, Union

import numpy as np
from omegaconf import DictConfig, OmegaConf

from openelm.diff_model import PromptMutationForImgTask, PromptMutationForSodarace
from openelm.environments.sodaracer import SodaraceSimulator

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


GenoType = TypeVar("GenoType", bound=Genotype)


class BaseEnvironment(ABC, Generic[GenoType]):
    def __init__(self) -> None:
        self.genotype_space: np.ndarray

    @abstractmethod
    def random(self, **kwarg) -> list[GenoType]:
        raise NotImplementedError

    @abstractmethod
    def mutate(self, x: GenoType, **kwarg) -> list[GenoType]:
        raise NotImplementedError

    @abstractmethod
    def fitness(self, x: GenoType) -> float:
        raise NotImplementedError

    @abstractmethod
    def to_behavior_space(self, x: GenoType) -> Optional[Phenotype]:
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

    @staticmethod
    def _load_config(config) -> DictConfig:
        if isinstance(config, str):
            return OmegaConf.load(config)
        elif isinstance(config, (dict, DictConfig)):
            return DictConfig(config)
        else:
            raise ValueError


class ArrayGenotype(Genotype, np.ndarray):
    def __new__(cls, input_array):
        obj = np.asarray(input_array).view(cls)
        return obj

    def __str__(self) -> str:
        return f'({", ".join(map(str, np.asarray(self)))})'


# find all local maxima of a multimodal function
class FunctionOptim(BaseEnvironment[ArrayGenotype]):
    def __init__(self, ndim=2):
        self.genotype_ndim = ndim
        self.genotype_space = np.repeat([[-4, 4]], self.genotype_ndim, axis=0).T

    def random(self, **kwarg) -> list[ArrayGenotype]:
        return [ArrayGenotype(np.random.uniform(*self.genotype_space))]

    def mutate(self, x: ArrayGenotype, **kwarg) -> list[ArrayGenotype]:
        x = x.copy()
        ix = np.random.randint(self.genotype_ndim)
        x[ix] = x[ix] + np.random.uniform(-1, 1)
        return [x]

    def fitness(self, x: ArrayGenotype) -> float:
        return ackley(x[None])[0]

    def to_behavior_space(self, x: ArrayGenotype) -> Phenotype:
        return np.asarray(x)


class ImageGeneration(Genotype):
    """Genotype for generated images."""

    def __init__(self, program_str: str, result_obj: dict, error_code: bool):
        self.program_str = program_str
        self.result_obj = result_obj
        self.error_code = error_code
        self.valid = self.validate()

    def __str__(self) -> str:
        if self.valid:
            assert isinstance(self.result_obj, np.ndarray)
            return str(self.result_obj.reshape((-1, 3)).mean(axis=0).astype(int))
        else:
            return ""

    def validate(self) -> bool:
        return (
            isinstance(self.result_obj, np.ndarray)
            and len(self.result_obj.shape) == 3
            and self.result_obj.shape[2] == 3
        )

    def three_channel_average(self) -> Phenotype:
        """
        Average RGB channels.

        Assume the input is of shape (height, width, channel), and we average each channel to get (channel,)
        """
        # Code with invalid return -> return a `None` Phenotype.
        if self.valid:
            assert isinstance(self.result_obj, np.ndarray)
            return np.average(self.result_obj.reshape((-1, 3)), axis=0)
        else:
            return None


class ImageOptim(BaseEnvironment[ImageGeneration]):
    """
    Mutate programs that return images.

    Fitness is simply the absolute difference between the returning
    image and the target image. To map into the behavior space,
    if behavior_mode=="3-channel", the image will be divided into blocks (specified in `block_size`), and average
    values of RGB channels in each block will be put together as a point in the behavior space (average-pooling).
    """

    default_diff_model_cls = PromptMutationForImgTask
    # Record different definitions of behavior spaces in a dict. Feel free to add.
    behavior_mode_spec = {"3-channel": {"genotype_ndim": 3}}

    def __init__(
        self,
        seed: dict,
        config: Union[str, dict, DictConfig],
        target_img: np.ndarray,
        diff_model,
        behavior_mode: str = "3-channel",
        run_name: Optional[str] = None,
    ):
        """
        Args:
            seed: the seed dict.
            config: the config file path or dict.
            target_img: the target image.
            diff_model: the diff model (or alternatives).
            behavior_mode: (Optional) a string indicating the way an individual is mapped into behavior space.
            run_name: (Optional) override the run_name in config.
        """
        if isinstance(seed, dict):
            self.seed = ImageGeneration(**seed)
        else:
            raise TypeError

        self.config = self._load_config(config)
        if run_name is not None:
            self.config.run_name = run_name

        self.target_img = target_img
        self.shape = target_img.shape

        if diff_model is None:
            self.diff_model = self.default_diff_model_cls(self.config)
        else:
            self.diff_model = diff_model

        self.behavior_mode = behavior_mode
        self.genotype_ndim: int = self.behavior_mode_spec[self.behavior_mode][
            "genotype_ndim"
        ]
        self.genotype_space = np.repeat([[0, 255]], self.genotype_ndim, axis=0).T

    def generate_program(self, code: str) -> list[ImageGeneration]:
        """
        Call LM to generate a new program and run it.

        Returns:
            An ImageGeneration object containing the code, the resulting image
            and the error code.
        """
        return [
            ImageGeneration(**generated)
            for generated in self.diff_model.generate_program(code)
        ]

    def random(self) -> list[ImageGeneration]:
        """
        Randomly generate a batch of codes and evaluate their outputs.

        Returns:
            a tuple of the code string and the returning result (None if there is error).
        """
        new_images = self.generate_program(self.seed.program_str)
        return new_images

    def mutate(self, x: ImageGeneration) -> list[ImageGeneration]:
        """
        Randomly mutate a batch of codes from a given individual and evaluate their outputs.

        Args:
            x: the individual to be mutated.

        Returns:
            a tuple of the code string and the returning result (None if there is error).
        """
        new_images = self.generate_program(x.program_str)
        return new_images

    def fitness(self, x: ImageGeneration) -> float:
        if not x.valid or x.result_obj.shape != self.shape:
            return -np.inf
        return -np.abs(x.result_obj - self.target_img).sum()

    def to_behavior_space(self, x: ImageGeneration) -> Optional[Phenotype]:
        if self.behavior_mode == "3-channel":
            return x.three_channel_average()
        return None


class StringArrayGenotype(ArrayGenotype):
    def __str__(self) -> str:
        x: np.ndarray = np.round(self)
        return "".join(
            string.ascii_letters[ix]
            for ix in np.clip(x.astype(int), 0, len(string.ascii_letters) - 1)
        )


class MatchString(BaseEnvironment[StringArrayGenotype]):
    # find a string by mutating one character at a time

    def __init__(self, target: str):
        self.alphabet = string.ascii_letters

        self.target = np.array([self.alphabet.index(ch) for ch in target])
        self.genotype_ndim = self.target.shape[0]
        self.genotype_space = np.repeat(
            [[0, len(self.alphabet)]], self.genotype_ndim, axis=0
        ).T

    def random(self, **kwarg) -> list[StringArrayGenotype]:
        return [StringArrayGenotype(np.random.uniform(*self.genotype_space))]

    def mutate(self, x: StringArrayGenotype, **kwarg) -> list[StringArrayGenotype]:
        x = x.copy()
        ix = np.random.randint(self.genotype_ndim)
        x[ix] = x[ix] + np.random.uniform(-5, 5)
        return [x]

    def fitness(self, x: StringArrayGenotype) -> float:
        return -np.abs(x - self.target).sum()

    def to_behavior_space(self, x: StringArrayGenotype) -> Phenotype:
        return np.asarray(x)


class Sodaracer(Genotype):
    def __init__(self, program_str: str, result_obj: dict, error_code: bool):
        """
        The Sodaracer genotype.

        Args:
            program_str: the string for the original code.
            result_obj: the dict of sodaracer.
            error_code: whether the code executes in the sandbox.
        """
        self.program_str = program_str
        self.result_obj = result_obj
        self.error_code = error_code

        # Check whether the Sodaracer is valid.
        if self.error_code == 0:
            try:
                # Test the Sodaracer by actually invoking all the necessary simulations/evaluations.
                self.simulator = SodaraceSimulator(body=self.result_obj)
                self.morphology = self.simulator.morphology
                self.evaluate(0)
                self.valid = True
            except Exception as e:
                self.valid = False
        else:
            self.valid = False

    def evaluate(self, timesteps: int) -> float:
        return self.simulator.evaluate(timesteps)

    def __str__(self) -> str:
        return self.program_str


class Sodarace(BaseEnvironment[Sodaracer]):
    default_diff_model_cls = PromptMutationForSodarace

    def __init__(
        self,
        seed: dict,
        config: Union[str, dict, DictConfig],
        diff_model,
        eval_steps: int,
        max_height: int = 1000,
        max_width: int = 1000,
        max_mass: int = 2000,
        ndim: int = 3,
        run_name: Optional[str] = None,
    ) -> None:
        """
        Sodarace environment.

        Args:
            seed: the seed dict.
            config: the config file path or dict.
            diff_model: the diff model (or alternatives).
            eval_steps: the number of steps for sodaracer evaluation.
            max_height: (Optional) the maximal height.
            max_width: (Optional) the maximal width.
            max_mass: (Optional) the maximal mass.
            ndim: (Optional) the dimension of behavior space.
            run_name: (Optional) override the run_name in config.
        """
        if isinstance(seed, dict):
            self.seed = Sodaracer(**seed)
        else:
            raise TypeError
        self.config = self._load_config(config)
        if run_name is not None:
            self.config.run_name = run_name

        if diff_model is None:
            self.diff_model = self.default_diff_model_cls(self.config)
        else:
            self.diff_model = diff_model

        self.eval_steps = eval_steps
        self.genotype_ndim = ndim
        self.genotype_space = np.array(
            [[0, max_height], [0, max_width], [0, max_mass]]
        ).T

    def generate_program(self, code: str) -> list[Sodaracer]:
        # Call LM to generate a new program and run it, returning a dict containing the program string
        # and the dict from running it.
        return [
            Sodaracer(**generated)
            for generated in self.diff_model.generate_program(code)
        ]

    def fitness(self, x: Sodaracer) -> float:
        # Call Sodaracers environment to get the fitness.
        if x.valid:
            return x.evaluate(self.eval_steps)
        else:
            return -np.inf

    def random(self) -> list[Sodaracer]:
        new_sodaracers = self.generate_program(self.seed.program_str)
        return new_sodaracers

    def mutate(self, x: Sodaracer) -> list[Sodaracer]:
        new_sodaracers = self.generate_program(x.program_str)
        return new_sodaracers

    def to_behavior_space(self, x: Sodaracer) -> Optional[Phenotype]:
        # Map from floats of h,w,m to behavior space grid cells.
        if x.valid:
            return np.array(
                [x.morphology["height"], x.morphology["width"], x.morphology["mass"]]
            ).astype(int)
        else:
            return None
