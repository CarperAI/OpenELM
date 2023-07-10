import json
import math
import string
import sys
import warnings
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Generic, Optional, TypeVar, Union

import numpy as np
import requests

from openelm.configs import EnvConfig, ImageEnvConfig, StringEnvConfig
from openelm.environments.utils import NULL_SEED, get_image_target
from openelm.mutation_model import MutationModel
from openelm.utils.code_eval import pool_exec_processes

if (
    (sys.version_info >= (3, 9, 14) and sys.version_info <= (3, 10))
    or (sys.version_info >= (3, 10, 7) and sys.version_info <= (3, 11))
    or sys.version_info >= (3, 11)
):
    # remove length limitation for int->str conversion
    # (model sometimes outputs really long ints)
    sys.set_int_max_str_digits(0)

Phenotype = Optional[np.ndarray]


def ackley(x: np.ndarray) -> np.ndarray:
    d = x.shape[-1]
    a = 5
    b = 0.1

    o1 = -a * np.exp(-b * np.sqrt(np.sum(x**2, axis=1) / d))
    o2 = -np.exp(np.sum(np.cos(math.tau * x) / d, axis=1))

    return -(a + math.exp(1) + o1 + o2)


def numpy_to_ascii_art(arr):
    """Convert a numpy array with dimensions (width, height, channels) to ascii art."""
    art_chars = " .:-=#"
    im = np.sum(arr, axis=-1)  # we can't do colors
    idx = np.round(np.interp(im, (im.min(), im.max()), (0, len(art_chars) - 1))).astype(
        "int"
    )
    chars = np.choose(idx, art_chars)
    ascii_art = "\n".join(["".join(x) for x in chars])
    return ascii_art


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
    def get_rng_state(self) -> Optional[np.random._generator.Generator]:
        raise NotImplementedError

    @abstractmethod
    def set_rng_state(self, rng_state: Optional[np.random._generator.Generator]):
        raise NotImplementedError

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
    def __init__(self, ndim=2, seed=None):
        self.genotype_ndim = ndim
        self.genotype_space = np.repeat([[-4, 4]], self.genotype_ndim, axis=0).T
        self.batch_size: int = 1
        self.rng = np.random.default_rng(seed)

    def get_rng_state(self) -> Optional[np.random._generator.Generator]:
        return self.rng

    def set_rng_state(self, rng_state: Optional[np.random._generator.Generator]):
        self.rng = rng_state

    def random(self) -> list[ArrayGenotype]:
        return [
            ArrayGenotype(self.rng.uniform(*self.genotype_space))
            for _ in range(self.batch_size)
        ]

    def mutate(self, x: list[ArrayGenotype]) -> list[ArrayGenotype]:
        for i in range(self.batch_size):
            ix = self.rng.integers(self.genotype_ndim)
            x[i][ix] = x[i][ix] + self.rng.uniform(-1, 1)
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

    def __init__(self, config: StringEnvConfig):
        self.alphabet = string.ascii_letters

        self.config: StringEnvConfig = config
        self.batch_size = self.config.batch_size
        self.target = np.array([self.alphabet.index(ch) for ch in self.config.target])
        self.genotype_ndim = self.target.shape[0]
        self.genotype_space = np.repeat(
            [[0, len(self.alphabet)]], self.genotype_ndim, axis=0
        ).T
        self.rng = np.random.default_rng(self.config.seed)

    def get_rng_state(self) -> Optional[np.random._generator.Generator]:
        return self.rng

    def set_rng_state(self, rng_state: Optional[np.random._generator.Generator]):
        self.rng = rng_state

    def random(self) -> list[StringArrayGenotype]:
        return [
            StringArrayGenotype(self.rng.uniform(*self.genotype_space))
            for _ in range(self.batch_size)
        ]

    def mutate(self, genomes: list[StringArrayGenotype]) -> list[StringArrayGenotype]:
        x = deepcopy(genomes)
        for i in range(self.batch_size):
            ix = self.rng.integers(self.genotype_ndim)
            x[i][ix] = x[i][ix] + self.rng.uniform(-1, 1)
        return x

    def fitness(self, x: StringArrayGenotype) -> float:
        return -np.abs(x.to_phenotype() - self.target).sum()


class ImageGeneration(Genotype):
    """Genotype for generated images."""

    def __init__(self, program_str: str, result_obj: np.ndarray):
        self.program_str = program_str
        self.result_obj = result_obj
        self.valid = self.validate()

    def __str__(self) -> str:
        if self.valid:
            return numpy_to_ascii_art(self.result_obj)
            # return str(self.result_obj.reshape((-1, 3)).mean(axis=0).astype(int))
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

    def visualize(self, ax) -> None:
        if self.valid:
            ax.imshow(self.result_obj)


class ImageOptim(BaseEnvironment[ImageGeneration]):
    """
    Mutate programs that return images.

    Fitness is simply the absolute difference between the returning
    image and the target image. To map into the behavior space,
    if behavior_ndims=="3-channel", the image will be divided into blocks
    (specified in `block_size`), and average
    values of RGB channels in each block will be put together as a point in the
    behavior space (average-pooling).
    """

    # Record different definitions of behavior spaces in a dict.
    behavior_ndims = {"3-channel": 3}

    def __init__(
        self,
        config: ImageEnvConfig,
        mutation_model: MutationModel,
    ):
        self.config: ImageEnvConfig = config
        self.batch_size = self.config.batch_size
        self.target_img: np.ndarray = get_image_target(self.config.target)
        self.seed: str = NULL_SEED
        self.mutation_model: MutationModel = mutation_model

        self.behavior_mode: str = self.config.behavior_mode
        self.genotype_ndim: int = self.behavior_ndims[self.behavior_mode]
        self.genotype_space = np.repeat([[0, 255]], self.genotype_ndim, axis=0).T
        self.rng = None

    def get_rng_state(self) -> Optional[np.random._generator.Generator]:
        warnings.warn("WARNING: rng state not used in this environment")
        return None

    def set_rng_state(self, rng_state: Optional[np.random._generator.Generator]):
        warnings.warn("WARNING: rng state not used in this environment")
        pass

    def construct_prompt(
        self, code_batch: Optional[Union[list[str], str]] = None
    ) -> dict[str, str]:
        prompt_str: str = """
import math
import numpy as np
"""
        instruction_str: str = """
# Fixed version of draw()
def draw():
    \"\"\"
    Draws a yellow circle with radius 10 in the middle of a 32 by 32 black image.

    Returns:
        np.ndarray: the image
    \"\"\"
    pic = np.zeros((32, 32, 3))
"""
        import_str: str = prompt_str
        if code_batch is None:
            # Initialization steps
            prompt_str += self.seed
        else:
            prompt_str += """
# Old version of draw()
# TODO: fix bugs in the code below

"""
            # Evolution steps
            if isinstance(code_batch, list):
                prompt_str += code_batch[0]
            elif isinstance(code_batch, str):
                prompt_str += code_batch
        import_str += instruction_str
        prompt_str += instruction_str
        return {"prompt": prompt_str, "template": import_str}

    def generate_programs(
        self, code_batch: list[dict[str, str]]
    ) -> list[ImageGeneration]:
        func_name: str = "draw"
        generated_programs = self.mutation_model.generate_programs(
            code_batch, local_scope_truncate=True
        )
        if self.config.sandbox:
            results = []
            for code in generated_programs:
                resp = requests.post(
                    f"{self.config.sandbox_server}/eval_imageoptim_func",
                    json={
                        "code": code,
                        "func_name": func_name,
                        "timeout": self.config.timeout,
                    },
                    timeout=self.config.timeout,
                )
                if resp.status_code == 200:
                    return_dict = json.loads(resp.text)
                    results.append(return_dict)
            return [ImageGeneration(**p) for p in results]
        # for i in range(len(results)):
        #     results[i]["result_obj"] = np.array(results[i]["result_obj"])
        # return results
        else:
            results = pool_exec_processes(
                generated_programs,
                func_name=func_name,
                timeout=self.config.timeout,
                processes=self.config.processes,
                debug=self.config.debug,
            )
            result_list: list = []
            for i, result in enumerate(results):
                try:
                    if isinstance(result, np.ndarray):
                        result_list.append(
                            {
                                "program_str": generated_programs[i],
                                "result_obj": result,
                            }
                        )
                    else:
                        if self.config.debug:
                            print("Failed execution, type:", result)
                            print(generated_programs[i])
                except Exception as e:
                    if self.config.debug:
                        print(type(e), e)
            return [ImageGeneration(**p) for p in result_list]

    def random(self) -> list[ImageGeneration]:
        program_list = [self.construct_prompt() for _ in range(self.config.batch_size)]
        new_images = self.generate_programs(program_list)
        return new_images

    def mutate(self, images_list: list[ImageGeneration]) -> list[ImageGeneration]:
        images = [img.program_str for img in images_list]
        program_list = list(map(self.construct_prompt, images))
        new_images = self.generate_programs(program_list)
        return new_images

    def fitness(self, x: ImageGeneration) -> float:
        if not x.valid or x.result_obj.shape != self.target_img.shape:
            return -np.inf
        return -np.abs(x.result_obj - self.target_img).sum()
