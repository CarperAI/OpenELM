import functools
import json
import math
import string
from abc import ABC, abstractmethod
from typing import Tuple, Union, List

import numpy as np
import requests
import torch
from omegaconf import DictConfig, OmegaConf

from codegen.codegen_utilities import model_setup, sample, truncate
from diff_model import DiffModel
from numpy import array
from sodaracer_env import simulator

from map_elites.map_elites import Genotype, Phenotype


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


class ImageOptim(BaseEnvironment):
    """
    This will try to mutate programs that return images. Fitness is simply the absolute difference between the returning
    image and the target image.
    To map into the behaviour space, the image will be divided into blocks (specified in `block_size`), and average
    values of RGB channels in each block will be put together as a point in the behaviour space (basically it is
    average-pooling).
    """
    default_extra_string = '\t"""Draw a yellow circle.\n\t"""'
    default_import = 'import math\nimport numpy as np'

    def __init__(self, seed: str, config: Union[str, dict, DictConfig], target_img: np.ndarray, func_name: str,
                 block_size=(16, 16), extra_string_for_mutation=default_extra_string, import_for_mutation=default_import,
                 sandbox_server='localhost:5000'):
        """
        Parameters:
            seed: the seed string.
            config: the config file or dict.
            target_img: the target image.
            func_name: the name of the function to be called to return images.
            block_size: (Optional) the size of each block (used to calculate the behavior space).
            extra_string_for_mutation: (Optional) the extra string attached under the function definition in prompt.
            import_for_mutation: (Optional) the import lines for the prompt while mutating.
            sandbox_server: (Optional) the address of sandbox server: 'domain:port'.
        """
        self.seed = seed
        if isinstance(config, str):
            self.config = OmegaConf.load(config)
        elif isinstance(config, (dict, DictConfig)):
            self.config = DictConfig(config)
        else:
            raise ValueError

        self.target_img = target_img
        self.shape = target_img.shape
        self.func_name = func_name
        self.block_size = block_size
        self.import_prompt = import_for_mutation
        self.extra_prompt = f'\ndef {self.func_name}():\n{extra_string_for_mutation}\n\tpic = np.zeros({self.shape})'

        self.model, self.tokenizer = model_setup(self.config)

        # Use RNG to rotate random seeds during inference.
        self.rng = np.random.default_rng(seed=self.config.seed)

        height, width, _ = self.shape
        self.genotype_ndim = 3 * math.ceil(height / block_size[0]) * math.ceil(width / block_size[1])
        self.genotype_space = np.repeat([[0, 255]], self.genotype_ndim, axis=0).T

        self.sandbox_server = sandbox_server

    def random(self) -> Genotype:
        code = self._generate_code(self.seed + self.extra_prompt)[0]
        result = self._evaluate_code(self.extra_prompt + code)
        # If needed, the result can be further classified based on the error type.
        return code, None if isinstance(result, Exception) else result

    def mutate(self, x: Genotype) -> Genotype:
        code = self._generate_code(self.extra_prompt + x[0] + self.extra_prompt)[0]
        result = self._evaluate_code(self.extra_prompt + code)
        return code, None if isinstance(result, Exception) else result

    def fitness(self, x: Genotype) -> float:
        if not isinstance(x[1], np.ndarray) or x[1].shape != self.shape:
            return -np.inf
        return -np.abs(x[1] - self.target_img).sum()

    def to_behaviour_space(self, x: Genotype) -> Phenotype:
        height, width = self.shape
        bh = self.block_size[0]
        bw = self.block_size[1]
        ny = math.ceil(height / bh)
        nx = math.ceil(width / bw)
        result = np.zeros((ny, nx, 3))

        # Assume all-zero behaviour space if the program ended up with error?
        if x[1] is None:
            return result.reshape(-1)

        for i in range(ny):
            for j in range(nx):
                for ch in range(3):
                    result[i, j, ch] = np.mean(x[1][i * bh:(i+1) * bh, j * bw: (j+1) * bw, ch])

        return result.reshape(-1)

    def to_string(self, x: Genotype) -> str:
        return x[0]

    def _generate_code(self, seed: str, num=1) -> List[str]:
        """
        Parameters:
            seed: the seed text.
            num: (Optional) batch size.
        Returns:
            a list of code(s) generated by the model.
        """
        encoding = self.tokenizer([seed], truncation=True, padding=True,
                                  max_length=self.config.max_length,
                                  return_tensors='pt')
        self.config.batch_size = num
        self._update_seed()
        with torch.no_grad():
            completion = sample(self.config, self.model, self.tokenizer, encoding)
        truncation = list(map(functools.partial(truncate, print_num=np.inf, only_local_scope=True), completion))

        return truncation

    def _evaluate_code(self, code: str):
        """
        Call the sandbox server to execute the code, and obtain the result.
        Parameters:
            code: the full code string.
        Returns:
            a numpy array (if successful) or the exception object.
        """
        try:
            x = requests.post(f"http://{self.sandbox_server}/eval_func",
                              json={"code": code, "func_name": self.func_name}, timeout=5)
            result = np.array(json.loads(x.text))
        except Exception as e:
            result = e

        return result

    def _update_seed(self):
        """
        Update the random seed in `self.config.seed` using `self.rng`.
        """
        self.config.seed = int(self.rng.integers(0, 1e8))

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
        self.diff_model: DiffModel = diff_model
        self.genotype_ndim = ndim
        self.genotype_space = np.array([[0, max_height], [0, max_width], [0, max_mass]]).T

        self.simulator = simulator.SodaraceSimulator(body=self.seed["sodaracer"])

    def generate_program(self, x: str) -> Genotype:
        # Call LM to generate a new program and run it, returning a dict containing the program string
        # and the dict from running
        return self.diff_model.generate_program(x)

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
