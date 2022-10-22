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
    def random(self, **kwarg) -> List[Genotype]:
        raise NotImplementedError

    @abstractmethod
    def mutate(self, x: Genotype, **kwarg) -> List[Genotype]:
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

    def random(self, **kwarg) -> List[Genotype]:
        return [np.random.uniform(*self.genotype_space)]

    def mutate(self, x: Genotype, **kwarg) -> List[Genotype]:
        x = x.copy()
        ix = np.random.randint(self.genotype_ndim)
        x[ix] = x[ix] + np.random.uniform(-1, 1)
        return [x]

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


def _three_channel_average(x: Genotype) -> Phenotype:
    """
    Assume the input is of shape (height, width, channel), and we average each channel to get (channel,)
    """
    assert isinstance(x, tuple) and len(x) == 2
    # Code with invalid return -> return a `None` Phenotype.
    if not isinstance(x[1], np.ndarray) or len(x[1].shape) != 3 or x[1].shape[2] != 3:
        return None

    return np.average(x[1].reshape((-1, 3)), axis=0)


class ImageOptim(BaseEnvironment):
    """
    This will try to mutate programs that return images. Fitness is simply the absolute difference between the returning
    image and the target image.
    To map into the behaviour space, the image will be divided into blocks (specified in `block_size`), and average
    values of RGB channels in each block will be put together as a point in the behaviour space (basically it is
    average-pooling).
    """
    default_docstring = '\t"""Draw a yellow circle.\n\t"""'
    default_import = 'import math\nimport numpy as np\n'

    # Record different definitions of behaviour spaces in a dict. Feel free to add.
    behaviour_mode_spec = {'3-channel':
                               {'genotype_ndim': 3,
                                'behaviour_space_fn': _three_channel_average}}

    def __init__(self, seed: str, config: Union[str, dict, DictConfig], target_img: np.ndarray, func_name: str,
                 docstring=default_docstring, import_text=default_import, sandbox_server='localhost:5000',
                 behaviour_mode: str = '3-channel'):
        """
        Parameters:
            seed: the seed string.
            config: the config file or dict.
            target_img: the target image.
            func_name: the name of the function to be called to return images.
            docstring: (Optional) the extra docstring attached under the function definition in a prompt.
            import_text: (Optional) the import lines to run the codes.
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
        self.import_text = import_text + '\n'
        # These prompts can probably be improved.
        self.def_and_docstring = f'\ndef {self.func_name}():\n{docstring}\n\tpic = np.zeros({self.shape})\n'
        self.def_for_mutation = f'\ndef {self.func_name}_old():\n\tpic = np.zeros({self.shape})\n'

        self.model, self.tokenizer = model_setup(self.config)

        # Use RNG to rotate random seeds during inference.
        self.rng = np.random.default_rng(seed=self.config.seed)

        self.behaviour_mode = behaviour_mode
        self.genotype_ndim = self.behaviour_mode_spec[self.behaviour_mode]['genotype_ndim']
        self.genotype_space = np.repeat([[0, 255]], self.genotype_ndim, axis=0).T

        self.sandbox_server = sandbox_server

    def random(self, **kwargs) -> List[Genotype]:
        """
        Randomly generate a batch of codes and evaluate their outputs.
        Returns:
            a tuple of the code string and the returning result (None if there is error).
        """
        return self._get_code_result_pair(self.seed + self.def_and_docstring, **kwargs)

    def mutate(self, x: Genotype, **kwargs) -> List[Genotype]:
        """
        Randomly mutate a batch of codes from a given individual and evaluate their outputs.
        Parameters:
            x: the individual to be mutated.
        Returns:
            a tuple of the code string and the returning result (None if there is error).
        """
        return self._get_code_result_pair(self.def_for_mutation + x[0] + self.def_and_docstring, **kwargs)

    def fitness(self, x: Genotype) -> float:
        if not isinstance(x[1], np.ndarray) or x[1].shape != self.shape:
            return -np.inf
        return -np.abs(x[1] - self.target_img).sum()

    def to_behaviour_space(self, x: Genotype) -> Phenotype:
        return self.behaviour_mode_spec[self.behaviour_mode]['behaviour_space_fn'](x)

    def to_string(self, x: Genotype) -> str:
        return str(x[1].reshape((-1, 3)).mean(axis=0).astype(int)) if self._has_valid_output(x) else None

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

    def _evaluate_code(self, code: str) -> Union[np.ndarray, Exception]:
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

    def _get_code_result_pair(self, prompt, batch_size=32) -> List[Tuple[str, np.ndarray]]:
        """
        Parameters:
            prompt: the prompt input.
            batch_size: (Optional) the batch size.
        Returns:
            a list of tuples (code, result).
            `result` is a numpy array if the code returns an array or a list (uniform size).
            `result` is None if otherwise.
        """
        codes = self._generate_code(prompt, num=batch_size)
        results = []
        for i in range(len(codes)):
            result = self._evaluate_code(self.import_text + self.def_and_docstring + codes[i])
            if isinstance(result, np.ndarray):
                results.append((codes[i], result))
            else:
                results.append((codes[i], None))
        # If needed, the result can be further classified based on the error type.
        return results

    @staticmethod
    def _has_valid_output(x: Genotype) -> bool:
        return isinstance(x[1], np.ndarray) and len(x[1].shape) == 3 and x[1].shape[2] == 3

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

    def random(self, **kwarg) -> List[Genotype]:
        return [np.random.uniform(*self.genotype_space)]

    def mutate(self, x: Genotype, **kwarg) -> List[Genotype]:
        x = x.copy()
        ix = np.random.randint(self.genotype_ndim)
        x[ix] = x[ix] + np.random.uniform(-5, 5)
        return [x]

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

    def random(self, **kwarg) -> List[Genotype]:
        program_dict = self.generate_program(self.seed["program_str"])
        # TODO: consider storing morphology dict inside genotype?
        self.simulator = simulator.SodaraceSimulator(body=program_dict["sodaracer"])
        return [program_dict]

    def mutate(self, x: Genotype, **kwarg) -> List[Genotype]:
        # TODO: maybe create proper Genotype class.
        program_dict = self.generate_program(x["program_str"])
        self.simulator = simulator.SodaraceSimulator(body=program_dict["sodaracer"])
        return [program_dict]

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
