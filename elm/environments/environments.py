import functools
import json
import math
import string
from abc import ABC, abstractmethod
from typing import Generic, Optional, TypeVar, Union

import numpy as np
import requests
import torch
from omegaconf import DictConfig, OmegaConf

from elm.codegen.codegen_utilities import model_setup, sample, truncate
from elm.environments.sodaracer import SodaraceSimulator

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
    def to_behavior_space(self, x: GenoType) -> Phenotype:
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
    def __init__(self, input_str: str, result: Optional[np.ndarray]):
        self.input_str = input_str
        self.result = result
        self.valid = self.validate()

    def __str__(self) -> str:
        if self.valid:
            return str(self.result.reshape((-1, 3)).mean(axis=0).astype(int))
        else:
            return ""

    def validate(self) -> bool:
        return (
            isinstance(self.result, np.ndarray)
            and len(self.result.shape) == 3
            and self.result.shape[2] == 3
        )

    def _three_channel_average(self) -> Phenotype:
        """
        Assume the input is of shape (height, width, channel), and we average each channel to get (channel,)
        """
        # Code with invalid return -> return a `None` Phenotype.
        return np.average(self.result.reshape((-1, 3)), axis=0) if self.valid else None


class ImageOptim(BaseEnvironment[ImageGeneration]):
    """
    This will try to mutate programs that return images. Fitness is simply the absolute difference between the returning
    image and the target image.
    To map into the behavior space, the image will be divided into blocks (specified in `block_size`), and average
    values of RGB channels in each block will be put together as a point in the behavior space (basically it is
    average-pooling).
    """

    default_docstring = '\t"""Draw a yellow circle.\n\t"""'
    default_import = "import math\nimport numpy as np\n"

    # Record different definitions of behavior spaces in a dict. Feel free to add.
    behavior_mode_spec = {"3-channel": {"genotype_ndim": 3}}

    def __init__(
        self,
        seed: str,
        config: Union[str, dict, DictConfig],
        target_img: np.ndarray,
        func_name: str,
        docstring=default_docstring,
        import_text=default_import,
        sandbox_server="localhost:5000",
        behavior_mode: str = "3-channel",
    ):
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
        # TODO: test config loading types
        if isinstance(config, str):
            self.config = OmegaConf.load(config)
        elif isinstance(config, (dict, DictConfig)):
            self.config = DictConfig(config)
        else:
            raise ValueError

        self.target_img = target_img
        self.shape = target_img.shape
        self.func_name = func_name
        self.import_text = import_text + "\n"
        # These prompts can probably be improved.
        self.def_and_docstring = (
            f"\ndef {self.func_name}():\n{docstring}\n\tpic = np.zeros({self.shape})\n"
        )
        self.def_for_mutation = (
            f"\ndef {self.func_name}_old():\n\tpic = np.zeros({self.shape})\n"
        )

        self.model, self.tokenizer = model_setup(self.config)

        # Use RNG to rotate random seeds during inference.
        self.rng = np.random.default_rng(seed=self.config.seed)

        self.behavior_mode = behavior_mode
        self.genotype_ndim: int = self.behavior_mode_spec[self.behavior_mode][
            "genotype_ndim"
        ]
        self.genotype_space = np.repeat([[0, 255]], self.genotype_ndim, axis=0).T

        self.sandbox_server = sandbox_server

    def random(self, **kwargs) -> list[ImageGeneration]:
        """
        Randomly generate a batch of codes and evaluate their outputs.
        Returns:
            a tuple of the code string and the returning result (None if there is error).
        """
        return self._get_code_result_pair(self.seed + self.def_and_docstring, **kwargs)

    def mutate(self, x: ImageGeneration, **kwargs) -> list[ImageGeneration]:
        """
        Randomly mutate a batch of codes from a given individual and evaluate their outputs.
        Parameters:
            x: the individual to be mutated.
        Returns:
            a tuple of the code string and the returning result (None if there is error).
        """
        return self._get_code_result_pair(
            self.def_for_mutation + x.input_str + self.def_and_docstring, **kwargs
        )

    def fitness(self, x: ImageGeneration) -> float:
        if not x.valid or x.result.shape != self.shape:
            return -np.inf
        return -np.abs(x.result - self.target_img).sum()

    def to_behavior_space(self, x: ImageGeneration) -> Phenotype:
        if self.behavior_mode == "3-channel":
            return x._three_channel_average()
        return None

    def _generate_code(self, seed: str, num=1) -> list[str]:
        """
        Parameters:
            seed: the seed text.
            num: (Optional) batch size.
        Returns:
            a list of code(s) generated by the model.
        """
        encoding = self.tokenizer(
            [seed],
            truncation=True,
            padding=True,
            max_length=self.config.max_length,
            return_tensors="pt",
        )
        self.config.batch_size = num
        self._update_seed()
        with torch.no_grad():
            completion = sample(self.config, self.model, self.tokenizer, encoding)
        truncation = list(
            map(
                functools.partial(truncate, print_num=np.inf, only_local_scope=True),
                completion,
            )
        )

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
            x = requests.post(
                f"http://{self.sandbox_server}/eval_imageoptim_func",
                json={"code": code, "func_name": self.func_name},
                timeout=5,
            )
            result = np.array(json.loads(x.text))
        except Exception as e:
            result = e

        return result

    def _get_code_result_pair(self, prompt, batch_size=32) -> list[ImageGeneration]:
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
            result = self._evaluate_code(
                self.import_text + self.def_and_docstring + codes[i]
            )
            if isinstance(result, np.ndarray):
                results.append(ImageGeneration(codes[i], result))
            else:
                results.append(ImageGeneration(codes[i], None))
        # If needed, the result can be further classified based on the error type.
        return results

    def _update_seed(self):
        """
        Update the random seed in `self.config.seed` using `self.rng`.
        """
        self.config.seed = int(self.rng.integers(0, 1e8))


class StringArrayGenotype(ArrayGenotype):
    def __str__(self) -> str:
        x: np.ndarray = np.round(self)
        return "".join(
            string.ascii_letters[ix]
            for ix in np.clip(x.astype(int), 0, len(string.ascii_letters) - 1)
        )


# find a string by mutating one character at a time
class MatchString(BaseEnvironment[StringArrayGenotype]):
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
    def __init__(self, program_str: str, result_dict: dict):
        self.program_str = program_str
        self.result_dict = result_dict
        self.simulator = SodaraceSimulator(body=self.result_dict)
        self.morphology = self.simulator.morphology

    def evaluate(self, timesteps: int) -> float:
        return self.simulator.evaluate(timesteps)

    def __str__(self) -> str:
        return self.program_str[:10]


class Sodarace(BaseEnvironment[Sodaracer]):
    def __init__(
        self,
        seed: dict,
        diff_model,
        eval_steps: int,
        max_height: int = 1000,
        max_width: int = 1000,
        max_mass: int = 2000,
        ndim: int = 3,
    ) -> None:
        self.seed = Sodaracer(**seed)
        self.diff_model = diff_model
        self.eval_steps = eval_steps
        self.genotype_ndim = ndim
        self.genotype_space = np.array(
            [[0, max_height], [0, max_width], [0, max_mass]]
        ).T

    def generate_program(self, x: str) -> Sodaracer:
        # Call LM to generate a new program and run it, returning a dict containing the program string
        # and the dict from running it.
        return Sodaracer(**self.diff_model.generate_program(x))

    def fitness(self, x: Sodaracer) -> float:
        # Call Sodaracers environment to get the fitness.
        return x.evaluate(self.eval_steps)

    def random(self, **kwarg) -> list[Sodaracer]:
        new_sodaracer = self.generate_program(self.seed.program_str)
        return [new_sodaracer]

    def mutate(self, x: Sodaracer, **kwarg) -> list[Sodaracer]:
        new_sodaracer = self.generate_program(x.program_str)
        return [new_sodaracer]

    def to_behavior_space(self, x: Sodaracer) -> Phenotype:
        # Map from floats of h,w,m to behavior space grid cells.
        return np.array(
            [x.morphology["height"], x.morphology["width"], x.morphology["mass"]]
        ).astype(int)
