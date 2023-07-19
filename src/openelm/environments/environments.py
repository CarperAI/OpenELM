import json
import math
import string
import sys
import warnings
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Generic, Optional, Type, TypeVar, Union, Any
from collections import deque

import random
import warnings
import re

import numpy as np

import requests
import torch
from transformers import pipeline

from langchain import PromptTemplate
from langchain.chains import LLMChain
from transformers import pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from openai.embeddings_utils import get_embedding, cosine_similarity

from openelm.configs import (
    EnvConfig,
    ImageEnvConfig,
    P3ProblemEnvConfig,
    P3ProbSolEnvConfig,
    PromptEnvConfig,
    SodaraceEnvConfig,
    StringEnvConfig,
    LMXGenerationEnvConfig,
)
from openelm.environments.env_utils import (
    NULL_SEED,
    AnimalPromptTask,
    AntonymPromptTask,
    COTPromptTask,
    ToyPromptTask,
    get_image_target,
    AIFeedback,
    cosine_similarity,
)
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
from openelm.environments.p3 import (
    P3_PROBLEM_MED_SEED,
    P3_PROBLEM_LONG_SEED,
    P3_PROBSOL_MED_SEED,
    P3_PROBSOL_LONG_SEED,
    P3_IMPORTS,
)
from openelm.mutation_model import MutationModel
from openelm.utils.code_eval import pool_exec_processes, type_check, pass_at_k
from openelm.sandbox.server.sandbox_codex_execute import ExecResult

from aleph_alpha_client import (
    Client,
    Prompt,
    EvaluationRequest,
    SemanticEmbeddingRequest,
    SemanticRepresentation,
)

sys.set_int_max_str_digits(0)  # remove length limitation for int->str conversion
# (model sometimes outputs really long ints)

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


def get_positive_score(sentiment, mode="distilbert"):
    """Get the positive score from a sentiment analysis result."""
    if mode == "distilbert":
        return next(
            result["score"] for result in sentiment if result["label"] == "POSITIVE"
        )
    elif mode == "roberta":
        return next(
            result["score"] for result in sentiment if result["label"] == "LABEL_2"
        )
    else:
        raise ValueError(f"Invalid mode: {mode}")


def get_negative_score(sentiment, mode="distilbert"):
    """Get the negative score from a sentiment analysis result."""
    if mode == "distilbert":
        return next(
            result["score"] for result in sentiment if result["label"] == "NEGATIVE"
        )
    elif mode == "roberta":
        return next(
            result["score"] for result in sentiment if result["label"] == "LABEL_0"
        )
    else:
        raise ValueError(f"Invalid mode: {mode}")


def get_sentiment_score(sentiment, mode="distilbert"):
    return get_positive_score(sentiment, mode) - get_negative_score(sentiment, mode)


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


class PromptGenotype(Genotype):
    """
    Genotype wrapper for a LangChain template.

    This consists of a base format for all individuals, as well as individual-specific fields which will be evolved.
    Remaining fields will be filled in at evaluation time.

    Args:
        prompt (PromptTemplate): The base template for all individuals.
        fixed_inputs (dict[str, str], optional): Individual-specific fields to fill in. Defaults to None.
    """

    def __init__(
        self,
        prompt: PromptTemplate,
        fixed_inputs: Optional[dict[str, str]] = None,
        behavior_model=None,
    ):
        self.fixed_inputs = fixed_inputs
        if fixed_inputs:
            self.prompt = prompt.partial(**fixed_inputs)
        else:
            self.prompt = prompt
        self.result_obj = None
        if behavior_model:
            # assume sentiment analysis; can expand this later
            sentiment = behavior_model(self.__str__())
            self.behavior = (
                len(self.fixed_inputs["instruction_str"]),
                get_sentiment_score(
                    sentiment[0], mode=behavior_model.model.config.model_type
                ),
            )
        else:
            self.behavior = (len(self.fixed_inputs["instruction_str"]),)

    def __str__(self) -> str:
        return self.fixed_inputs["instruction_str"]
        # return self.prompt.template

    def format(self, **kwargs) -> str:
        return self.prompt.format(**kwargs)

    def evaluate(self, model, inputs):
        chain = LLMChain(llm=model.model, prompt=self.prompt)
        self.result_obj = {
            "prompt": self.format(**inputs),
            "output": chain(inputs),
        }
        return self.result_obj["output"]

    def to_phenotype(self):
        return self.behavior


class PromptEvolution(BaseEnvironment[PromptGenotype]):
    """Evolves a LangChain prompt."""

    def __init__(
        self,
        config: PromptEnvConfig,
        mutation_model: MutationModel,
        fitness_model=None,
    ):
        self.config: PromptEnvConfig = config
        self.batch_size = self.config.batch_size
        self.mutation_model = mutation_model
        if fitness_model is None:
            self.fitness_model = mutation_model
        self.behavior_model = pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment",
            # model="distilbert-base-uncased-finetuned-sst-2-english",
            top_k=None,
            # return_all_scores=True,
        )

        self.task_name = self.config.task_name
        if self.task_name == "toy":
            self.genotype_ndim = 1
            self.genotype_space = np.array([[0], [250]])
            self.task = ToyPromptTask()
        elif self.task_name == "antonym":
            self.genotype_ndim = 2
            self.genotype_space = np.array([[0, -1], [250, 1]])
            self.task = AntonymPromptTask()
        elif self.task_name == "animal":
            self.genotype_ndim = 2
            self.genotype_space = np.array([[0, -1], [250, 1]])
            self.task = AnimalPromptTask()
        elif self.task_name == "cot":
            self.genotype_ndim = 2
            self.genotype_space = np.array([[0, -1], [250, 1]])
            self.task = COTPromptTask()
        else:
            raise ValueError(f"Unknown task: {self.task_name}")

        self.base_prompt = PromptTemplate(
            template=self.task.base_template, input_variables=self.task.input_variables
        )
        self.rng = np.random.default_rng(self.config.seed)

    def get_rng_state(self) -> Optional[np.random._generator.Generator]:
        return self.rng

    def set_rng_state(self, rng_state: Optional[np.random._generator.Generator]):
        self.rng = rng_state

    def random(self) -> list[PromptGenotype]:
        return [self.random_prompt() for _ in range(self.batch_size)]

    def random_prompt(self):
        if self.task_name == "toy":
            inputs = {
                "n_repetitions": str(self.rng.integers(10)),
                "instruction_str": self.task.instruction_str,
                "few_shot_examples": self.task.create_few_shot_examples(
                    self.task.instruction_str
                ),
            }
        elif (
            self.task_name == "antonym"
            or self.task_name == "animal"
            or self.task_name == "cot"
        ):
            few_shot_examples = self.task.create_few_shot_examples(
                n_examples=10,
            )
            generation_prompt = PromptTemplate(
                input_variables=["few_shot_examples"],
                template=self.task.generation_instruction,
            )
            generation_chain = LLMChain(
                llm=self.fitness_model.model, prompt=generation_prompt
            )
            result = generation_chain({"few_shot_examples": few_shot_examples})
            new_instruction_str = result["text"]

            # take only the first sentence
            new_instruction_str = (
                new_instruction_str.replace('"', "")
                .lstrip("0123456789. \n")
                .split(".")[0]
                + "."
            )

            inputs = {
                "instruction_str": new_instruction_str,
            }

        return PromptGenotype(
            prompt=self.base_prompt,
            fixed_inputs=inputs,
            behavior_model=self.behavior_model,
        )

    def mutate(self, genomes: list[PromptGenotype]) -> list[PromptGenotype]:
        prompts = [self.mutate_prompt(prompt) for prompt in genomes]
        return prompts

    def mutate_prompt(self, prompt):
        if self.task_name == "toy":
            # mutate the instruction string; note that we also need to change the few shot examples to match
            old_instruction_str = prompt.fixed_inputs["instruction_str"]
            result = self.rewrite_string(
                input_str=old_instruction_str,
                rewrite_instruction=np.random.choice(self.task.mutation_instructions),
                variable_name="instruction_str",
            )
            new_instruction_str = (
                result["text"].strip().split()[0]
            )  # take the first word

            inputs = {
                "n_repetitions": str(np.random.randint(10)),
                "instruction_str": new_instruction_str,
                "few_shot_examples": self.task.create_few_shot_examples(
                    new_instruction_str
                ),
            }
        elif (
            self.task_name == "antonym"
            or self.task_name == "animal"
            or self.task_name == "cot"
        ):
            if np.random.random() > 0.3:
                # rewrite the instruction string
                old_instruction_str = prompt.fixed_inputs["instruction_str"]
                result = self.rewrite_string(
                    input_str=old_instruction_str,
                    rewrite_instruction=np.random.choice(
                        self.task.mutation_instructions
                    ),
                    variable_name="instruction_str",
                )
                new_instruction_str = (
                    result["text"]
                    .replace('"', "")
                    .lstrip("0123456789. \n")
                    .split(".")[0]
                    + "."
                )  # take the first sentence
                inputs = {
                    "instruction_str": new_instruction_str,
                }
            else:
                # otherwise, just generate a random prompt
                return self.random_prompt()

        return PromptGenotype(
            prompt=self.base_prompt,
            fixed_inputs=inputs,
            behavior_model=self.behavior_model,
        )

    def rewrite_string(self, input_str, rewrite_instruction, variable_name):
        """
        Prompts an LLM to rewrite a string.

        Args:
            input_str: The string to rewrite.
            rewrite_instruction: String prompt template for the LLM
            variable_name: The name of the variable in the template to replace with input_str
        """
        rewrite_prompt = PromptTemplate(
            input_variables=[variable_name],
            template=rewrite_instruction,
        )
        rewrite_chain = LLMChain(llm=self.mutation_model.model, prompt=rewrite_prompt)
        result = rewrite_chain({variable_name: input_str})
        # if self.config.debug:
        #     print(
        #         f"-- Rewrite Instruction --\n{rewrite_instruction}\n-- Input --\n{input_str}\n-- Output --\n{result['text']}\n"
        #     )
        return result

    def fitness(self, x: PromptGenotype) -> float:
        if self.task_name == "toy":
            inputs = {
                "target": self.task.target,
            }
            result = x.evaluate(model=self.fitness_model, inputs=inputs)

            # fitness is number of times it generated the target word in a row
            count = 0
            for word in result["text"].strip().split():
                if word.lower() == self.task.target:
                    count += 1
                else:
                    break

            fitness = count
            if self.config.debug:
                print(
                    f"-- Prompt --\n{x.result_obj['prompt']}\n-- Fitness: {fitness} --\n-- Behavior: {x.to_phenotype()} --\n"
                )
        elif (
            self.task_name == "antonym"
            or self.task_name == "animal"
            or self.task_name == "cot"
        ):
            fitnesses = []
            eval_template = PromptTemplate(
                input_variables=["instruction_str", "input_str", "output_str"],
                template=self.task.evaluation_instruction,
            )
            inputs, outputs = self.task.get_random_data(
                n_examples=self.config.evals_per_prompt
            )
            for input_str, output_str in zip(inputs, outputs):
                fitnesses.append(
                    self.evaluate_template(
                        eval_template,
                        x.fixed_inputs["instruction_str"],
                        input_str,
                        output_str,
                    )
                )
            fitness = np.mean(fitnesses)
            if self.config.debug:
                print(
                    f"-- instruction_str --\n{x.fixed_inputs['instruction_str']}\n-- Fitness: {fitness} --\n-- Behavior: {x.to_phenotype()} --\n"
                )
        elif self.task_name == "imagegen":
            # fitness_prompt = PromptTemplate(
            #     input_variables=["program_str", "instruction_str"],
            #     template=self.task.fitness_template,
            # )
            pass

        return fitness

    def evaluate_template(self, eval_template, instruction_str, input_str, output_str):
        """
        Evaluates a template on the log likelihood of the output_str, given the instruction_str and input_str.

        Args:
            eval_template: The template to evaluate.
            instruction_str: The instruction string.
            input_str: The input string.
            output_str: The output string.

        Returns:
            The log likelihood of the tokens in the output string, given the instruction and input strings.
        """
        model = self.fitness_model.model.model
        tokenizer = self.fitness_model.model.tokenizer

        partial_template = eval_template.partial(instruction_str=instruction_str)
        filled_prompt = partial_template.format(
            input_str=input_str, output_str=output_str
        )
        # hack; replace the output string to figure out which token numbers correspond to the output (see APE)
        reference_prompt = partial_template.format(input_str=input_str, output_str="~")

        tokens_filled = tokenizer.encode(filled_prompt, return_tensors="pt")
        tokens_reference = tokenizer.encode(reference_prompt, return_tensors="pt")

        # We label only the tokens of interest, and mask otherwise (set to -100)
        # This assumes there's only one section in the middle that we're interested in
        # forward alignment; mask duplicate tokens starting from beginning
        labels = tokens_filled.clone()
        for i, (t1, t2) in enumerate(zip(tokens_filled[0], tokens_reference[0])):
            if t1 == t2:
                labels[0, i] = -100 * torch.ones_like(labels[0, i])
            else:
                break

        # backward alignment
        for i, (t1, t2) in enumerate(
            zip(torch.flip(tokens_filled[0], [0]), torch.flip(tokens_reference[0], [0]))
        ):
            if t1 == t2:
                labels[0, -i - 1] = -100 * torch.ones_like(
                    labels[0, -i - 1]
                )  # adjust index for reversed
            else:
                break

        outputs = model(tokens_filled.cuda(), labels=labels.cuda())

        # self.print_labels(tokens_filled, tokens_reference, labels, tokenizer)
        return -outputs.loss.item()

    def print_labels(self, tokens_filled, tokens_reference, labels, tokenizer):
        from itertools import zip_longest

        print(
            f"{'Label':<10}{'Token Filled':<20}{'Token ID':<10}{'Token Reference':<20}{'Token ID':<10}"
        )

        for tf, tr, label in zip_longest(
            tokens_filled[0], tokens_reference[0], labels[0]
        ):
            decoded_tf, decoded_tr = " ", " "
            if tf is not None:
                decoded_tf = tokenizer.decode(
                    [tf]
                )  # Wrap tf in a list because .decode() expects a list
            if tr is not None:
                decoded_tr = tokenizer.decode([tr])  # Same for tr
            if label is None:
                label = ""
            if tr is None:
                tr = ""
            if tf is None:
                tf = ""
            print(f"{label:<10}{decoded_tf:<20}{tf:<10}{decoded_tr:<20}{tr:<10}")


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
        #     print(self.valid)
        #     self.simulator = SodaraceSimulator(body=self.result_obj)
        #     print(self.evaluate(0))
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
        config: SodaraceEnvConfig,
        mutation_model: MutationModel,
    ) -> None:
        """
        Sodarace environment.

        Args:
            config: the environment config.
            mutation_model: the mutation model.
        """
        self.config: SodaraceEnvConfig = config
        self.batch_size = self.config.batch_size
        self.mutation_model: MutationModel = mutation_model

        self.genotype_space = np.array(self.config.behavior_space).T
        self.genotype_ndim = self.genotype_space.shape[1]

        self.seed_strs: list[str] = self.config.starting_seeds
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
        """
        Constructs a prompt for generating Sodaracers.

        Args:
            code_batch (Optional[Union[list[str], str]], optional): A
            list of program strings or a single program string. Defaults to None.

        Returns:
            dict[str, str]: A dictionary containing two keys: "prompt" and
            "template". The "prompt" key maps to a string containing the
            full prompt for generating a Sodaracer program. The "template"
            key maps to a string containing the required imports and
            instruction for generating a Sodaracer program.

        The method constructs a prompt for generating Sodaracer programs
        based on the seeds and configuration settings specified in self.seed_strs
        and self.config.
        """
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
                    # TODO: get nearby genotypes
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
        """
        Generate new programs with a mutation model and evaluate them.

        Args:
            code_batch (list[dict[str, str]): a list of program strings.

        Returns:
            list[Sodaracer]: A list of Sodaracer objects.
        """
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
        """
        Generates a batch of Sodaracer programs with the specified batch size.

        Returns a list of new Sodaracer programs.

        Returns:
            list[Sodaracer]: A list of random Sodaracer programs.
        """
        program_list = [self.construct_prompt() for _ in range(self.config.batch_size)]
        new_sodaracers = self.generate_programs(program_list)
        return new_sodaracers

    def mutate(self, sodaracer_list: list[Sodaracer]) -> list[Sodaracer]:
        """
        Mutates a list of Sodaracer programs.

        Given a list of Sodaracer programs, constructs a prompt for each program,
        generate a list of new programs by mutating the prompts, and returns a
        list of new Sodaracer programs.

        Args:
            sodaracer_list (list[Sodaracer]): A list of Sodaracer programs to be mutated.

        Returns:
            list[Sodaracer]: A list of new Sodaracer programs generated by mutating the prompts.
        """
        sodaracers = [sr.program_str for sr in sodaracer_list]
        program_list = list(map(self.construct_prompt, sodaracers))
        new_sodaracers = self.generate_programs(program_list)
        return new_sodaracers

    def fitness(self, x: Sodaracer) -> float:
        """
        Evaluates the fitness of a Sodaracer program.

        Args:
            x (Sodaracer): A Sodaracer to evaluate.

        Returns:
            float: fitness of the Sodaracer.

        The method first checks whether the Sodaracer program is valid or not using
        the `.evaluate()` method of the Sodaracer. If the program is invalid,
        the method returns -np.inf to indicate that the program is not fit.
        """
        if x.valid:
            return x.evaluate(self.config.eval_ms)
        else:
            return -np.inf


class P3Solution(Genotype):
    def __init__(self, program_str: str, result_obj: dict, config: P3ProblemEnvConfig):
        """
        Genotype for a programming puzzle solution.
        Args:
            program_str: the solution program string (the g6() function).
            result_obj: dict.
            config: environment config
        """
        self.program_str = program_str
        self.result_obj = result_obj
        self.config = config

        # When comparing for phenotype, just use import statement and new solution function
        baseline = '''from typing import List

def g1():
    """Find a string that when concatenated onto 'Hello ' gives 'Hello world'."""
    return "world")'''
        self.baseline_emb = np.array(
            get_embedding(baseline, engine=self.config.embedding_model_path)
        )

        if self.config.embedding_model_type == "hf":
            self.pl = pipeline(
                "feature-extraction", model=self.config.embedding_model_path
            )
            seed_features = np.array(self.pl(baseline))
            self.scaler = StandardScaler()
            seed_features_scaled = self.scaler.fit_transform(np.squeeze(seed_features))
            self.pca = PCA(0.95)
            self.pca.fit(seed_features_scaled)

    def to_phenotype(self) -> Optional[Phenotype]:
        if self.config.embedding_model_type == "openai":
            compare_str = self.program_str
            i_assert = compare_str.find("assert")
            if i_assert > -1:
                compare_str = compare_str[:i_assert]
            emb = np.array(
                get_embedding(compare_str, engine=self.config.embedding_model_path)
            )
            return cosine_similarity(emb, self.baseline_emb)
        elif self.config.embedding_model_type == "hf":
            features = np.array(self.pl(self.program_str))
            features_scaled = self.scaler.transform(np.squeeze(features))
            pca_features = self.pca.transform(features_scaled)
            return pca_features.max(axis=0).flatten()

    def __str__(self) -> str:
        return self.program_str

    def __getstate__(self):
        state = self.__dict__.copy()
        if "pl" in state:
            del state["pl"]
        if "scaler" in state:
            del state["scaler"]
        if "pca" in state:
            del state["pca"]
        return state


class P3Problem(BaseEnvironment[P3Solution]):
    def __init__(
        self,
        config: P3ProblemEnvConfig,
        mutation_model: MutationModel,
        problem_str: str = None,
        solution_preamble: str = None,
    ) -> None:
        """
        The objective is to generate solutions to a given programming puzzle problem.
        Args:
            seed: the seed dict.
            config: the config file path or dict.
            mutation_model: the diff model (or alternatives).
            problem_str: an optional puzzle problem
            solution_preamble: accompanies optional problem_str
        """
        self.mutation_model = mutation_model
        self.config = config
        self.batch_size = self.config.batch_size
        self.seed_index = self.config.starting_seed
        self.rng = None

        if self.config.prompt_size == "long":
            self.prompt_seed = P3_PROBLEM_LONG_SEED
        elif self.config.prompt_size == "med":
            self.prompt_seed = P3_PROBLEM_MED_SEED
        else:
            raise ValueError("No seed string found")

        # Get info for the puzzle that will be solved
        if problem_str is None:
            # This puzzle is at the index of the puzzles array specified by self.seed_index
            puzzles = requests.get(
                "https://raw.githubusercontent.com/microsoft/PythonProgrammingPuzzles/v0.2/puzzles/puzzles.json"
            ).json()
            puzzle = puzzles[self.seed_index]

            self.problem_func = puzzle["sat"].replace(
                "def sat(", "def f6("
            )  # prompt form is f6()
            self.solution_preamble = puzzle["sol_header"].replace(
                "def sol(", "def g6("
            )  # solution form is g6()
            if self.config.prompt_size == "long":
                self.solution_preamble += (
                    "\n" + puzzle["sol_docstring"]
                )  # add in the docstring
            self.ans_type = puzzle["ans_type"]
        else:
            self.problem_func = problem_str
            self.solution_preamble = solution_preamble
            # TODO: generate a docstring?
            self.ans_type = None

        # Use the first example in the prompt seed as basis for embedding sizes
        i_first = self.prompt_seed.find("assert")
        first_example = self.prompt_seed[:i_first].strip()

        if self.config.embedding_model_type == "openai":
            self.genotype_ndim: int = 1
            self.genotype_space = np.repeat([[0, 1]], self.genotype_ndim, axis=0).T
        elif self.config.embedding_model_type == "hf":
            # Dummy to get behavior space shape
            dummy_pl = pipeline(
                "feature-extraction", model=self.config.embedding_model_path
            )
            dummy_scaler = StandardScaler()
            dummy_features = np.array(dummy_pl(first_example))
            dummy_features_scaled = dummy_scaler.fit_transform(
                np.squeeze(dummy_features)
            )
            dummy_pca = PCA(0.95)
            dummy_pca_features = dummy_pca.fit_transform(
                np.squeeze(dummy_features_scaled)
            )
            self.genotype_ndim: int = dummy_pca_features.shape[-1]
            self.genotype_space = np.repeat([[-20, 20]], self.genotype_ndim, axis=0).T

    def get_rng_state(self) -> Optional[np.random._generator.Generator]:
        warnings.warn("WARNING: rng state not used in this environment")
        return None

    def set_rng_state(self, rng_state: Optional[np.random._generator.Generator]):
        warnings.warn("WARNING: rng state not used in this environment")
        pass

    def construct_prompt(
        self, code_batch: Optional[Union[list[str], str]] = None
    ) -> dict[str, str]:
        prompt_str = self.prompt_seed

        prompt_str += f"\n\n{self.problem_func}"  # add this particular problem, f6(), to the prompt
        if code_batch is None:
            prompt_str += "\n"
        else:
            prompt_str += (
                f"\n\n# Old version of g6()" f"\n# TODO: fix bugs in the code below\n"
            )
            if isinstance(code_batch, list):
                # TODO: get nearby genotypes
                prompt_str += code_batch[0]
            elif isinstance(code_batch, str):
                prompt_str += code_batch

            prompt_str += f"\n\n# Fixed version of g6()"

        prompt_str += f"\n{self.solution_preamble}"

        template = f"{P3_IMPORTS}\n{self.solution_preamble}"
        return {"prompt": prompt_str, "template": template}

    def generate_programs(self, code_batch: list[str]) -> list[P3Solution]:
        """Generate new programs with a mutation model and evaluate them."""
        local_scope_exec = True
        generated_programs = self.mutation_model.generate_programs(
            code_batch, local_scope_exec, do_trunc=False
        )

        for i, gp in enumerate(generated_programs):
            i_assert = gp.find("assert")
            generated_programs[i] = gp[:i_assert].strip()

        if self.config.sandbox:
            results = []
            for code in generated_programs:
                resp = requests.post(
                    f"{self.sandbox_server}/eval_p3_solution",
                    json={"code": code, "timeout": self.config.timeout},
                    timeout=self.config.timeout,
                )
                if resp.status_code == 200:
                    return_dict = json.loads(resp.text)
                    results.append(return_dict)
        else:
            # TODO: handle (probably inside of pool_exec_processes) all cases where the generated code returns
            # a generator type. The multithreaded execution pickles things and generators can't be pickled
            # which causes the whole thing to error out.
            # For now, try/except and re-try.
            try:
                results = pool_exec_processes(
                    generated_programs,
                    func_name="g6",
                    timeout=self.config.timeout,
                    processes=self.config.processes,
                    debug=self.config.debug,
                )
            except Exception as e:
                return self.generate_programs(code_batch)

        results = [
            {"program_str": gen_prog, "result_obj": res_obj, "config": self.config}
            for (gen_prog, res_obj) in zip(generated_programs, results)
        ]
        return [P3Solution(**p) for p in results]

    def evaluate_solution(self, sol: P3Solution) -> bool:
        """
        Returns whether or not the solution solves this problem
        """
        if self.ans_type is not None:
            return type_check(self.ans_type, sol.result_obj)

        eval_code = (
            f"{P3_IMPORTS}\n"
            f"{self.problem_func}\n"
            f"def run_eval():\n"
            f"    return f6({sol.result_obj})"
        )

        result = pool_exec_processes(
            eval_code,
            func_name="run_eval",
            timeout=self.config.timeout,
            processes=self.config.processes,
            debug=self.config.debug,
        )

        return result[0]

    def fitness(self, sol: P3Solution) -> float:
        """
        If passing the solution to the problem returns True, fitness is 1.0
            else -np.inf
        """
        result = self.evaluate_solution(sol)

        if result == True:
            return 1.0
        else:
            return -np.inf

    def random(self) -> list[P3Solution]:
        program_list = [self.construct_prompt() for _ in range(self.config.batch_size)]
        new_solutions = self.generate_programs(program_list)
        return new_solutions

    def mutate(self, sol_list: list[P3Solution]) -> list[P3Solution]:
        sols = [s.program_str for s in sol_list]
        program_list = list(map(self.construct_prompt, sols))
        new_sols = self.generate_programs(program_list)
        return new_sols


class LMXGeneration(Genotype):
    def __init__(
        self,
        prompt_metadata: dict[str, Any],
        generated_completion: str,
        behavior_measure: bool,
        classifier_model: str,
        ai_feedback_entries: dict,
        aa_client: str,
    ):
        """
        Genotype for a generated completion, with parent few-shot prompt.

        Args:
            prompt_metadata: the prompt, and items of few-shot prompt
            generated_completion: generated completion
        """
        self.prompt_metadata = prompt_metadata
        self.generated_completion = generated_completion
        self.behavior_measure = behavior_measure
        self.ai_feedback_entries = ai_feedback_entries
        self.classifier_model = classifier_model
        self.aa_client = aa_client

    def __str__(self) -> str:
        return self.generated_completion

    def to_phenotype(self) -> Phenotype:
        assert (
            self.generated_completion != "" and self.generated_completion is not None
        ), "generated completion can not be None or an empty string"
        assert (
            self.ai_feedback_entries is not None
        ), "ai_feedback_entries must be defined when using ai_feedback"

        feedback_scores_list = []
        for feedback_type in self.ai_feedback_entries:
            answer_space = self.ai_feedback_entries[feedback_type]["answer_space"]
            feedback_prompt_template = self.ai_feedback_entries[feedback_type][
                "feedback_prompt_template"
            ]
            ai_feedback_evaluator = AIFeedback(
                classifier_model=self.classifier_model,
                label_options=answer_space,
                feedback_template=feedback_prompt_template,
                aa_client=self.aa_client,
            )
            feedback_scores = ai_feedback_evaluator.evaluate(
                {"genotype": self.generated_completion}
            )
            feedback_scores_list.append(feedback_scores[answer_space[0]])

        return np.array(feedback_scores_list)


class LMXGenerationEnvironment(BaseEnvironment[LMXGeneration]):
    def __init__(self, config, mutation_model, num_fewshot=3):
        """
        Language Model Crossover Environment.
        Q: Quality metric (e.g asymmetric search with a reference sentence)
        D: Diversity metric (e.g. Sentiment classifier score)
        Steps:
            1. Initialize seed with seed generations, and generate some random completions
            2. Mutation operator: re-ordering of few-shot prompt and temperature in genotype,
            3. Compute Q and D

        Args:
            config: the config file path or dict.
            mutation_model: Language Model API.
            num_fewshot: number of few shot prompts that for the meta prompt
        """

        self.config: LMXGenerationEnvConfig = config
        self.genotype_ndim = len(self.config.ai_feedback_entries.keys())
        self.genotype_space = np.repeat([[0, 1]], self.genotype_ndim, axis=0).T
        self.mutation_model = mutation_model
        self.num_fewshot = num_fewshot
        self.solution_init_method = self.config.solution_init_method
        self.init_prompt_template = self.config.few_shot_template
        self.batch_size = self.config.batch_size
        self.max_prompt_pool_size = self.config.max_prompt_pool_size
        self.init_size_prompt_pool = self.config.init_size_prompt_pool
        self.latest_genomes = None
        self.num_generations = 0
        self.behavior_measure = self.config.behavior_measure
        self.mutation_method = self.config.mutation_method
        
        with open(self.mutation_model.api_token_file, "r") as file:
            self.api_token = file.read().strip()
        self.aa_client = Client(token=self.api_token)

        self._init_prompt_pool()

        if self.config.fitness_method == "ai_feedback":
            self.quality_ai_feedback = AIFeedback(
                classifier_model=self.config.classifier_model,
                label_options=self.config.quality_ai_feedback_entries["quality"][
                    "answer_space"
                ],
                feedback_template=self.config.quality_ai_feedback_entries["quality"][
                    "feedback_prompt_template"
                ],
                aa_client=self.aa_client,
            )

    def _init_prompt_pool(self):
        self.prompt_pool = deque(maxlen=self.max_prompt_pool_size)

        # prompt pool to sample few shot prompt from, when using 'replace'
        if self.solution_init_method == "generated":
            assert (
                self.init_size_prompt_pool >= self.num_fewshot
            ), f"{self.init_size_prompt_pool} should be greater than the number of fewshots to generate a valid pool"
            for _ in range(self.init_size_prompt_pool):
                completion = self.generate_completion(
                    self.init_prompt_template, is_init=True
                )
                self.prompt_pool.append(completion)
        elif self.solution_init_method == "seed":
            try:
                with open(self.config.prompt_pool_path) as file:
                    seed_prompts = file.read().splitlines()
                assert (
                    len(seed_prompts) >= self.num_fewshot
                ), f"number of seed examples in the pool should be equal to or greater than the number of few shots"
                self.prompt_pool.extend(seed_prompts)
            except FileExistsError:
                print(f"file {self.config.prompt_pool_path} does not exist")

    def get_rng_state(self) -> Optional[np.random._generator.Generator]:
        return None

    def set_rng_state(self, rng_state: Optional[np.random._generator.Generator]):
        pass

    def construct_prompt(self, list_of_fewshot_reviews: list[str]) -> str:
        prompt = self.init_prompt_template + list_of_fewshot_reviews[0]
        for i in range(1, len(list_of_fewshot_reviews)):
            prompt += "\n###\n" + self.init_prompt_template + list_of_fewshot_reviews[i]
        prompt += "\n###\n" + self.init_prompt_template
        return prompt

    def generate_completion(self, prompt: str, is_init: bool = False) -> str:
        # ensure non-empty string is returned during init
        while True:
            completion = self.mutation_model.generate_programs(prompt)
            self.num_generations += 1
            if completion != "":  # non-empty completion
                if completion[-1] != ".":
                    completion += "."  # add extra full stop to ensure few-shot examples always end with stop
                return completion
            elif not is_init:  # else continue loop, only returning non-empty completion
                return completion

    def random(self) -> list[LMXGeneration]:
        list_of_generations = []
        for _ in range(self.batch_size):
            init_fewshot = random.sample(self.prompt_pool, k=self.num_fewshot)
            completion, prompt_metadata = self.make_prompt_and_completion(
                init_fewshot, is_init=True
            )
            list_of_generations.append(
                LMXGeneration(
                    prompt_metadata=prompt_metadata,
                    generated_completion=completion,
                    behavior_measure=self.behavior_measure,
                    classifier_model=self.config.classifier_model,
                    ai_feedback_entries=self.config.ai_feedback_entries,
                    aa_client=self.aa_client,
                )
            )

        return list_of_generations

    def make_prompt_and_completion(
        self, init_fewshot: list[str], is_init: bool = False
    ):
        prompt = self.construct_prompt(init_fewshot)
        completion = self.generate_completion(prompt, is_init)
        prompt_metadata = {"fewshot_items": init_fewshot, "prompt": prompt}
        return completion, prompt_metadata

    def mutate(
        self, x: Union[list[LMXGeneration], list[LMXGeneration, int]]
    ) -> list[LMXGeneration]:  # during default search, this could return none
        list_of_reviews = []
        for i in range(self.batch_size):
            x_individual = x[i]
            if type(x_individual) is tuple:
                genotype_object = x_individual[0]
                genotype_bin_idx = x_individual[1]
            else:
                genotype_object = x_individual

            if self.config.mutation_method == "replace":
                fewshot_items = genotype_object.prompt_metadata["fewshot_items"]
                idx_to_replace = random.choice(range(len(fewshot_items)))
                fewshot_items[idx_to_replace] = random.choice(self.prompt_pool)
            elif self.config.mutation_method == "lmx_near":
                # original formula ref in colab: https://colab.research.google.com/drive/1SXrq-YGffg6M725hgKXY1lqxTMsa1wLl?usp=sharing
                assert (
                    len(np.where(self.latest_genomes != 0)[0]) >= self.num_fewshot
                ), f"fewer than expected items in latest_genomes ({len(np.where(self.latest_genomes != 0)[0])})"
                fewshot_items = []
                orig_idx = np.array(
                    genotype_bin_idx
                )  # in list of individuals, access individual, and its bin idx
                non_empty_idx = np.where(self.latest_genomes != 0.0)
                non_empty_idx_array = np.array(non_empty_idx)
                non_empty_pop = self.latest_genomes[non_empty_idx]
                # all euclidean distances, between non-empty map indices and currently selected origin index
                dists = np.array(
                    [
                        1.0
                        / (
                            1
                            + np.linalg.norm(orig_idx - non_empty_idx_array[:, idx])
                            ** 3
                        )
                        for idx in range(len(non_empty_pop))
                    ]
                )
                dists /= np.sum(dists)
                examples = np.random.choice(
                    non_empty_pop, p=dists, replace=False, size=self.num_fewshot
                )
                examples = list(examples)
                for j, example in enumerate(examples):
                    fewshot_items.append(example.generated_completion)
            # gather chosen few-shot items and generate completion (crossover)
            completion, prompt_metadata = self.make_prompt_and_completion(fewshot_items)
            list_of_reviews.append(
                LMXGeneration(
                    prompt_metadata=prompt_metadata,
                    generated_completion=completion,
                    behavior_measure=self.behavior_measure,
                    classifier_model=self.config.classifier_model,
                    ai_feedback_entries=self.config.ai_feedback_entries,
                    aa_client=self.aa_client,
                )
            )

        return list_of_reviews

    def fitness(self, x: LMXGeneration) -> float:
        if x.generated_completion == "":
            return -np.inf
        if self.config.fitness_method == "embedding":
            query = self.config.fitness_query
            asymmetric_query = self.embed(query, SemanticRepresentation.Query)
            asymmetric_embedding = self.embed(
                x.generated_completion, SemanticRepresentation.Document
            )
            return cosine_similarity(asymmetric_query, asymmetric_embedding)
        elif self.config.fitness_method == "ai_feedback":
            feedback_scores = self.quality_ai_feedback.evaluate(
                {"genotype": x.generated_completion}
            )
            return float(feedback_scores[self.quality_ai_feedback.label_options[0]])
        else:
            raise NotImplementedError

    # helper function to embed text using the symmetric or asymmetric model
    def embed(self, text: str, representation: SemanticRepresentation):
        request = SemanticEmbeddingRequest(
            prompt=Prompt.from_text(text), representation=representation
        )
        result = self.aa_client.semantic_embed(request, model="luminous-base")
        return result.embedding


class P3ProbSolResult(Genotype):
    def __init__(self, program_str: str, result_obj: dict, config: P3ProbSolEnvConfig):
        """
        Genotype for a programming puzzle problem+solution pair.
        Args:
            program_str: the code for the pair.
            result_obj: the result of the solution.
            config: environment config
        """
        self.program_str = program_str
        self.result_obj = result_obj
        self.config = config

        i_f6 = program_str.find("def f6_2(")
        i_g6 = program_str.find("def g6_2(")
        i_assert = program_str.find("assert")
        self.problem_func = self.program_str[i_f6:i_g6].strip()
        self.solution_func = self.program_str[i_g6:i_assert].strip()

        # When comparing for phenotype, just use import statement and new probsol
        baseline = '''from typing import List

def f1_1(s: str):
    return "Hello " + s == "Hello world"

def g1_1():
    """Find a string that when concatenated onto 'Hello ' gives 'Hello world'."""
    return "world"'''
        self.baseline_emb = np.array(
            get_embedding(baseline, engine=self.config.embedding_model_path)
        )

        if self.config.embedding_model_type == "hf":
            self.pl = pipeline(
                "feature-extraction", model=self.config.embedding_model_path
            )
            seed_features = np.array(self.pl(baseline))
            self.scaler = StandardScaler()
            seed_features_scaled = self.scaler.fit_transform(np.squeeze(seed_features))
            self.pca = PCA(0.95)
            self.pca.fit(seed_features_scaled)

    def __str__(self) -> str:
        return self.program_str

    def to_phenotype(self) -> Optional[Phenotype]:
        if self.config.embedding_model_type == "openai":
            compare_str = (
                self.program_str
            )  # TODO: remove comments from f6_2 for diversity measurement
            i_assert = compare_str.find("assert")
            if i_assert > -1:
                compare_str = compare_str[:i_assert]
            emb = np.array(
                get_embedding(compare_str, engine=self.config.embedding_model_path)
            )
            return cosine_similarity(emb, self.baseline_emb)
        elif self.config.embedding_model_type == "hf":
            features = np.array(self.pl(self.program_str))
            features_scaled = self.scaler.transform(np.squeeze(features))
            pca_features = self.pca.transform(features_scaled)
            return pca_features.max(axis=0).flatten()

    def __getstate__(self):
        state = self.__dict__.copy()
        if "pl" in state:
            del state["pl"]
        if "scaler" in state:
            del state["scaler"]
        if "pca" in state:
            del state["pca"]
        return state


class P3ProbSol(BaseEnvironment[P3ProbSolResult]):
    def __init__(
        self,
        config: P3ProbSolEnvConfig,
        mutation_model: MutationModel,
    ) -> None:
        """
        The objective is to generate problem+solution pairs.
        Args:
            config: the config file path or dict.
            mutation_model: the diff model (or alternatives).
            ans_type: answer type
        """
        self.mutation_model = mutation_model
        self.config = config
        self.batch_size = self.config.batch_size
        self.seed_index = self.config.starting_seed
        self.rng = None

        if self.config.prompt_size == "long":
            self.prompt_seed = P3_PROBSOL_LONG_SEED
        elif self.config.prompt_size == "med":
            self.prompt_seed = P3_PROBSOL_MED_SEED
        else:
            raise ValueError("No seed string found")

        # Use the first example in the prompt seed as basis for embedding sizes
        i_first = self.prompt_seed.find("assert")
        first_example = self.prompt_seed[:i_first].strip()

        if self.config.embedding_model_type == "openai":
            self.genotype_ndim: int = 1
            self.genotype_space = np.repeat([[0, 1]], self.genotype_ndim, axis=0).T
        elif self.config.embedding_model_type == "hf":
            # Dummy to get behavior space shape
            dummy_pl = pipeline(
                "feature-extraction", model=self.config.embedding_model_path
            )
            dummy_features = np.array(dummy_pl(first_example))
            dummy_scaler = StandardScaler()
            dummy_features_scaled = dummy_scaler.fit_transform(
                np.squeeze(dummy_features)
            )
            dummy_pca = PCA(0.95)
            dummy_pca_features = dummy_pca.fit_transform(dummy_features_scaled)
            self.genotype_ndim: int = dummy_pca_features.shape[-1]
            self.genotype_space = np.repeat([[-20, 20]], self.genotype_ndim, axis=0).T

        # Get info for the seed puzzle that will be mutated
        # This puzzle is at the index of the puzzles array specified by self.seed_index
        # TODO: put this in a method or in construct_prompt()?
        puzzles = requests.get(
            "https://raw.githubusercontent.com/microsoft/PythonProgrammingPuzzles/v0.2/puzzles/puzzles.json"
        ).json()
        puzzle = puzzles[self.seed_index]
        if len(puzzle["sol_bodies"]) == 0:
            raise ValueError(
                f"No sample solution is provided for the puzzle at index {self.seed_index}"
            )

        f6_1 = puzzle["sat"].replace("def sat(", "def f6_1(")  # problem form is f6_1()
        g6_1 = puzzle["sol_header"].replace(
            "def sol(", "def g6_1("
        )  # solution form is g6_1()
        if self.config.prompt_size == "long":
            g6_1 += "\n" + puzzle["sol_docstring"]  # add in the docstring
        g6_1 += (
            "\n" + puzzle["sol_bodies"][0]
        )  # include the first example solution function body

        self.original_probsol = f6_1 + "\n\n" + g6_1 + "\n\n" + "assert f6_1(g6_1())"
        self.new_probsol_preamble = "def f6_2("

    def get_rng_state(self) -> Optional[np.random._generator.Generator]:
        warnings.warn("WARNING: rng state not used in this environment")
        return None

    def set_rng_state(self, rng_state: Optional[np.random._generator.Generator]):
        warnings.warn("WARNING: rng state not used in this environment")
        pass

    def construct_prompt(
        self, code_batch: Optional[Union[list[str], str]] = None
    ) -> dict[str, str]:
        prompt_str = self.prompt_seed

        if code_batch is None:
            # prompt with prob+sol from P3 dataset
            prompt_str += (
                f"\n\n{self.original_probsol}"  # add this particular probsol, f6_1() and g6_1(), to the prompt
                f"\n\n{self.new_probsol_preamble}"  # add f6_2() preamble to the prompt
            )
        else:
            # prompt with prob+sol that is given (one that was the output of a prev mutation)
            if isinstance(code_batch, list):
                # TODO: get nearby genotypes
                program_str = code_batch[0]
            elif isinstance(code_batch, str):
                program_str = code_batch

            # the prev output was f6_2 and g6_2, so now make it f6_1 and g6_1 for the prompt
            # and remove comments (which contain changes from prev f6_1) from new f6_1
            # TODO: pass in the whole object instead of the program_str since it already parsed some of this?
            i_f6 = program_str.find("def f6_2")
            program_str = program_str[i_f6:]  # remove import statement
            program_str = program_str.replace("f6_2(", "f6_1(")
            program_str = program_str.replace("g6_2(", "g6_1(")
            i_g6 = program_str.find("def g6_1(")
            # remove comments with """
            program_str = (
                re.sub('""".*"""', "", program_str[:i_g6]) + program_str[i_g6:]
            )
            # remove comments with # (and remove empty lines)
            i_g6 = program_str.find("def g6_1(")
            lines = program_str[:i_g6].strip().split("\n")
            new_lines = []
            for l in lines:
                if l.strip().startswith("#") or len(l.strip()) == 0:
                    continue
                new_lines.append(l)
            program_str = "\n".join(new_lines) + "\n\n" + program_str[i_g6:]
            program_str = program_str.strip()

            prompt_str += f"\n\n{program_str}" f"\n\n{self.new_probsol_preamble}"

        template = f"{P3_IMPORTS}\n{self.new_probsol_preamble}"
        return {"prompt": prompt_str, "template": template}

    def generate_programs(self, code_batch: list[str]) -> list[P3ProbSolResult]:
        """Generate new programs with a mutation model and evaluate them."""
        local_scope_exec = False
        generated_programs = self.mutation_model.generate_programs(
            code_batch, local_scope_exec
        )

        if self.config.sandbox:
            results = []
            for code in generated_programs:
                resp = requests.post(
                    f"{self.sandbox_server}/eval_p3_solution",
                    json={"code": code, "timeout": self.config.timeout},
                    timeout=self.config.timeout,
                )
                if resp.status_code == 200:
                    return_dict = json.loads(resp.text)
                    results.append(return_dict)
        else:
            # TODO: handle (probably inside of pool_exec_processes) all cases where the generated code returns
            # a generator type. The multithreaded execution pickles things and generators can't be pickled
            # which causes the whole thing to error out.
            # For now, try/except and re-try.
            try:
                results = pool_exec_processes(
                    generated_programs,
                    func_name="g6_2",
                    timeout=self.config.timeout,
                    processes=self.config.processes,
                    debug=self.config.debug,
                )
            except Exception as e:
                return self.generate_programs(code_batch)

        results = [
            {"program_str": gen_prog, "result_obj": res_obj, "config": self.config}
            for (gen_prog, res_obj) in zip(generated_programs, results)
        ]
        return [P3ProbSolResult(**p) for p in results]

    def fitness(self, probsol: P3ProbSolResult) -> float:
        """
        Fitness is the inverse of pass@k of the problem func.
        We want a pass@k of >0 so that the problem is reasonably solvable.
        So fitness=0 if unsolved (which is still better than -np.inf).
        Other than that, more difficult (lower pass@k) => higher fitness.
        """
        if isinstance(probsol.result_obj, ExecResult):
            return -np.inf

        # TODO: check type expected by f6_2 if any?
        # TODO: implement checks for absolute triviality of f6_2 requirements
        #   the fitness function being based on pass@k might take care of this though

        eval_code = (
            f"{P3_IMPORTS}\n"
            f"{probsol.problem_func}\n"
            f"def run_eval():\n"
            f"    return f6_2({probsol.result_obj})"
        )

        # Run code to see if g6_2 solves f6_2
        result = pool_exec_processes(
            eval_code,
            func_name="run_eval",
            timeout=self.config.timeout,
            processes=self.config.processes,
            debug=self.config.debug,
        )

        if result[0] != True:
            return -np.inf

        ### Do pass@k eval ###

        # Get f6_2() and make it the new f6()
        problem_str = probsol.problem_func.replace("def f6_2(", "def f6(")
        # Remove comments with """
        problem_str = re.sub('""".*"""', "", problem_str)
        # Remove comments with # (and remove empty lines)
        lines = problem_str.strip().split("\n")
        new_lines = []
        for l in lines:
            if l.strip().startswith("#") or len(l.strip()) == 0:
                continue
            new_lines.append(l)
        problem_str = "\n".join(new_lines)
        # Get solution_preamble for g6()
        i_end_preamble = probsol.solution_func.find("):")
        solution_preamble = probsol.solution_func[: i_end_preamble + 2].replace(
            "def g6_2(", "def g6("
        )

        p3_problem = P3Problem(
            self.config,  # TODO: make an actual P3ProblemEnvConfig
            self.mutation_model,
            problem_str=problem_str,
            solution_preamble=solution_preamble,
        )
        solutions = []
        for _ in range(self.config.eval_k // self.config.batch_size + 1):
            solutions += p3_problem.random()

        c = 0
        for s in solutions:
            if p3_problem.evaluate_solution(s) == True:
                c += 1

        pak = pass_at_k(len(solutions), c, self.config.eval_k)
        return 1 / pak if pak > 0 else 0

    def random(self) -> list[P3ProbSolResult]:
        program_list = [self.construct_prompt() for _ in range(self.config.batch_size)]
        new_probsols = self.generate_programs(program_list)
        return new_probsols

    def mutate(self, probsol_list: list[P3ProbSolResult]) -> list[P3ProbSolResult]:
        probsols = [pb.program_str for pb in probsol_list]
        program_list = list(map(self.construct_prompt, probsols))
        new_probsols = self.generate_programs(program_list)
        return new_probsols
