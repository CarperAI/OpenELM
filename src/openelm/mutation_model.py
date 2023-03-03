import functools
import os
import re
from abc import ABC, abstractmethod

import numpy as np

from openelm.codegen import model_setup, sample, set_seed, truncate
from openelm.configs import ModelConfig
from openelm.utils.diff_eval import apply_diff, split_diff


class MutationModel(ABC):
    """Base model class for all mutation models."""

    def __init__(self) -> None:
        self.config: ModelConfig

    @abstractmethod
    def generate_programs(self, *args, **kwargs) -> list[str]:
        raise NotImplementedError


class PromptModel(MutationModel):
    """Mutation model that uses prompts to change a seed."""

    def __init__(
        self,
        cfg: ModelConfig,
    ) -> None:

        self.config: ModelConfig = cfg
        seed: int = set_seed(self.config.seed)
        # Use RNG to rotate random seeds during inference.
        self.rng = np.random.default_rng(seed=seed)
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.model, self.tokenizer, self.device = model_setup(self.config)

    def generate_programs(
        self, prompt_dicts: list[dict[str, str]], local_scope_truncate: bool, **kwargs
    ) -> list[str]:
        """
        Generate new programs from a batch of programs.

        Given a piece of code, do prompt mutation, execute the code,
        and return the result.

        Args:
            code_batch (list[str]): A list of code strings.
            local_scope_truncate (bool): Whether to truncate the code to the
            local scope.

        Returns:
            A list of code strings.
        """
        prompts = [prompt_dict["prompt"] for prompt_dict in prompt_dicts]
        templates = [prompt_dict["template"] for prompt_dict in prompt_dicts]
        encodings = self.tokenizer(
            prompts,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        completions: list[str] = sample(
            encodings,
            self.config,
            self.model,
            self.tokenizer,
            batch_size=1,
        )
        trunc = functools.partial(truncate, only_local_scope=local_scope_truncate)
        truncations: list[str] = [
            templates[i] + trunc(completions[i]) for i in range(len(completions))
        ]
        return truncations


class DiffModel(PromptModel):
    def __init__(
        self,
        cfg: ModelConfig,
    ) -> None:
        super().__init__(cfg)

    def construct_prompt(self, code: str) -> tuple[str, str]:
        prompt_list = [
            "<NME> walker.py\n<BEF> ",
            code,
            "\n<MSG> Fixed bugs",
        ]
        prompt_str = "".join(prompt_list)
        prompt_str = (
            code + self.func_template.instruction + self.func_template.func_preamble
        )
        preamble_str = (
            self.func_template.import_line
            + self.func_template.instruction
            + self.func_template.func_preamble
        )
        return prompt_str, preamble_str

    def generate_programs(self, code_batch: list[str]) -> list[str]:
        """
        Generate a new program for a diff model from a batch of programs.

        Given a piece of code, do prompt mutation, execute the code,
        and return the result.

        Args:
            code (str): The full code string.

        Returns:
            A numpy array (if successful) or the exception object.
        """
        prompts, preamble_strings = zip(*map(self.construct_prompt, code_batch))
        encodings = self.tokenizer(
            list(prompts),
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        completions: list[str] = sample(
            encodings,
            self.config,
            self.model,
            self.tokenizer,
            batch_size=1,
        )

        local_scope_exec: bool = len(self.func_template.func_preamble) > 0
        end_of_diff = re.compile("\n[^ +-@]+")
        trunc = functools.partial(truncate, only_local_scope=local_scope_exec)
        self.truncations: list[str] = [
            preamble_strings[i] + trunc(completions[i]) for i in range(len(completions))
        ]
        outputs = []
        for i, code in enumerate(self.truncations):
            # split the diff text according to <NME>, <BEF>, <MSG>, <DFF>.
            parsed: dict = split_diff(code)
            # truncate the diff hunk at the first line not starting with " ",
            # "+", "-", or "@".
            if parsed and all(
                (s in parsed for s in ["name", "file", "message", "diff"])
            ):
                diff_hunk: str = end_of_diff.split(parsed["diff"])[0]
                nme_idx: int = diff_hunk.find("<NME>")
                if nme_idx != -1:
                    diff_hunk = diff_hunk[:nme_idx]
                outputs.append(apply_diff(prompts[i], diff_hunk))
        return outputs
