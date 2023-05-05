import functools
import os
import re
from abc import ABC, abstractmethod
from typing import Any, Optional

import numpy as np
from langchain.llms.base import LLM
from langchain.schema import Generation, LLMResult

from openelm.codegen import model_setup, sample, set_seed, truncate
from openelm.configs import LangChainModelConfig, ModelConfig
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

    def __init__(self, config: ModelConfig) -> None:
        self.config: ModelConfig = config
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
            prompt_dicts (list[dict[str, str]): A list of dictionaries containing
            the prompt and template for each program.
            local_scope_truncate (bool): Whether or not to truncate the code to
            the local scope.

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
            num_return_sequences=1,
        )
        trunc = functools.partial(truncate, only_local_scope=local_scope_truncate)
        truncations: list[str] = [
            templates[i] + trunc(completions[i]) for i in range(len(completions))
        ]
        return truncations


class DiffModel(PromptModel):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)

    def generate_programs(
        self, prompt_dicts: list[dict[str, str]], local_scope_truncate: bool, **kwargs
    ) -> list[str]:
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
            num_return_sequences=1,
        )

        end_of_diff = re.compile("\n[^ +-@]+")
        trunc = functools.partial(truncate, only_local_scope=local_scope_truncate)
        truncations: list[str] = [
            templates[i] + trunc(completions[i]) for i in range(len(completions))
        ]
        outputs: list[str] = []
        for i, code in enumerate(truncations):
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


class LangChainPromptModel(LLM):
    config: LangChainModelConfig
    model: Any
    tokenizer: Any
    device: Any

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.model, self.tokenizer, self.device = model_setup(self.config)

    def _call(self, prompt: str, stop: Optional[list[str]] = None) -> str:
        """Run the LLM on the given prompt and input."""
        raise NotImplementedError

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "huggingface"

    def _generate(
        self, prompts: list[str], stop: Optional[list[str]] = None
    ) -> LLMResult:
        """Run the LLM on the given prompt and input."""
        generations = []
        batch_size = self.config.batch_size
        # Get the total number of batches
        total_batches = (len(prompts) + batch_size - 1) // batch_size
        # print("Total batches: ", total_batches)
        # TODO: encode before loop, Use num_return sequences for this
        for i in range(total_batches):
            start_index = i * batch_size
            end_index = min((i + 1) * batch_size, len(prompts))
            batched_prompts = prompts[start_index:end_index]
            encodings = self.tokenizer(
                batched_prompts,
                truncation=True,
                padding=True,
                return_tensors="pt",
            )
            texts = sample(
                encodings,
                cfg=self.config,
                model=self.model,
                tokenizer=self.tokenizer,
                num_return_sequences=1,
            )
            # results: list[str] = list(map(truncate, texts))
            generations.append([Generation(text=text) for text in texts])
        # TODO: return logprobs
        return LLMResult(generations=generations)
