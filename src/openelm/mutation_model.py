import functools
import os
import re
from abc import ABC, abstractmethod
from typing import Any, Optional

import numpy as np
import openai

from openelm.codegen import model_setup, sample, set_seed, truncate
from openelm.configs import ModelConfig
from openelm.utils.diff_eval import apply_diff, split_diff


def get_model(config: ModelConfig) -> tuple[Any, Optional[Any], Optional[Any]]:
    if config.model_type == "hf":
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        model, tokenizer, device = model_setup(config)
        return model, tokenizer, device
    else:
        return model, None, None


def sample_openai(batch: list[str], config, *args, **kwargs) -> list[str]:
    completions: list[str] = []
    responses = openai.Completion.create(
        model=config.model_name,
        prompt=batch,
        max_tokens=config.gen_max_len,
        n=1,
        stop=None,
        temperature=config.temp,
        top_p=config.top_p,
    )
    for response in responses:
        completions.append(response.choices[0].text.strip())
    return completions


def sample_openai_chat(batch: list[str], config, *args, **kwargs) -> list[str]:
    completions: list[str] = []
    # TODO: async calls
    for prompt in batch:
        messages = [
            {"role": "system", "content": "You are an AI that can generate code."},
            {"role": "user", "content": prompt},
        ]

        response = openai.ChatCompletion.create(
            model=config.model_name,
            messages=messages,
            max_tokens=config.gen_max_len,
            n=1,
            stop=None,
            temperature=config.temp,
            top_p=config.top_p,
        )

        completions.append(response.choices[0].text.strip())
    return completions


def generate(batch, config: ModelConfig, *args, **kwargs):
    if config.model_type == "hf":
        tokenizer = kwargs.get("tokenizer", None)
        if tokenizer is None:
            raise ValueError("No tokenizer found in args.")
        encodings = tokenizer(
            batch,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        # TODO: batching within sample
        return sample(encodings, config, *args, **kwargs)
    elif config.model_type == "oai":
        # TODO: query model name and divide into chat and non-chat
        return sample_openai_chat(batch, config, *args, **kwargs)


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
        self.model, self.tokenizer, self.device = get_model(self.config)

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
        completions: list[str] = generate(
            batch=prompts,
            config=self.config,
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device,
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
        completions: list[str] = generate(
            batch=prompts,
            config=self.config,
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device,
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
