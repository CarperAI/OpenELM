import os
import random
import re
from typing import Optional

import numpy as np
import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)

from openelm.configs import ModelConfig


def set_seed(seed=None, deterministic=False) -> int:
    if seed is None:
        seed = np.random.default_rng().integers(2**32 - 1)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = not deterministic
        # torch.use_deterministic_algorithms(deterministic)
    return seed


def truncate(completion: str, def_num=1, print_num=0, only_local_scope=False):
    def find_re(string, pattern, start_pos):
        m = pattern.search(string, start_pos)
        return m.start() if m else -1

    terminals = [
        re.compile(r, re.MULTILINE)
        for r in ["^#", re.escape("<|endoftext|>"), "^'''", '^"""', "\n\n\n"]
    ]
    if print_num > 0:
        prints = list(re.finditer("^print", completion, re.MULTILINE))
        if print_num >= 0 and len(prints) > print_num:
            completion = completion[: prints[print_num].start()]

    if only_local_scope:
        global_lines = list(re.finditer("^[a-zA-Z]", completion, re.MULTILINE))
        if global_lines:
            completion = completion[: global_lines[0].start()]
    else:
        defs = list(re.finditer("^def", completion, re.MULTILINE))
        if len(defs) > def_num:
            completion = completion[: defs[def_num].start()]

    start_pos = 0

    terminals_pos = [
        pos
        for pos in [find_re(completion, terminal, start_pos) for terminal in terminals]
        if pos != -1
    ]
    if len(terminals_pos) > 0:
        return completion[: min(terminals_pos)]
    else:
        return completion


def model_setup(cfg: ModelConfig, device=None, codegen_tokenizer: bool = True):
    set_seed(cfg.seed)
    if device is None:
        device = torch.device("cuda")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_path)
    # TODO: may need to check model type to determine padding
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.model_max_length > 32768:
        tokenizer.model_max_length = 2048

    tokenizer.pad_token = tokenizer.eos_token

    autoconfig = AutoConfig.from_pretrained(
        cfg.model_path, trust_remote_code=cfg.trust_remote_code
    )

    if autoconfig.model_type == "t5":
        model_cls = AutoModelForSeq2SeqLM
    else:
        model_cls = AutoModelForCausalLM
    model = model_cls.from_pretrained(
        cfg.model_path,
        torch_dtype=torch.float16 if cfg.fp16 else None,
        low_cpu_mem_usage=cfg.fp16,
        trust_remote_code=cfg.trust_remote_code,
        # device_map="auto",
    ).to(device)

    return model, tokenizer, device


def sample(
    batch,
    cfg: ModelConfig,
    model,
    tokenizer,
    decode: bool = True,
    starting_idx: Optional[int] = None,
    num_return_sequences: Optional[int] = None,
    **kwargs,
) -> list[str]:
    """Run a model on a batch of contexts for a particular task."""
    if num_return_sequences is None:
        num_return_sequences = cfg.batch_size
    device = kwargs.get("device", torch.device("cuda"))
    temperature = kwargs.get("temperature", cfg.temp)
    top_p = kwargs.get("top_p", cfg.top_p)
    gen_max_len = kwargs.get("gen_max_len", cfg.gen_max_len)

    input_ids_len = batch["input_ids"].shape[1]
    if starting_idx is None:
        starting_idx = input_ids_len
    with torch.inference_mode():
        batch = batch.to(device)
        tokens = model.generate(
            **batch,
            do_sample=True,
            num_return_sequences=num_return_sequences,
            temperature=temperature,
            max_new_tokens=gen_max_len,
            top_p=top_p,
            pad_token_id=tokenizer.pad_token_id,
            use_cache=True,
        )
        if decode:
            text: list[str] = tokenizer.batch_decode(tokens[:, starting_idx:, ...])
            return text
        else:
            return tokens[:, starting_idx:, ...]
