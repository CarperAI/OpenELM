import os
import random
import re

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def set_seed(seed=None, deterministic=True) -> int:
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


def create_model(path=None, fp16=True):
    if fp16:
        return AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=torch.float16, low_cpu_mem_usage=True
        )
    else:
        return AutoModelForCausalLM.from_pretrained(path)


def truncate(completion: str, def_num=1, print_num=0, only_local_scope=False):
    def find_re(string, pattern, start_pos):
        m = pattern.search(string, start_pos)
        return m.start() if m else -1

    terminals = [
        re.compile(r, re.MULTILINE)
        for r in ["^#", re.escape("<|endoftext|>"), "^'''", '^"""', "\n\n\n"]
    ]
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


def model_setup(cfg, device=None):
    set_seed(cfg.seed, deterministic=True)
    if device is None:
        device = torch.device("cuda" if cfg.cuda else "cpu")
    use_fp16 = True
    if not cfg.fp16 or device.type == "cpu":
        use_fp16 = False

    if cfg.model.startswith("codegen-16B"):
        use_fp16 = True

    tokenizer = AutoTokenizer.from_pretrained(cfg.model)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = 50256

    model_path = cfg.model
    if cfg.gpus > 1:
        model = torch.nn.DataParallel(
            create_model(model_path, fp16=use_fp16), device_ids=list(range(cfg.gpus))
        ).to(device)
    else:
        model = create_model(model_path, fp16=use_fp16).to(device)
    return model, tokenizer, device


def sample(
    batch, cfg, model, tokenizer, decode: bool = True, starting_idx=None, **kwargs
) -> list[str]:
    """Run a model on a batch of contexts for a particular task."""
    batch_size = kwargs.get("batch_size", cfg.batch_size)
    device = kwargs.get("device", torch.device("cuda" if cfg.cuda else "cpu"))
    temperature = kwargs.get("temperature", cfg.temp)
    top_p = kwargs.get("top_p", cfg.top_p)
    gen_max_len = kwargs.get("gen_max_len", cfg.gen_max_len)

    input_ids_len = batch["input_ids"].shape[1]
    if starting_idx is None:
        starting_idx = input_ids_len
    with torch.inference_mode():
        batch = batch.to(device)
        # TODO: num_gpus > 1
        if cfg.gpus > 1:
            tokens = model.module.generate(
                **batch,
                do_sample=True,
                num_return_sequences=batch_size,
                temperature=temperature,
                max_new_tokens=gen_max_len,
                top_p=top_p,
                pad_token_id=tokenizer.pad_token_id,
                use_cache=True,
            )
        else:
            tokens = model.generate(
                **batch,
                do_sample=True,
                num_return_sequences=batch_size,
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
