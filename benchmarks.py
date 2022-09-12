import itertools
import os
import pathlib
import random
import re
from typing import Dict, Iterator, List

import hydra
import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          CodeGenForCausalLM, GPT2TokenizerFast)

from constants import PROJECT_PATH


def set_seed(seed, deterministic=True):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = not deterministic
        # torch.use_deterministic_algorithms(deterministic)


def download_checkpoint(model_name: str, save_dir: pathlib.Path):
    """Download a checkpoint from the Hugging Face Hub."""
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)


def create_model(ckpt_path, fp16=True):
    if fp16:
        return CodeGenForCausalLM.from_pretrained(ckpt_path, revision='float16',
                                                  torch_dtype=torch.float16,
                                                  low_cpu_mem_usage=True)
    else:
        return CodeGenForCausalLM.from_pretrained(ckpt_path)


def create_tokenizer():
    t = GPT2TokenizerFast.from_pretrained('gpt2')
    t.max_model_input_sizes['gpt2'] = 1e20
    return t


def include_whitespace(t, n_min=2, n_max=20, as_special_tokens=False):
    t.add_tokens([' ' * n for n in reversed(range(n_min, n_max))],
                 special_tokens=as_special_tokens)
    return t


def include_tabs(t, n_min=2, n_max=20, as_special_tokens=False):
    t.add_tokens(['\t' * n for n in reversed(range(n_min, n_max))],
                 special_tokens=as_special_tokens)
    return t


def create_custom_gpt2_tokenizer():
    t = create_tokenizer()
    t = include_whitespace(t=t, n_min=2, n_max=32, as_special_tokens=False)
    t = include_tabs(t=t, n_min=2, n_max=10, as_special_tokens=False)
    return t


def truncate(completion):

    def find_re(string, pattern, start_pos):
        m = pattern.search(string, start_pos)
        return m.start() if m else -1

    terminals = [
        re.compile(r, re.MULTILINE)
        for r in
        [
            '^#',
            re.escape('<|endoftext|>'),
            "^'''",
            '^"""',
            '\n\n\n'
        ]
    ]

    prints = list(re.finditer('^print', completion, re.MULTILINE))
    if len(prints) > 1:
        completion = completion[:prints[1].start()]

    defs = list(re.finditer('^def', completion, re.MULTILINE))
    if len(defs) > 1:
        completion = completion[:defs[1].start()]

    start_pos = 0

    terminals_pos = [pos for pos in [find_re(completion, terminal, start_pos)
                                     for terminal in terminals] if pos != -1]
    if len(terminals_pos) > 0:
        return completion[:min(terminals_pos)]
    else:
        return completion


def model_setup(cfg):
    device = torch.device(cfg.device)
    use_fp16 = True
    if (not cfg.fp16 or device.type == "cpu"):
        use_fp16 = False

    if cfg.model.startswith("codegen-16B"):
        use_fp16 = True

    tokenizer = create_custom_gpt2_tokenizer()
    tokenizer.padding_side = 'left'
    tokenizer.pad_token = cfg.pad_token

    ckpt_path = PROJECT_PATH / "checkpoints" / cfg.model
    model = create_model(ckpt_path, fp16=use_fp16).to(device)
    return model, tokenizer


def four_parity_reference(b1, b2, b3, b4):
    bit_sum = sum([b1, b2, b3, b4])
    return bit_sum % 2


def eval_code_string(code_str: str, ground_truth: Dict):
    code_dct: Dict = {}
    func_match = re.search(r"def (\w+)\s*\((.*?)\):", code_str)
    if func_match:
        func_name = func_match.group(1)
    else:
        return 3  # No function found in code.
    try:
        exec(code_str, {}, code_dct)
        if not all([code_dct[func_name](*i) == res for i, res in ground_truth.items()]):
            return 1  # Error in code, but it runs.
        else:
            return 0  # Passes all tests.
    except Exception:
        return 2  # Code fails to run.


def eval_completions(model_output: Iterator[str], task="4-parity"):
    """Evaluate a batch of prompt completions on a task."""
    if task == "4-parity":
        inputs = [i for i in itertools.product(range(2), repeat=4)]
        ground_truth = {i: four_parity_reference(*i) for i in inputs}
        for completion in model_output:
            # completion = truncate(completion)
            if len(completion) > 0:
                yield eval_code_string(completion, ground_truth)
            else:
                yield 3  # No code found.
    else:
        raise ValueError("Unknown task: {}".format(task))


def sample(cfg, model, tokenizer, contexts: List[str]):
    """Run a model on a batch of contexts for a particular task."""
    device = torch.device(cfg.device)
    inputs = tokenizer(
        contexts,
        truncation=True,
        padding=True,
        max_length=2048,
        return_tensors='pt',
    )
    input_ids_len = inputs["input_ids"].shape[1]
    assert input_ids_len < cfg.max_length

    with torch.no_grad():
        inputs = inputs.to(device)
        tokens = model.generate(
            **inputs,
            do_sample=True,
            num_return_sequences=1,
            temperature=cfg.temp,
            max_length=input_ids_len + cfg.max_length,
            top_p=cfg.top_p,
            pad_token_id=cfg.pad_token,
            use_cache=True,
        )
        # "input_ids_len:" removes the prompt
        text = tokenizer.batch_decode(tokens[:, input_ids_len:, ...])
    return text


def run_benchmark(cfg):
    model, tokenizer = model_setup(cfg)
    code_example = """# A buggy implementation\n#!/usr/bin/python3\ndef parity(b1,b2,b3,b4):\n  \"\"\" Return binary parity of a sequence of input bits. Return 0 for even parity, 1 for odd parity \"\"\"\n  bit_sum = sum([b1,b2,b3,b4])\n  return bit_sum % 2\n# Fixed bugs\n"""
    contexts = [code_example, code_example]
    completions = sample(cfg, model, tokenizer, contexts)
    # TODO: better truncation?
    truncations = map(truncate, completions)

    eval_results = list(eval_completions(truncations, task=cfg.tasks[0]))
    print(eval_results)
    print(list(truncations))


# Load hydra config from yaml files and command line arguments.
@hydra.main(config_path=str(PROJECT_PATH), config_name="benchmark_cfg")
def main(cfg):
    run_benchmark(cfg)


if __name__ == "__main__":
    main()
