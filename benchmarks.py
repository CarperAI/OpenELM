import itertools
import os
import random
import re
from typing import Dict, Iterator, List

import hydra
import numpy as np
import torch
from transformers import GPT2TokenizerFast

from codegen.modelling_codegen import CodeGenForCausalLM
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


def create_model(ckpt_path, fp16=True):
    if fp16:
        return CodeGenForCausalLM.from_pretrained(ckpt_path,
                                                  # revision='float16',
                                                  torch_dtype=torch.float16,
                                                  low_cpu_mem_usage=True)
    else:
        return CodeGenForCausalLM.from_pretrained(f"Salesforce/{ckpt_path.name}",
                                                  cache_dir=ckpt_path)


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
    set_seed(cfg.seed, deterministic=True)
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


def parity(b1, b2, b3, b4):
    """Return binary parity of a sequence of input bits. Return 0 for even parity, 1 for odd parity."""
    bit_sum = sum([b1, b2, b3, b4])
    return bit_sum % 2


def quadratic(a, b, c, x):
    """Return quadratic: a,b,c are coefficients and x is the independent variable."""
    return a * x ** 2 + b * x + c


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


def eval_completions(model_output: Iterator[str], task="parity"):
    """Evaluate a batch of prompt completions on a task."""
    if task == "parity":
        inputs = [i for i in itertools.product(range(2), repeat=4)]
        ground_truth = {i: parity(*i) for i in inputs}
        for completion in model_output:
            # print(completion)
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


def mutate_code(n_bugs: int = 5, task: str = "parity"):
    """Mutate code to create n bugs."""
    mutation_template = ['# A buggy implementation\n#!/usr/bin/python3\n', '\n# Fixed bugs\n']
    if task == "parity":
        vars = ['b', 'b', 'b', 'b', 2]
        for i in range(n_bugs):
            vars[i] = 'c' if i < 4 else 3
        func_str = ('def parity(b1,b2,b3,b4):\n    """Return binary parity of a sequence of input bits.'
                    ' Return 0 for even parity, 1 for odd parity."""\n    bit_sum = sum(['
                    f'{vars[0]}1,{vars[1]}2,{vars[2]}3,{vars[3]}4])\n    return bit_sum % {vars[4]}')
        mutation_template.insert(1, func_str)
        return ''.join(mutation_template)
    else:
        raise ValueError("Unknown task: {}".format(task))


def run_benchmark(cfg):
    model, tokenizer = model_setup(cfg)
    mutated_str = mutate_code(n_bugs=cfg.n_bugs, task=cfg.tasks[0])

    contexts = [mutated_str] * cfg.batch_size
    completions = sample(cfg, model, tokenizer, contexts)
    # TODO: better truncation?
    # truncations = map(truncate, completions)
    truncations = completions

    eval_results = np.fromiter(eval_completions(truncations, task=cfg.tasks[0]),
                               dtype=np.byte)
    successes = np.count_nonzero(eval_results == 0)
    print("Results: ", successes, cfg.batch_size, (successes / cfg.batch_size) * 100)


# Load hydra config from yaml files and command line arguments.
@hydra.main(config_path=str(PROJECT_PATH), config_name="benchmark_cfg",
            version_base="1.2")
def main(cfg):
    run_benchmark(cfg)


if __name__ == "__main__":
    main()
