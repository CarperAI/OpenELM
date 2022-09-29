import itertools
import os
import random
import re
import shutil
from typing import Dict, Iterator, List

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm
from transformers import GPT2TokenizerFast

from codegen.modelling_codegen import CodeGenForCausalLM
from codex_execute import (TimeoutException, create_tempdir, reliability_guard,
                           swallow_io, time_limit)
from constants import PROJECT_PATH


def set_seed(seed, deterministic=True):
    random.seed(seed)
    # os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = not deterministic
        # torch.use_deterministic_algorithms(deterministic)


def create_model(ckpt_path, fp16=True):
    if fp16:
        return CodeGenForCausalLM.from_pretrained(ckpt_path,
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
    if cfg.gpus > 1:
        model = torch.nn.DataParallel(create_model(ckpt_path, fp16=use_fp16),
                                      device_ids=list(range(cfg.gpus))).to(device)
    else:
        model = create_model(ckpt_path, fp16=use_fp16).to(device)
    return model, tokenizer


def parity_reference(b1, b2, b3, b4):
    """Return binary parity of a sequence of input bits. Return 0 for even parity, 1 for odd parity."""
    bit_sum = sum([b1, b2, b3, b4])
    return bit_sum % 2


def quadratic(a, b, c, x):
    """Return quadratic: a,b,c are coefficients and x is the independent variable."""
    return a * x ** 2 + b * x + c


def reset_os_funcs(rmtree, rmdir, chdir):
    shutil.rmtree = rmtree
    os.rmdir = rmdir
    os.chdir = chdir


def eval_code_string(code_str: str, ground_truth: Dict, timeout: int = 5):
    if len(code_str) == 0:
        return 6  # No code found.
    code_dct: Dict = {}
    func_match = re.search(r"def (\w+)\s*\((.*?)\):", code_str)
    if func_match:
        func_name = func_match.group(1)
    else:
        return 6  # No function found in code.
    with create_tempdir():

        # These system calls are needed when cleaning up tempdir.
        import os
        import shutil

        rmtree = shutil.rmtree
        rmdir = os.rmdir
        chdir = os.chdir

        # Disable functionalities that can make destructive changes to the test.
        reliability_guard()
        try:
            # TODO: Check https://arxiv.org/pdf/2209.07753.pdf
            with swallow_io():
                with time_limit(timeout):
                    exec(code_str, {}, code_dct)
                    if not all([code_dct[func_name](*i) == res for i, res in ground_truth.items()]):
                        reset_os_funcs(rmtree, rmdir, chdir)
                        return 1  # Code runs but fails a test.
                    else:
                        reset_os_funcs(rmtree, rmdir, chdir)
                        return 0  # Passes all tests.
        except TimeoutException:
            reset_os_funcs(rmtree, rmdir, chdir)
            return 2  # Code takes too long to run.
        except RuntimeError:
            reset_os_funcs(rmtree, rmdir, chdir)
            return 3  # Code runs but crashes.
        except SyntaxError:
            reset_os_funcs(rmtree, rmdir, chdir)
            return 4  # Code does not run - syntax error.
        except TypeError:
            reset_os_funcs(rmtree, rmdir, chdir)
            return 5  # Code does not run - type error.
        except Exception:
            reset_os_funcs(rmtree, rmdir, chdir)
            return 6  # Code fails to run - other error.


def eval_completions(model_output: Iterator[str], task: str = "parity", timeout: int = 5):
    """Evaluate a batch of prompt completions on a task."""
    if task == "parity":
        ground_truth = {i: parity_reference(*i) for i in itertools.product(range(2), repeat=4)}
        for completion in model_output:
            # print(completion)
            # completion = truncate(completion)
            yield eval_code_string(completion, ground_truth, timeout)
    else:
        raise ValueError(f"Unknown task: {task}")


def sample(cfg, model, tokenizer, batch):
    """Run a model on a batch of contexts for a particular task."""
    device = torch.device(cfg.device)

    input_ids_len = batch["input_ids"].shape[1]
    assert input_ids_len < cfg.max_length
    with torch.no_grad():
        batch = batch.to(device)
        if cfg.gpus > 1:
            tokens = model.module.generate(
                **batch,
                do_sample=True,
                num_return_sequences=cfg.batch_size,
                temperature=cfg.temp,
                max_length=input_ids_len + cfg.max_length,
                top_p=cfg.top_p,
                pad_token_id=cfg.pad_token,
                use_cache=True,
            )
        else:
            tokens = model.generate(
                **batch,
                do_sample=True,
                num_return_sequences=cfg.batch_size,
                temperature=cfg.temp,
                max_length=input_ids_len + cfg.max_length,
                top_p=cfg.top_p,
                pad_token_id=cfg.pad_token,
                use_cache=True,
            )
        # "input_ids_len:" removes the prompt
        # - 1 adds in "def"
        text = tokenizer.batch_decode(tokens[:, input_ids_len - 1:, ...])
    return text


def mutate_code(n_bugs: int = 5, task: str = "parity"):
    """Mutate code to create n bugs."""
    mutation_template = ['# A buggy implementation\n#!/usr/bin/python3\n', '\n# Fixed bugs\ndef']
    if task == "parity":
        vars = ['b', 'b', 'b', 'b', 2]
        for i in range(n_bugs):
            vars[i] = 'c' if i < 4 else 3
        func_str = ('def parity(b1,b2,b3,b4):\n    """Return binary parity of a sequence of input bits.'
                    ' Return 0 for even parity, 1 for odd parity."""\n    bit_sum = sum(['
                    '{}1,{}2,{}3,{}4])\n    return bit_sum % {}'.format(*vars))
        mutation_template.insert(1, func_str)
        return ''.join(mutation_template)
    else:
        raise ValueError(f"Unknown task: {task}")


def run_benchmark(cfg):
    model, tokenizer = model_setup(cfg)
    mutated_str = mutate_code(n_bugs=cfg.n_bugs, task=cfg.tasks[0])
    # mutated_encoding = tokenizer([mutated_str] * cfg.gpus, truncation=True, padding=True,
    mutated_encoding = tokenizer([mutated_str], truncation=True, padding=True,
                                max_length=2048,
                                return_tensors='pt')
    num_batches = cfg.n_trials // cfg.batch_size
    for i in tqdm(range(num_batches), desc=f"Running benchmark with {cfg.n_bugs} bugs"):
        set_seed(torch.random.seed())
        completions = sample(cfg, model, tokenizer, mutated_encoding)
        truncations = map(truncate, completions)
        if i == 0:
            eval_results = np.fromiter(eval_completions(truncations, task=cfg.tasks[0], timeout=cfg.timeout),
                                                        dtype=np.byte)
        else:
            eval_results = np.vstack((eval_results,
                                      np.fromiter(eval_completions(truncations, task=cfg.tasks[0], timeout=cfg.timeout),
                                                  dtype=np.byte)))
    corr_cnt = np.count_nonzero(eval_results == 0)
    print(f"Number of bugs: {cfg.n_bugs}")
    print(f"Result: {corr_cnt} successful completions in {cfg.n_trials} trials, {(corr_cnt / cfg.n_trials) * 100}%")


# Load hydra config from yaml files and command line arguments.
@hydra.main(config_path=str(PROJECT_PATH), config_name="benchmark_cfg",
            version_base="1.2")
def main(cfg):
    print('----------------- Config ---------------')
    print(OmegaConf.to_yaml(cfg))
    print('-----------------  End -----------------')
    run_benchmark(cfg)


if __name__ == "__main__":
    main()
