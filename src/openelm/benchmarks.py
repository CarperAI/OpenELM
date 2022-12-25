import itertools
import os
import re
import shutil
from typing import Iterator

import hydra
import numpy as np
import torch
from openelm.constants import SRC_PATH
from omegaconf import OmegaConf
from tqdm import tqdm

from openelm.codegen.codegen_utilities import model_setup, sample, set_seed, truncate
from openelm.codegen.codex_execute import (
    TimeoutException,
    create_tempdir,
    reliability_guard,
    swallow_io,
    time_limit,
)


def parity_reference(b1, b2, b3, b4):
    """Return binary parity of a sequence of input bits. Return 0 for even parity, 1 for odd parity."""
    bit_sum = sum([b1, b2, b3, b4])
    return bit_sum % 2


def quadratic(a, b, c, x):
    """Return quadratic: a,b,c are coefficients and x is the independent variable."""
    return a * x**2 + b * x + c


def reset_os_funcs(rmtree, rmdir, chdir):
    shutil.rmtree = rmtree
    os.rmdir = rmdir
    os.chdir = chdir


def eval_code_string(code_str: str, ground_truth: dict, timeout: int = 5):
    if len(code_str) == 0 or "def " not in code_str:
        return 6  # No code found or no function found.
    code_dct: dict = {}
    func_match = re.search(r"def (\w+)\s*\((.*?)\):", code_str)
    if func_match:
        func_name = func_match.group(1)
    else:
        return 6  # No proper function found in code.
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
                    if not all(
                        [
                            code_dct[func_name](*i) == res
                            for i, res in ground_truth.items()
                        ]
                    ):
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


def eval_completions(
    model_output: Iterator[str], task: str = "parity", timeout: int = 5
):
    """Evaluate a batch of prompt completions on a task."""
    if task == "parity":
        ground_truth = {
            i: parity_reference(*i) for i in itertools.product(range(2), repeat=4)
        }
        for completion in model_output:
            # print(completion)
            # completion = truncate(completion)
            yield eval_code_string(completion, ground_truth, timeout)
    else:
        raise ValueError(f"Unknown task: {task}")


def mutate_code(n_bugs: int = 5, task: str = "parity"):
    """Mutate code to create n bugs."""
    mutation_template = [
        "# A buggy implementation\n#!/usr/bin/python3\n",
        "",  # placeholder for the context, e.g., the buggy code
        "\n# Fixed bugs\ndef",
    ]
    if task == "parity":
        vars = ["b", "b", "b", "b", 2]
        for i in range(n_bugs):
            vars[i] = "c" if i < 4 else 3
        func_str = (
            'def parity(b1,b2,b3,b4):\n    """Return binary parity of a sequence of input bits.'
            ' Return 0 for even parity, 1 for odd parity."""\n    bit_sum = sum(['
            "{}1,{}2,{}3,{}4])\n    return bit_sum % {}".format(*vars)
        )
        mutation_template[1] = func_str
        return "".join(mutation_template)
    else:
        raise ValueError(f"Unknown task: {task}")


def run_benchmark(cfg):
    model, tokenizer = model_setup(cfg)
    mutated_str = mutate_code(n_bugs=cfg.n_bugs, task=cfg.tasks[0])
    # mutated_encoding = tokenizer([mutated_str] * cfg.gpus, truncation=True, padding=True,
    mutated_encoding = tokenizer(
        [mutated_str],
        truncation=True,
        padding=True,
        max_length=2048,
        return_tensors="pt",
    )
    input_ids_len = mutated_encoding.input_ids.shape
    text = []
    for i in range(input_ids_len[1]):
        text.append(tokenizer.batch_decode(mutated_encoding.input_ids[:, i]))
    num_batches = cfg.n_trials // cfg.batch_size
    for i in tqdm(range(num_batches), desc=f"Running benchmark with {cfg.n_bugs} bugs"):
        set_seed(torch.random.seed())
        completions = sample(cfg, model, tokenizer, mutated_encoding, add_def=True)
        truncations = map(truncate, completions)
        if i == 0:
            eval_results = np.fromiter(
                eval_completions(truncations, task=cfg.tasks[0], timeout=cfg.timeout),
                dtype=np.byte,
            )
        else:
            eval_results = np.vstack(
                (
                    eval_results,
                    np.fromiter(
                        eval_completions(
                            truncations, task=cfg.tasks[0], timeout=cfg.timeout
                        ),
                        dtype=np.byte,
                    ),
                )
            )
    corr_cnt = np.count_nonzero(eval_results == 0)
    print(f"Number of bugs: {cfg.n_bugs}")
    print(
        f"Result: {corr_cnt} successful completions in {cfg.n_trials} trials, {(corr_cnt / cfg.n_trials) * 100}%"
    )


# Load hydra config from yaml files and command line arguments.
@hydra.main(
    config_path=str(SRC_PATH / "config"),
    config_name="benchmark_cfg",
    version_base="1.2",
)
def main(cfg):
    print("----------------- Config ---------------")
    print(OmegaConf.to_yaml(cfg))
    print("-----------------  End -----------------")
    run_benchmark(cfg)


if __name__ == "__main__":
    main()
