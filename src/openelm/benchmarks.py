import itertools
import os
import re
import shutil
from typing import Iterator

import hydra
import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
import torch

from openelm.codegen.codegen_utilities import model_setup, sample, truncate
from openelm.codegen.codex_execute import (
    TimeoutException,
    create_tempdir,
    reliability_guard,
    swallow_io,
    time_limit,
)
from openelm.constants import SRC_PATH


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


def run_benchmark(cfg, model, tokenizer, device, n_bugs, temperature):
    mutated_str = mutate_code(n_bugs=n_bugs, task=cfg.tasks[0])
    # mutated_encoding = tokenizer([mutated_str] * cfg.gpus, truncation=True, padding=True,
    mutated_encoding = tokenizer(
        [mutated_str],
        truncation=True,
        padding=True,
        return_tensors="pt",
    ).to(device)
    token_len = mutated_encoding.input_ids.shape[1]
    num_batches = cfg.n_trials // cfg.batch_size
    for i in tqdm(range(num_batches), desc=f"Running benchmark with {n_bugs} bugs",
                  disable=False):
        # completions = sample(cfg, model, tokenizer, mutated_encoding, add_def=True)
        with torch.inference_mode():
            tokens = model.generate(
                **mutated_encoding,
                do_sample=True,
                num_return_sequences=cfg.batch_size,
                temperature=temperature,
                max_length=token_len + cfg.gen_max_len,
                top_p=cfg.top_p,
                pad_token_id=cfg.pad_token,
                use_cache=True,
            )
            text = tokenizer.batch_decode(tokens[:, token_len - 1:, ...])
        truncations = map(truncate, text)
        if i == 0:
            eval_results = np.fromiter(
                eval_completions(truncations, task=cfg.tasks[0], timeout=cfg.timeout),
                dtype=np.byte,
                count=cfg.batch_size,
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
                        count=cfg.batch_size,
                    ),
                )
            )
    corr_cnt = np.count_nonzero(eval_results == 0)
    # print(f"Number of bugs: {n_bugs}")
    # print(
    #     f"Result: {corr_cnt} successful completions in {cfg.n_trials} trials,",
    #     f"{(corr_cnt / cfg.n_trials) * 100}%"
    # )
    return (corr_cnt / cfg.n_trials) * 100

def benchmark_bugs(cfg, model, tokenizer, device, n_trials_per_task):
    # Load bugs data
    with open(cfg.bugs_data_path, "r") as f:
        bugs = json.load(f)
    eval_results = [-1] * len(bugs) * cfg.batch_size
    i = 0
    for _ in tqdm(
        range(n_trials_per_task * 2),
        desc=f"Running {cfg.batch_size} trials for each bug",
    ):
        if i == 4:
            continue
        encoding = tokenizer(
            [bugs[i]["prompt"][:-6]], return_tensors="pt", padding=True, truncation=True
        ).to(device)
        token_len = encoding["input_ids"].shape[1]
        with torch.inference_mode():
            tokens = model.generate(
                **encoding,
                do_sample=True,
                num_return_sequences=cfg.batch_size,
                temperature=cfg.temp,
                max_length=token_len + cfg.gen_max_len,
                top_p=cfg.top_p,
                pad_token_id=cfg.pad_token,
            )

            texts = tokenizer.batch_decode(tokens)

        end_of_diff = re.compile("\n[^ +-@]+")
        section_names = set(["name", "file", "message", "diff"])
        for j in range(len(texts)):
            # split the diff text according to <NME>, <BEF>, <MSG>, <DFF>.
            parsed = split_diff(texts[j])
            # truncate the diff hunk at the first line not starting with " ",
            # "+", "-", or "@".
            if parsed and all((s in parsed for s in section_names)):
                diff_hunk = end_of_diff.split(parsed["diff"])[0]
                nme_idx = diff_hunk.find("<NME>")
                if nme_idx != -1:
                    diff_hunk = diff_hunk[:nme_idx]
                res = apply_diff(bugs[i]["prompt_code"], diff_hunk)
                if res == bugs[i]["correct_code"]:
                    eval_results[i * cfg.batch_size + j] = 0
                else:
                    eval_results[i * cfg.batch_size + j] = 1
        i += 1
        if i == n_trials_per_task:
            i = 500

    corr_cnt = np.count_nonzero(np.asarray(eval_results) == 0)
    # print(
    #     f"Result: {corr_cnt} successful completions in {n_trials_per_task * cfg.batch_size} trials,",
    #     f"{(corr_cnt / (len(bugs) * cfg.batch_size)) * 100}%",
    # )
    return (corr_cnt / (len(bugs) * cfg.batch_size)) * 100

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
    device = torch.device("cuda" if cfg.cuda else "cpu")
    config = AutoConfig.from_pretrained(cfg.model)
    # Sometimes our model just fresh came out of training. Force use_cache to be true.
    config.use_cache = True
    tokenizer = AutoTokenizer.from_pretrained(cfg.model)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = 50256

    if cfg.fp16:
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model, torch_dtype=torch.float16, low_cpu_mem_usage=True
        ).to(device)
    else:
        model = AutoModelForCausalLM.from_pretrained(cfg.model, config=config).to(
            device
        )
    #model, tokenizer = model_setup(cfg)
    if "parity" in cfg.tasks:
        results = {}
        for i in tqdm(range(1, 6), desc="Evaluating bugs on parity:"):
                results[i] = max([run_benchmark(cfg, model, tokenizer, device, i, temp) for temp in (0.7,0.8,0.9)])
        print(results)

if __name__ == "__main__":
    main()
