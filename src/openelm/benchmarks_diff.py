"""This is the benchmark test of diff models."""
import functools
import itertools
import json
import multiprocessing as mp
import re
from typing import Union

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from openelm.benchmarks import eval_code_string, parity_reference
from openelm.constants import SRC_PATH
from openelm.utils.diff_eval import apply_diff, split_diff

parity_test_data = {
    i: parity_reference(*i) for i in itertools.product(range(2), repeat=4)
}


def eval_completions(
    eval_results: Union[str, list[str]], task: str = "parity", timeout: int = 5
) -> Union[int, list[int]]:
    """
    Evaluate (a batch of) the modified eval_results on a task.

    Args:
        eval_results: either a string or a batch of strings. The code(s) to be evaluated.
        task: (Optional) the task to be performed.
        timeout: (Optional) the timeout (in seconds).

    Returns:
        either the status code of the string or a list of status eval_results of the batch
        of strings (depending on whether `eval_results` is batched).
    """
    if task == "parity":
        _eval_results = eval_results if isinstance(eval_results, list) else [eval_results]
        results = []
        for code in _eval_results:
            results.append(eval_code_string(code, parity_test_data, timeout))
        if isinstance(eval_results, list):
            # Batch evaluation returns the batch
            return results
        else:
            # Single evaluation returns a single result
            return results[0]
    else:
        raise ValueError(f"Unknown task: {task}")


def mutate_code(n_bugs: int = 5, task: str = "parity") -> tuple:
    """
    Mutate code to create n bugs. Output the prompt in diff format.

    Args:
        n_bugs: number of bugs to introduce (from 1 to 5).
        task: (Optional) the task to be performed.

    Returns:
        mutated_code, function_string
    """
    mutation_template = [
        f"<NME> {task}.py\n<BEF> ",
        "",  # placeholder for the context, e.g., the buggy code
        "\n<MSG> Fixed bugs",
    ]
    if task == "parity":
        variables = ["b", "b", "b", "b", 2]
        for i in range(n_bugs):
            variables[i] = "c" if i < 4 else 3
        func_str = (
            'def parity(b1,b2,b3,b4):\n    """Return binary parity of a sequence of input bits.'
            ' Return 0 for even parity, 1 for odd parity."""\n    bit_sum = sum(['
            "{}1,{}2,{}3,{}4])\n    return bit_sum % {}".format(*variables)
        )
        mutation_template[1] = func_str
        return "".join(mutation_template), func_str
    else:
        raise ValueError(f"Unknown task: {task}")


def benchmark_parity(cfg, model, tokenizer, device):
    # Prepare the mutated eval_results with cfg.n_bugs bugs.
    mutated_str, function_str = mutate_code(n_bugs=cfg.n_bugs, task="parity")
    mutated_encoding = tokenizer(
        [mutated_str],
        return_tensors="pt",
    ).to(device)
    token_len = mutated_encoding["input_ids"].shape[1]

    # Generate eval_results
    num_batches = cfg.n_trials // cfg.batch_size
    end_of_diff = re.compile("\n[^ +-@]+")
    eval_results = []
    for _ in tqdm(range(num_batches), desc=f"Running benchmark with {cfg.n_bugs} bugs"):
        with torch.inference_mode():
            tokens = model.generate(
                **mutated_encoding,
                do_sample=True,
                num_return_sequences=cfg.batch_size,
                temperature=cfg.temp,
                max_length=token_len + cfg.gen_max_len,
                top_p=cfg.top_p,
                pad_token_id=cfg.pad_token,
            )
            texts = tokenizer.batch_decode(tokens)
        for text in texts:
            # split the diff text according to <NME>, <BEF>, <MSG>, <DFF>.
            parsed = split_diff(text)
            # truncate the diff hunk at the first line not starting with " ", "+", "-", or "@".
            if parsed and all(
                [s in parsed for s in ["name", "file", "message", "diff"]]
            ):
                diff_hunk = end_of_diff.split(parsed["diff"])[0]
                eval_results.append(apply_diff(function_str, diff_hunk))
            else:
                # Invalid format. No patching.
                eval_results.append(function_str)

    # Evaluate the eval_results in threads (most importantly, separate reliability_guard in separate
    # threads as it would otherwise disable things in this current thread).
    # We apply diff patch loosely:
    #   1. it ignores the line numbers;
    #   2. it ignores invalid lines (not starting with " ", "+" or "-" and not being "@@ ... @@").
    with mp.Pool(processes=cfg.num_process) as pool:
        eval_fn = functools.partial(
            eval_completions, task="parity", timeout=cfg.timeout
        )
        results = list(pool.map(eval_fn, eval_results))

    corr_cnt = sum([r == 0 for r in results])
    print(f"Number of bugs: {cfg.n_bugs}\n")
    print(f"Mutated code to be fixed:\n{function_str}\n")
    print(
        f"Result: {corr_cnt} successful completions in {cfg.n_trials} trials, {(corr_cnt / cfg.n_trials) * 100}%"
    )


def benchmark_bugs(cfg, model, tokenizer, device):
    with open(cfg.bugs_data_path, "r") as f:
        bugs = json.load(f)
    eval_results = [-1] * len(bugs) * cfg.batch_size
    for i in tqdm(range(len(bugs)),
                  desc=f"Running {cfg.batch_size} trials for each bug"):
        encoding = tokenizer([bugs[i]['prompt'][:-6]], return_tensors="pt",
                             padding=True, truncation=True).to(device)
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
            # truncate the diff hunk at the first line not starting with " ", "+", "-", or "@".
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

    corr_cnt = np.count_nonzero(np.asarray(eval_results) == 0)
    print(
        f"Result: {corr_cnt} successful completions in {len(eval_results)} trials,",
        f"{(corr_cnt / (len(bugs) * cfg.batch_size)) * 100}%"
    )


@hydra.main(
    config_path=str(SRC_PATH / "config"),
    config_name="benchmark_diff_cfg",
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
        model = AutoModelForCausalLM.from_pretrained(cfg.model,
                                                     config=config).to(device)
    if "parity" in cfg.tasks:
        benchmark_parity(cfg, model, tokenizer, device)
    if "bugs" in cfg.tasks:
        benchmark_bugs(cfg, model, tokenizer, device)


if __name__ == "__main__":
    main()
