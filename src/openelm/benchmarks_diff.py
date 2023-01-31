"""This is the benchmark test of diff models."""
import functools
import json
import multiprocessing as mp
import re

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from openelm.constants import SRC_PATH
from openelm.utils.diff_eval import apply_diff, split_diff
from openelm.utils.code_eval import eval_completions, mutate_code


def benchmark_parity(cfg, model, tokenizer, device, n_bugs, temperature):
    # Prepare the mutated eval_results with cfg.n_bugs bugs.
    mutated_str, function_str = mutate_code(n_bugs=n_bugs, task="parity", mutate_method="diff")
    mutated_encoding = tokenizer(
        [mutated_str],
        truncation=True,
        padding=True,
        return_tensors="pt",
    ).to(device)
    token_len = mutated_encoding["input_ids"].shape[1]

    # Generate eval_results
    num_batches = cfg.n_trials // cfg.batch_size
    end_of_diff = re.compile("\n[^ +-@]+")
    eval_results = []
    for _ in tqdm(range(num_batches), desc=f"Running benchmark with {n_bugs} bugs"):
        with torch.inference_mode():
            tokens = model.generate(
                **mutated_encoding,
                do_sample=True,
                num_return_sequences=cfg.batch_size,
                temperature=temperature,
                max_length=token_len + cfg.gen_max_len,
                top_p=cfg.top_p,
                pad_token_id=cfg.pad_token,
            )
            texts = tokenizer.batch_decode(tokens)
        for text in texts:
            # split the diff text according to <NME>, <BEF>, <MSG>, <DFF>.
            parsed = split_diff(text)
            # truncate the diff hunk at the first line not starting with " ",
            # "+", "-", or "@".
            if parsed and all(
                [s in parsed for s in ["name", "file", "message", "diff"]]
            ):
                diff_hunk = end_of_diff.split(parsed["diff"])[0]
                eval_results.append(apply_diff(function_str, diff_hunk))
            else:
                # Invalid format. No patching.
                eval_results.append(function_str)

    # Evaluate the eval_results in threads (most importantly, separate
    # reliability_guard in separate
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
    # print(f"Number of bugs: {n_bugs}\n")
    # print(f"Mutated code to be fixed:\n{function_str}\n")
    # print(
    #     f"Result: {corr_cnt} successful completions in {cfg.n_trials} trials,",
    #     f"{(corr_cnt / cfg.n_trials) * 100}%",
    # )
    return (corr_cnt / cfg.n_trials) * 100


def benchmark_bugs(cfg, model, tokenizer, device, n_trials_per_task):
    # Load bugs data
    with open(cfg.bugs_data_path, "r") as f:
        bugs = json.load(f)
    eval_results = []
    i = 5
    for _ in tqdm(
        range(n_trials_per_task * 2),
        desc=f"Running {cfg.batch_size} trials for each bug",
    ):
        encoding = tokenizer(
            [bugs[i]["prompt"][:-6]],
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=2048
        ).to(device)
        token_len = encoding["input_ids"].shape[1]
        with torch.inference_mode():
            tokens = model.generate(
                **encoding,
                do_sample=True,
                num_return_sequences=cfg.batch_size,
                temperature=cfg.temp,
                max_length=min(2048, token_len + cfg.gen_max_len),
                top_p=cfg.top_p,
                pad_token_id=cfg.pad_token,
            )

            texts = tokenizer.batch_decode(tokens)

        end_of_diff = re.compile("\n[^ +-@]+")
        section_names = {"name", "file", "message", "diff"}
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
                    eval_results.append(0)
                else:
                    eval_results.append(1)
        i += 1
        if i == n_trials_per_task:
            i = 500

    corr_cnt = np.count_nonzero(np.asarray(eval_results) == 0)
    print(
        f"Result: {corr_cnt} successful completions in {(len(eval_results))} trials,",
        f"{(corr_cnt / (len(eval_results))) * 100}%",
    )
    return (corr_cnt / (len(eval_results))) * 100


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
        model = AutoModelForCausalLM.from_pretrained(cfg.model, config=config).to(
            device
        )
    if "parity" in cfg.tasks:
        results = {}
        for i in tqdm(range(1, 6), desc="Evaluating diff model on parity:"):
                #results[i] = max([benchmark_parity(cfg, model, tokenizer, device, i, temp) for temp in (0.7,0.8,0.9)])
                results[i] = benchmark_parity(cfg, model, tokenizer, device, i, 0.8)
        print(results)
    if "bugs" in cfg.tasks:
        results = benchmark_bugs(cfg, model, tokenizer, device, cfg.n_bugs_trials)


if __name__ == "__main__":
    main()
