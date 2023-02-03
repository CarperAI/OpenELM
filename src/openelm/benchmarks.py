import functools
import itertools
import json
import os
import re
from typing import Iterator

import hydra
import numpy as np
import multiprocessing as mp
from omegaconf import OmegaConf
from openelm.utils.diff_eval import split_diff, apply_diff
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
import torch

from openelm.codegen.codegen_utilities import truncate
from openelm.constants import SRC_PATH
from openelm.utils.code_eval import mutate_code, eval_completions


def run_benchmark(cfg, model, tokenizer, device, n_bugs, temperature):
    mutated_str = mutate_code(n_bugs=n_bugs, task=cfg.tasks[0], mutate_method='prompt')
    # mutated_encoding = tokenizer([mutated_str] * cfg.gpus, truncation=True, padding=True,
    mutated_encoding = tokenizer(
        [mutated_str],
        truncation=True,
        padding=True,
        return_tensors="pt",
    ).to(device)
    token_len = mutated_encoding.input_ids.shape[1]
    num_batches = cfg.n_trials // cfg.batch_size
    eval_results = []
    ev_func = functools.partial(eval_completions, task=cfg.tasks[0], timeout=cfg.timeout)
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
        truncations = list(map(truncate, text))
        # Run evaluation in separate processes
        with mp.Pool(processes=cfg.batch_size) as pool:
            eval_results.extend(list(pool.map(ev_func, truncations)))

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
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # Prevent annoying warning from tokenizers on multiple threads.
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
