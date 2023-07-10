import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Optional

import hydra
import numpy as np
import torch
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from tqdm import tqdm, trange

from openelm.codegen import model_setup, sample, truncate
from openelm.configs import BaseConfig
from openelm.sandbox.server.sandbox_codex_execute import ExecResult
from openelm.utils import apply_diff, eval_completions, mutate_code, split_diff

os.environ["TRANSFORMERS_CACHE"] = "/fsx/hyperion/hf_cache"


@dataclass
class BenchmarkBugsConfig(BaseConfig):
    hydra: Any = field(
        default_factory=lambda: {
            "run": {"dir": "logs/benchmarks/bugs/${hydra.job.override_dirname}"}
        }
    )
    model_path: str = "CarperAI/diff-codegen-2b-v2"
    bugs_data_path: str = "/fsx/shared/diff_benchmark.json"
    mode: str = "diff"
    seed: Optional[int] = None
    deterministic: bool = False
    fp16: bool = True
    cuda: bool = True
    debug: bool = False
    gpus: int = 1
    processes: int = 12
    temp: float = 0.8
    top_p: float = 0.95
    gen_max_len: int = 256
    batch_size: int = 32
    n_trials: int = 500
    n_bugs_trials: int = 100
    timeout: float = 5.0
    verbose: bool = True
    tasks: list[str] = field(default_factory=lambda: ["parity"])
    n_bugs: list[int] = field(default_factory=lambda: [1])
    temp_samples: list[float] = field(default_factory=lambda: [0.8])
    sweep: bool = False


class BenchmarkBugs:
    def __init__(self, cfg: BenchmarkBugsConfig):
        self.cfg: BenchmarkBugsConfig = cfg

        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        self.device = torch.device("cuda")
        self.model, self.tokenizer, self.device = model_setup(cfg, self.device)
        print("Number of parameters:", self.model.num_parameters())

    def benchmark_parity(self, n_bugs, **kwargs):
        mutated_str, function_str = mutate_code(
            n_bugs=n_bugs, task="parity", mutate_method=self.cfg.mode
        )
        print(mutated_str)
        encoding = self.tokenizer(
            [mutated_str],
            truncation=True,
            padding=True,
            return_tensors="pt",
        ).to(self.device)
        token_len: int = encoding.input_ids.shape[1]
        sample_idx: int = token_len - 16 if self.cfg.mode == "prompt" else 0
        num_batches: int = self.cfg.n_trials // self.cfg.batch_size
        results: list = []
        end_of_diff = re.compile("\n[^ +-@]+")
        for _ in trange(
            num_batches,
            desc=f"Running {self.cfg.mode} benchmark with {n_bugs} bugs",
            disable=not self.cfg.verbose,
        ):
            completions: list[str] = sample(
                encoding,
                self.cfg,
                self.model,
                self.tokenizer,
                starting_idx=sample_idx,
                num_return_sequences=1,
                **kwargs,
            )
            if self.cfg.mode == "prompt":
                eval_results: list[str] = list(map(truncate, completions))
            elif self.cfg.mode == "diff":
                eval_results = []
                for text in completions:
                    # print(text)
                    # split the diff text according to <NME>, <BEF>, <MSG>, <DFF>.
                    parsed: dict = split_diff(text)
                    # truncate the diff hunk at the first line not starting with
                    # " ", "+", "-", or "@".
                    if parsed and all(
                        (s in parsed for s in ["name", "file", "message", "diff"])
                    ):
                        diff_hunk: str = end_of_diff.split(parsed["diff"])[0]
                        # We apply diff patch loosely:
                        #   1. it ignores the line numbers;
                        #   2. it ignores invalid lines (not starting with " ",
                        #   "+" or "-" and not being "@@ ... @@").
                        eval_results.append(apply_diff(function_str, diff_hunk))
                        # print("Patched!")
                    else:
                        # Invalid format. No patching.
                        eval_results.append(function_str)
            results.extend(
                eval_completions(
                    eval_results,
                    task="parity",
                    timeout=self.cfg.timeout,
                    processes=self.cfg.processes,
                    debug=self.cfg.debug,
                )
            )
        corr_cnt = results.count(ExecResult.VALID)
        if self.cfg.verbose:
            print(f"Number of bugs: {n_bugs}\n")
            print(f"Mutated code to be fixed:\n{function_str}\n")
            print(
                f"Result: {corr_cnt} successful completions in {self.cfg.n_trials}",
                f"trials, {(corr_cnt / self.cfg.n_trials) * 100}%",
            )
        return (corr_cnt / self.cfg.n_trials) * 100

    def benchmark_bugs(self, **kwargs):
        # Load bugs data
        with open(self.cfg.bugs_data_path, "r") as f:
            bugs = json.load(f)
        if self.cfg.mode == "prompt":
            raise ValueError("Prompt mode is not yet supported for bugs benchmark.")
        results = []
        end_of_diff = re.compile("\n[^ +-@]+")
        i = 0
        num_evaluated_bugs = 0
        for _ in trange(
            self.cfg.n_bugs_trials * 2,
            desc=f"Running {self.cfg.batch_size} trials for each bug",
            disable=not self.cfg.verbose,
        ):
            if len(bugs[i]["prompt"][:-6]) > 1000:
                i += 1
                continue
            encoding = self.tokenizer(
                [bugs[i]["prompt"][:-6]],
                truncation=True,
                padding=True,
                return_tensors="pt",
            ).to(self.device)
            print(bugs[i]["prompt"][:-6])
            completions: list[str] = sample(
                encoding,
                self.cfg,
                self.model,
                self.tokenizer,
                starting_idx=0,
                num_return_sequences=1,
                **kwargs,
            )
            for _, text in enumerate(completions):
                # split the diff text according to <NME>, <BEF>, <MSG>, <DFF>.
                parsed: dict = split_diff(text)
                # truncate the diff hunk at the first line not starting with " ",
                # "+", "-", or "@".
                if parsed and all(
                    (s in parsed for s in ["name", "file", "message", "diff"])
                ):
                    diff_hunk: str = end_of_diff.split(parsed["diff"])[0]
                    nme_idx: int = diff_hunk.find("<NME>")
                    if nme_idx != -1:
                        diff_hunk = diff_hunk[:nme_idx]
                    res: str = apply_diff(bugs[i]["prompt_code"], diff_hunk)
                    print(res)
                    if res == bugs[i]["correct_code"]:
                        results.append(0)
                    else:
                        results.append(1)
                    num_evaluated_bugs += 1
            i += 1
            print(i)
            if i == self.cfg.n_bugs_trials:
                i = 500

        print(len(bugs))
        print(num_evaluated_bugs)
        corr_cnt = results.count(0)
        if self.cfg.verbose:
            print(
                f"Result: {corr_cnt} successful completions in",
                f"{num_evaluated_bugs} trials:",
                f"{(corr_cnt / num_evaluated_bugs) * 100}%",
            )
        return (corr_cnt / num_evaluated_bugs) * 100

    def run_benchmark(self, **kwargs) -> dict[str, float]:
        result_dict = {}
        if "parity" in self.cfg.tasks:
            for n_bug in self.cfg.n_bugs:
                if self.cfg.sweep:
                    temp_sweep = []
                    desc_str = f"Evaluating parity for {n_bug} bugs:"
                    for temp in tqdm(self.cfg.temp_samples, desc=desc_str):
                        acc = self.benchmark_parity(n_bugs=n_bug, temperature=temp)
                        temp_sweep.append(acc)
                    result = np.mean(temp_sweep)
                else:
                    result = self.benchmark_parity(n_bugs=n_bug)
                result_dict[str(n_bug)] = result
        if "bugs" in self.cfg.tasks:
            result_dict["bugs"] = self.benchmark_bugs(**kwargs)

        print(f"Final result: {result_dict}")
        return result_dict


cs = ConfigStore.instance()
cs.store(name="config", node=BenchmarkBugsConfig)


@hydra.main(version_base="1.2", config_name="config")
def main(cfg):
    print("----------------- Config ---------------")
    print(OmegaConf.to_yaml(cfg))
    print("-----------------  End -----------------")
    cfg = OmegaConf.to_object(cfg)
    bugs_benchmark = BenchmarkBugs(cfg)
    bugs_benchmark.run_benchmark()


if __name__ == "__main__":
    main()
