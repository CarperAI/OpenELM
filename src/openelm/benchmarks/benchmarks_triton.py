"""Benchmark for OpenELM on Triton."""
import functools
import multiprocessing as mp

import hydra
import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm

from openelm.codegen import truncate
from openelm.codegen.codegen_triton import sample_triton, setup_triton
from openelm.constants import SRC_PATH
from openelm.utils.code_eval import eval_completions, mutate_code


def run_benchmark(cfg):
    cg_triton, tokenizer = setup_triton(cfg)
    mutated_str = mutate_code(n_bugs=cfg.n_bugs, task=cfg.tasks[0])
    mutated_encoding = tokenizer(
        [mutated_str],
        truncation=True,
        padding=True,
        max_length=2048,
        return_tensors="np",
    )
    # Triton eats numpy arrays
    input_ids_len = mutated_encoding.input_ids.shape
    text = []
    for i in range(input_ids_len[1]):
        text.append(tokenizer.batch_decode(mutated_encoding.input_ids[:, i]))
    num_batches = cfg.n_trials // cfg.batch_size
    eval_results = []
    ev_func = functools.partial(
        eval_completions, task=cfg.tasks[0], timeout=cfg.timeout
    )
    for i in tqdm(range(num_batches), desc=f"Running benchmark with {cfg.n_bugs} bugs"):

        completions = sample_triton(
            cfg, cg_triton, tokenizer, mutated_encoding, add_def=True
        )
        truncations = list(map(truncate, completions))
        # Run evaluation in separate processes
        with mp.Pool(processes=cfg.batch_size) as pool:
            eval_results.extend(list(pool.map(ev_func, truncations)))

    eval_results = np.array(eval_results)
    corr_cnt = np.count_nonzero(eval_results == 0)
    print(f"Number of bugs: {cfg.n_bugs}")
    print(
        f"Result: {corr_cnt} successful completions in {cfg.n_trials} trials, {(corr_cnt / cfg.n_trials) * 100}%"
    )


# Load hydra config from yaml files and command line arguments.
@hydra.main(
    config_path=str(SRC_PATH / "config"),
    config_name="benchmark_cfg_triton",
    version_base="1.2",
)
def main(cfg):
    print("----------------- Config ---------------")
    print(OmegaConf.to_yaml(cfg))
    print("-----------------  End -----------------")
    run_benchmark(cfg)


if __name__ == "__main__":
    main()
