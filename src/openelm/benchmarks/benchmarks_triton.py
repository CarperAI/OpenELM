from dataclasses import dataclass, field
from typing import Any, Optional

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, OmegaConf
from tqdm import trange

from openelm.codegen import truncate
from openelm.codegen.codegen_triton import sample_triton, setup_triton
from openelm.configs import BaseConfig
from openelm.utils.code_eval import eval_completions, mutate_code


@dataclass
class BenchmarkTritonConfig(BaseConfig):
    hydra: Any = field(
        default_factory=lambda: {
            "run": {"dir": "logs/benchmarks/triton/${hydra.job.override_dirname}"}
        }
    )
    triton_host: str = MISSING
    triton_port: int = 8001
    seed: Optional[int] = None
    deterministic: bool = False
    fp16: bool = True
    cuda: bool = True
    debug: bool = False
    # gpus: int = 1
    processes: int = 1
    temp: float = 0.8
    top_p: float = 0.9
    gen_max_len: int = 128
    batch_size: int = 32
    n_trials: int = 3200
    timeout: float = 5.0  # Seconds
    n_bugs: int = 1


def run_benchmark(cfg):
    cg_triton, tokenizer = setup_triton(cfg)
    mutated_str = mutate_code(n_bugs=cfg.n_bugs, task="parity")
    encoding = tokenizer(
        [mutated_str],
        truncation=True,
        padding=True,
        max_length=2048,
        return_tensors="np",
    )
    # Triton eats numpy arrays
    num_batches: int = cfg.n_trials // cfg.batch_size
    eval_results: list = []
    for _ in trange(
        num_batches,
        desc=f"Running benchmark with {cfg.n_bugs} bugs",
        disable=not cfg.verbose,
    ):
        completions = sample_triton(encoding, cfg, cg_triton, tokenizer, add_def=True)
        truncations = map(truncate, completions)
        eval_results.extend(
            eval_completions(
                truncations,
                task="parity",
                timeout=cfg.timeout,
                processes=cfg.processes,
                debug=cfg.debug,
            )
        )

    corr_cnt = eval_results.count(0)
    print(f"Number of bugs: {cfg.n_bugs}")
    print(
        f"Result: {corr_cnt} successful completions in {cfg.n_trials} trials, ",
        f"{(corr_cnt / cfg.n_trials) * 100}%",
    )


cs = ConfigStore.instance()
cs.store(name="config", node=BenchmarkTritonConfig)


@hydra.main(version_base="1.2", config_name="config")
def main(cfg):
    print("----------------- Config ---------------")
    print(OmegaConf.to_yaml(cfg))
    print("-----------------  End -----------------")
    cfg = OmegaConf.to_object(cfg)
    run_benchmark(cfg)


if __name__ == "__main__":
    main()
