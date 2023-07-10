import functools
import json
import os
import time
from dataclasses import asdict, dataclass, field
from itertools import permutations
from pathlib import Path
from typing import Any, Optional, Union

import hydra
import numpy as np
import torch
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from tqdm import trange

from openelm.algorithms.map_elites import MAPElites
from openelm.codegen import model_setup, sample, truncate
from openelm.configs import BaseConfig, MAPElitesConfig
from openelm.environments.sodaracer import (
    CIRCLE,
    CPPN_FIXED,
    CPPN_MUTABLE,
    GALLOPER,
    GALLOPER_PREREQ,
    IMPORTS,
    QUERY_CPPN,
    RADIAL,
    RUNNER,
    SEEDS_DICT,
    SQUARE,
    SQUARE_PREREQ,
    WHEEL,
)
from openelm.environments.sodaracer.sodarace import Sodarace, Sodaracer
from openelm.environments.sodaracer.walker import Walker
from openelm.utils.code_eval import pool_exec_processes

INSTRUCTIONS = {
    0: "",
    1: "def make_walker():\n",
    2: "#Create a new walker by modifying the starting function above.\ndef make_walker():\n",
    3: "#Combine the ,starting programs above to make a new program.\ndef make_walker():\n",
}


@dataclass
class BenchmarkCrossoverConfig(BaseConfig):
    hydra: Any = field(
        default_factory=lambda: {
            "run": {"dir": "logs/benchmarks/crossover/${hydra.job.override_dirname}"}
        }
    )
    model: str = "Salesforce/codegen-2B-mono"
    save_path: str = "/fsx/home-hyperion/OpenELM/data"
    seed: Optional[int] = None
    deterministic: bool = False
    fp16: bool = True
    cuda: bool = True
    debug: bool = False
    gpus: int = 1
    processes: int = 1
    temp: float = 0.85
    top_p: float = 0.95
    gen_max_len: int = 512
    batch_size: int = 16
    n_trials: int = 1000
    eval_ms: int = 1000  # Milliseconds
    timeout: float = 5.0  # Seconds
    run_name: Optional[str] = None
    seeds: list[str] = field(default_factory=lambda: ["square", "radial", "cppn_fixed"])
    # Instruction = 0: No instruction
    # Instruction = 1: "def make_walker():\n"
    # Instruction = 2: "#Create a new walker by modifying the
    # starting function above.\ndef make_walker():\n"
    # Instruction = 3: "#Combine the {seed_one}, {seed_two}, {seed_three}
    # starting programs above to make a new program\ndef make_walker():\n"
    instruction: int = 0


class CrossoverBenchmark:
    def __init__(self, cfg: BenchmarkCrossoverConfig):
        self.cfg: BenchmarkCrossoverConfig = cfg
        self.reverse_seeds: dict[str, str] = {v: k for k, v in SEEDS_DICT.items()}

        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        self.device = torch.device("cuda")
        self.model, self.tokenizer, self.device = model_setup(cfg, self.device)

    def construct_prompt(self, seeds):
        prompt_str: str = IMPORTS
        seeds = [SEEDS_DICT[seed] for seed in seeds]
        if SQUARE in seeds:
            prompt_str += SQUARE_PREREQ
        if GALLOPER in seeds:
            prompt_str += GALLOPER_PREREQ
        if RADIAL in seeds or WHEEL in seeds:
            prompt_str += CIRCLE
        if CPPN_FIXED in seeds or CPPN_MUTABLE in seeds or RUNNER in seeds:
            prompt_str += QUERY_CPPN
        import_str: str = prompt_str
        if self.cfg.instruction == 3:
            instruction_str: str = INSTRUCTIONS[self.cfg.instruction].split(",")[0]
        for seed in seeds:
            prompt_str += seed
            if self.cfg.instruction == 3:
                instruction_str += self.reverse_seeds[seed] + ", "
        if self.cfg.instruction == 3:
            instruction_str += INSTRUCTIONS[self.cfg.instruction].split(",")[1]
        else:
            instruction_str = INSTRUCTIONS[self.cfg.instruction]
        if self.cfg.instruction != 0:
            import_str += instruction_str
        prompt_str += instruction_str
        return prompt_str, import_str

    def benchmark_seeds(self, seeds):
        prompt, preamble = self.construct_prompt(seeds)
        encoding = self.tokenizer(
            [prompt],
            truncation=True,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        sodarace_env = Sodarace(
            seed=SEEDS_DICT["square"],
            config=self.cfg,
            diff_model=self.model,
            eval_ms=self.cfg.eval_ms,
        )
        map_elites = MAPElites(
            env=sodarace_env, config=MAPElitesConfig(map_grid_size=(12,))
        )

        results: list[int] = []
        valid_fitnesses: list[float] = []
        local_scope_exec: bool = self.cfg.instruction != 0
        token_len: int = encoding.input_ids.shape[1]
        print("Benchmarking seeds: ", ", ".join(seeds))
        print(f"Prompt length: {token_len} tokens.")

        for _ in trange(self.cfg.n_trials // self.cfg.batch_size):
            completions: list[str] = sample(
                encoding,
                self.cfg,
                self.model,
                self.tokenizer,
            )
            trunc = functools.partial(truncate, only_local_scope=local_scope_exec)
            truncations: list[str] = list(
                preamble + trunc for trunc in map(trunc, completions)
            )
            execution_results = pool_exec_processes(
                truncations,
                func_name="make_walker",
                timeout=self.cfg.timeout,
                processes=self.cfg.processes,
                debug=self.cfg.debug,
            )
            for i, result in enumerate(execution_results):
                try:
                    if isinstance(result, Walker) and result.validate():
                        sodaracer = Sodaracer(
                            program_str=truncations[i],
                            result_obj=result.to_dict(),
                        )
                        if sodaracer.valid:
                            fitness: float = sodarace_env.fitness(sodaracer)
                            if fitness is not None and fitness < 4000:
                                valid_fitnesses.append(fitness)
                                map_idx = map_elites.to_mapindex(
                                    sodaracer.to_phenotype()
                                )
                                results.append(1)
                                if fitness > map_elites.fitnesses[map_idx]:
                                    map_elites.fitnesses[map_idx] = fitness
                    else:
                        if self.cfg.debug:
                            print("Failed execution, type:", result)
                            print(truncations[i])
                        results.append(result)
                except Exception as e:
                    if self.cfg.debug:
                        print(type(e), e)
                    results.append(6)

        result_dict: dict[str, Union[list, float, int]] = {
            "valid_rate": (results.count(1) / len(results)) * 100,
            "qd_score": map_elites.qd_score(),
            "niches_filled": map_elites.niches_filled(),
            "valid_fitnesses": valid_fitnesses,
        }

        print(f"Valid rate for {seeds}: {result_dict['valid_rate']}%")
        print(f"QD score: {result_dict['qd_score']}")
        print(f"Niches filled: {result_dict['niches_filled']}")
        print(f"Average fitness: {np.nanmean(valid_fitnesses)}")
        return result_dict

    def run_benchmark(self):
        perm: list[tuple] = list(permutations(self.cfg.seeds))
        valid_rates, qd_scores, niches, all_fitnesses = [], [], [], {}
        print("Permutations: ", perm)
        # TODO: add an option to disable permutations and instead sample
        # from each permutation randomly
        for seeds in reversed(perm):
            perm_results: dict = self.benchmark_seeds(seeds)
            valid_rates.append(perm_results["valid_rate"])
            qd_scores.append(perm_results["qd_score"])
            niches.append(perm_results["niches_filled"])
            all_fitnesses[", ".join(seeds)] = perm_results["valid_fitnesses"]

        valid_stats = (np.nanmean(valid_rates), np.nanstd(valid_rates))
        qd_stats = (np.nanmean(qd_scores), np.nanstd(qd_scores))
        niche_stats = (np.nanmean(niches), np.nanstd(niches))

        print(f"Validity stats: {valid_stats[0]:.2f}, {valid_stats[1]:.2f}")
        print(f"QD stats: {qd_stats[0]:.2f}, {qd_stats[1]:.2f}")
        print(f"Niche stats: {niche_stats[0]:.2f}, {niche_stats[1]:.2f}")
        results_dct = {
            "rates": valid_rates,
            "fitnesses": all_fitnesses,
            "qd_scores": qd_scores,
            "niches": niches,
            "valid_stats": valid_stats,
            "qd_stats": qd_stats,
            "niche_stats": niche_stats,
            "config": asdict(self.cfg),
            "permutations": perm,
        }

        Path(self.cfg.save_path, f"{time.strftime('%Y%m%d-%H%M%S')}.json").write_text(
            json.dumps(results_dct)
        )


cs = ConfigStore.instance()
cs.store(name="config", node=BenchmarkCrossoverConfig)


@hydra.main(version_base="1.2", config_name="config")
def main(cfg):
    print("----------------- Config ---------------")
    print(OmegaConf.to_yaml(cfg))
    print("-----------------  End -----------------")
    cfg = OmegaConf.to_object(cfg)
    crossover = CrossoverBenchmark(cfg)
    crossover.run_benchmark()


if __name__ == "__main__":
    main()
