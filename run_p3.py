import json
import logging
import pathlib
import time
from collections import Counter

import hydra
import requests
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf

from openelm.codegen.codegen_utilities import set_seed
from openelm.configs import P3Config
from openelm.environments.p3.p3 import P3Problem, P3ProbSol
from openelm.mutation_model import MutationModel, PromptModel
from openelm.sandbox.server.sandbox_codex_execute import ExecResult
from openelm.utils.code_eval import pass_at_k

"""
Use this file to evaluate models on the P3-based environments.
See P3.run() for more info.

Usage: python run_p3.py
This defaults to solving puzzle problems.

Example usage with mutating problem+solution pairs a.k.a "probsol", along with other config changes:
python run_p3.py probsol=True model.model_path=Salesforce/codegen-2B-mono env.batch_size=8 iterations_per_puzzle=16
"""


class P3:
    def __init__(self, config: P3Config) -> None:
        """
        Evaluate models on P3 dataset
        """
        self.config: P3Config = config

        # Model
        if self.config.model.model_name == "prompt":
            self.mutation_model: MutationModel = PromptModel(self.config.model)
        # elif self.config.model.model_name == 'diff':
        #     self.mutation_model: MutationModel = DiffModel(self.config.model)

        self.log_dir = self.cfg.output_dir

    def run(self):
        """
        Query PromptModel to generate
            self.config.probsol=False: solutions to given programming puzzle problems
            self.config.probsol=True:  new problem+solution pairs
        """
        puzzles = requests.get(
            "https://raw.githubusercontent.com/microsoft/PythonProgrammingPuzzles/v0.2/puzzles/puzzles.json"
        ).json()
        run_start_time = time.time()
        for puzzle_id in self.config.starting_seeds:
            self.config.env.starting_seed = puzzle_id

            puzzle = puzzles[puzzle_id]
            puzzle_start_time = time.time()
            puzzle_dict = {"name": puzzle["name"]}
            logging.info(puzzle["name"])

            if self.config.probsol:
                env = P3ProbSol(
                    config=self.config.env, mutation_model=self.mutation_model
                )
            else:
                env = P3Problem(
                    config=self.config.env, mutation_model=self.mutation_model
                )

            # Run
            solutions = []
            assert self.config.iterations_per_puzzle >= self.config.env.batch_size
            for i in range(
                self.config.iterations_per_puzzle // self.config.env.batch_size
            ):
                set_seed(i)  # Change seed for each query

                solutions += env.random()

            # Evaluate fitness of solutions
            res_sols_list = []
            solved = False
            for sol in solutions:
                res_sol_dict = {"program_str": sol.program_str}
                if self.config.save_result_obj is not None:
                    if isinstance(sol.result_obj, ExecResult):
                        res_sol_dict["result_obj"] = sol.result_obj.name
                    else:
                        res_sol_dict["result_obj"] = sol.result_obj

                fitness = env.fitness(sol)

                res_sol_dict["fitness"] = fitness
                res_sols_list.append(res_sol_dict)
                if fitness == 1.0:
                    solved = True  # just want to save if the current problem is solved by any attempt

            puzzle_dict["config"] = OmegaConf.to_container(self.config)
            puzzle_dict["solutions"] = res_sols_list
            puzzle_dict["solved"] = solved
            puzzle_dict["time_elapsed"] = time.time() - puzzle_start_time

            # Save results
            if self.config.save_results:
                dir = f'{self.log_dir}/{puzzle_dict["name"]}/{run_start_time}'
                pathlib.Path(dir).mkdir(parents=True, exist_ok=True)

                with open(f"{dir}/results.json", "w") as file:
                    file.write(json.dumps(puzzle_dict))

        logging.info(
            f"Successfully ran on {len(self.config.starting_seeds)}"
            + f"/{len(self.config.starting_seeds)}"
            + f" puzzles and saved any results to {self.log_dir}"
        )

    def eval_pass_at_k(self, timestamp: str, k: int):
        """
        pass@k metric over a subset of run logs

        Args:
            timestamp (str): (optional) go through all puzzles with a run generated with timestamp
                (if None, go through the latest run for every puzzle currently in logs)
            k (int): k for pass@k
        """

        path = pathlib.Path(self.log_dir)
        puzzle_paths = sorted(list(path.iterdir()))  # Get all logged puzzles
        paks = []
        for p in puzzle_paths:
            n = 0
            c = 0
            # Select one of the runs per puzzle
            if len(timestamp) == 0:
                # Get latest run
                path = pathlib.Path(p)
                run_paths = sorted(list(path.iterdir()))  # Get all the runs per puzzle
                run_path = run_paths[-1]
            else:
                # Get 'timestamp' run
                run_path = p / timestamp

            with open(f"{run_path}/results.json", "r") as f:
                results = json.load(f)
                n += len(results["solutions"])
                c += Counter(sol["fitness"] for sol in results["solutions"])[1.0]

                pak = pass_at_k(n=n, c=c, k=k)
                paks.append(pak)

        pak_overall = sum(paks) / len(paks)
        return pak_overall


# Load hydra config from yaml files and command line arguments.
@hydra.main(
    config_name="p3config",
    version_base="1.2",
)
def main(cfg):
    # Run
    cfg.output_dir = HydraConfig.get().runtime.output_dir
    logging.info("----------------- Config ---------------")
    logging.info(OmegaConf.to_yaml(cfg))
    logging.info("-----------------  End -----------------")
    p3 = P3(cfg)

    if cfg.eval_k > 0:
        logging.info(
            f"PASS@K: {p3.eval_pass_at_k(timestamp=cfg.eval_timestamp, k=cfg.eval_k)}"
        )
    else:
        p3.run()


if __name__ == "__main__":
    main()
