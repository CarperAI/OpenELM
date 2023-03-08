import logging
import pathlib
import requests
import time
import json
from collections import Counter

from openelm.environments import p3_long_init_args, p3_med_init_args, P3Problem
from openelm.mutation_model import DiffModel, MutationModel, PromptModel
from openelm.configs import P3Config
from openelm.sandbox.server.sandbox_codex_execute import ExecResult
from openelm.utils.code_eval import pass_at_k
from openelm.codegen.codegen_utilities import set_seed

import hydra
from omegaconf import OmegaConf


class P3:
    def __init__(self, cfg: P3Config) -> None:
        """
        Evaluate models on P3 dataset
        """
        self.cfg: P3Config = cfg

        # Prompt size
        if cfg.env.prompt_size == 'long':
            env_args = p3_long_init_args
        elif cfg.env.prompt_size == 'med':
            env_args = p3_med_init_args
        else:
            raise ValueError('No init args found')

        # Model
        if self.cfg.model.model_name == 'prompt':
            self.mutation_model: MutationModel = PromptModel(self.cfg.model)
        elif self.cfg.model.model_name == 'diff':
            self.mutation_model: MutationModel = DiffModel(self.cfg.model)

        self.seed = env_args["seed"]
        self.log_dir = 'logs/p3/problems'


    def run(self):
        """
        Query PromptMutationModelForP3 for solutions to programming puzzles
        """
        # Get problems
        problems = requests.get("https://raw.githubusercontent.com/microsoft/PythonProgrammingPuzzles/v0.2/puzzles/puzzles.json").json()
        run_start_time = time.time()
        num_problem_errors = 0
        for problem in problems:
            problem_start_time = time.time()
            problem_dict = {'name': problem['name']}
            logging.info(problem['name'])

            problem['problem_func'] = problem['sat'].replace('def sat(', 'def f6(') # prompt form is f6()
            problem['solution_preamble'] = problem['sol_header'].replace('def sol(', 'def g6(') # solution form is g6()
            if self.cfg.env.prompt_size == 'long':
                problem['solution_preamble'] = problem['solution_preamble'] + '\n' + problem['sol_docstring']

            env = P3Problem(seed=self.seed,
                            config=self.cfg,
                            mutation_model=self.mutation_model,
                            problem_func=problem['problem_func'],
                            solution_preamble=problem['solution_preamble'],
                            ans_type = problem['ans_type'])

            # Find solutions
            # If there is an error during finding a solution, log it and skip this problem
            solutions = []
            try:
                for i in range(self.cfg.env.solutions_per_problem // self.cfg.model.batch_size):
                    set_seed(i) # Change seed for each query

                    try:
                        solutions += env.random()
                    except Exception as e:
                        logging.error(f'ERROR with solution {i} in {problem["name"]}: {e}')
                        num_problem_errors += 1
                        raise(e)
            except Exception as e:
                continue

            # Evaluate fitness of solutions
            res_sols_list = []
            solved = False
            for sol in solutions:
                res_sol_dict = {}
                res_sol_dict['program_str'] = sol.program_str

                if isinstance(sol.result_obj, ExecResult):
                    if self.cfg.save_result_obj: res_sol_dict['result_obj'] = sol.result_obj.name
                    fitness = 0.0
                else:
                    if self.cfg.save_result_obj: res_sol_dict['result_obj'] = sol.result_obj
                    fitness = env.fitness(sol)

                res_sol_dict['fitness'] = fitness
                res_sols_list.append(res_sol_dict)
                if not solved and fitness == 1.0:
                    solved = True # just want to save if solved at all

            problem_dict['config'] = OmegaConf.to_container(self.cfg)
            problem_dict['solutions'] = res_sols_list
            problem_dict['solved'] = solved
            problem_dict['time_elapsed'] = time.time() - problem_start_time

            # Save results
            dir = f'{self.log_dir}/{problem_dict["name"]}/{run_start_time}'
            pathlib.Path(dir).mkdir(parents=True, exist_ok=True)

            with open(f'{dir}/results.json', 'w') as file:
                file.write(json.dumps(problem_dict))

        logging.info(f'Successfully ran on {len(problems)}/{len(problems)-num_problem_errors}' +
                        f' problems and saved results to {self.log_dir}')


    def eval_pass_at_k(self, timestamp: str, k: int):
        """
        pass@k metric over a subset of run logs
        
        Args:
            timestamp (str): (optional) go through all problems with a run generated with timestamp
                (if None, go through the latest run for every problem currently in logs)
            k (int): k for pass@k
        """

        path = pathlib.Path(self.log_dir)
        problem_paths = sorted(list(path.iterdir())) # Get all logged problems
        paks = []
        for p in problem_paths:
            n = 0
            c = 0
            # Select one of the runs per problem
            if len(timestamp) == 0:
                # Get latest run
                path = pathlib.Path(p)
                run_paths = sorted(list(path.iterdir())) # Get all the runs per problem
                run_path = run_paths[-1]
            else:
                # Get 'timestamp' run
                run_path = p / timestamp

            with open(f'{run_path}/results.json', 'r') as f:
                results = json.load(f)
                n += len(results['solutions'])
                c += Counter(sol['fitness'] for sol in results['solutions'])[1.0]

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
    logging.info("----------------- Config ---------------")
    logging.info(OmegaConf.to_yaml(cfg))
    logging.info("-----------------  End -----------------")
    p3 = P3(cfg)
    
    if cfg.eval_k > 0: logging.info(f"PASS@K: {p3.eval_pass_at_k(timestamp=cfg.eval_timestamp, k=cfg.eval_k)}")
    else: p3.run()


if __name__ == "__main__":
    main()
 