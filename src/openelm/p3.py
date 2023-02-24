import pathlib
import requests
import time
import json

from openelm.environments import p3_long_init_args, p3_med_init_args, P3Problem
from openelm.configs import P3ELMConfig
from codegen.codegen_utilities import set_seed
from openelm.sandbox.server.sandbox_codex_execute import ExecResult

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
cs = ConfigStore.instance()
cs.store(name="config", node=P3ELMConfig)

class P3:
    def __init__(self, cfg, diff_model_cls=None, env_args: dict = None) -> None:
        self.cfg = cfg

        # Get the defaults if `env_args` is not specified.
        if env_args is None:
            if cfg['prompt_size'] == 'long':
                env_args = p3_long_init_args
            elif cfg['prompt_size'] == 'med':
                env_args = p3_med_init_args
            else:
                raise ValueError('no init args found')

        env_args["config"] = self.cfg  # Override default environment config

        # Override diff model if `diff_model_cls` is specified.
        if diff_model_cls is not None:
            self.diff_model = diff_model_cls(self.cfg)
            env_args = {**env_args, "diff_model": self.diff_model}
        else:
            self.diff_model = None

        self.seed = env_args["seed"]


    def run(self):
        """
        Query PromptMutationModelForP3 for solutions to programming puzzles
        """
        # res_list = []
        problems = requests.get("https://raw.githubusercontent.com/microsoft/PythonProgrammingPuzzles/v0.2/puzzles/puzzles.json").json()

        run_start_time = time.time()
        num_problems_solved = 0
        for problem in problems:
            problem_start_time = time.time()
            problem_dict = {}
            problem_dict['name'] = problem['name']
            print(problem['name'])

            problem['problem_func'] = problem['sat'].replace('def sat(', 'def f6(') # prompt form is f6()
            problem['solution_preamble'] = problem['sol_header'].replace('def sol(', 'def g6(') # solution form is g6()
            if self.cfg['prompt_size'] == 'long':
                problem['solution_preamble'] = problem['solution_preamble'] + '\n' + problem['sol_docstring']


            env = P3Problem(seed=self.seed,
                            config=self.cfg,
                            diff_model=self.diff_model,
                            problem_func=problem['problem_func'],
                            solution_preamble=problem['solution_preamble'])

            solutions = []
            for i in range(self.cfg['solutions_per_problem'] // self.cfg['batch_size']):
                set_seed(i) # Change seed for each query (need this?)
                solutions += env.random()

            res_sols_list = []
            solved = False
            for sol in solutions:
                res_sol_dict = {}
                res_sol_dict['program_str'] = sol.program_str

                if isinstance(sol.result_obj, ExecResult):
                    # res_sol_dict['result_obj'] = sol.result_obj.name
                    fitness = 0.0
                else:
                    # res_sol_dict['result_obj'] = sol.result_obj
                    fitness = env.fitness(sol)

                res_sol_dict['fitness'] = fitness
                res_sols_list.append(res_sol_dict)
                if not solved and fitness == 1.0:
                    solved = True # just want to count num problems solved
                    num_problems_solved += 1

            problem_dict['config'] = OmegaConf.to_container(self.cfg)
            problem_dict['solutions'] = res_sols_list
            problem_dict['solved'] = solved
            problem_dict['time_elapsed'] = time.time() - problem_start_time
            # res_list.append(problem_dict)

            # Save results
            dir = f'logs/elm/p3/{problem_dict["name"]}/{run_start_time}'
            pathlib.Path(dir).mkdir(parents=True, exist_ok=True)

            with open(f'{dir}/results.json', 'w') as file:
                file.write(json.dumps(problem_dict))


        # res = {}
        # res['config'] = OmegaConf.to_container(self.cfg)
        # res['num_problems'] = len(problems)
        # res['num_problems_solved'] = num_problems_solved
        # res['time_elapsed'] = time.time() - run_start_time
        # res['problems'] = res_list

        return

# Load hydra config from yaml files and command line arguments.
@hydra.main(
    config_name="config",
    version_base="1.2",
)
def main(cfg):
    # Run
    print("----------------- Config ---------------")
    print(OmegaConf.to_yaml(cfg))
    print("-----------------  End -----------------")
    p = P3(cfg)
    p.run()


if __name__ == "__main__":
    main()
 