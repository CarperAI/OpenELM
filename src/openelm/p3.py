import argparse
import yaml
import requests
import time
import json

from openelm.environments import p3_init_args, P3Problem

class P3:
    def __init__(self, cfg, diff_model_cls=None, env_args: dict = None) -> None:
        self.cfg = cfg

        # Get the defaults if `env_args` is not specified.
        if env_args is None:
            env_args = p3_init_args
        env_args["config"] = self.cfg  # Override default environment config

        # Override diff model if `diff_model_cls` is specified.
        if diff_model_cls is not None:
            self.diff_model = diff_model_cls(self.cfg)
            env_args = {**env_args, "diff_model": self.diff_model}
        else:
            self.diff_model = None

        self.seed = env_args["seed"]


    def run(self):
        res_list = []
        problems = requests.get("https://raw.githubusercontent.com/microsoft/PythonProgrammingPuzzles/v0.2/puzzles/puzzles.json").json()
        problems = problems[:25]

        num_problems_solved = 0
        for problem in problems:
            res_dict = {}
            res_dict['name'] = problem['name']
            print(problem['name'])

            problem['problem_func'] = problem['sat'].replace('def sat(', 'def f6(') # prompt form is f6()
            problem['solution_preamble'] = problem['sol_header'].replace('def sol(', 'def g6(') # solution form is g6()

            env = P3Problem(seed=self.seed,
                            config=self.cfg,
                            diff_model=self.diff_model,
                            problem_func=problem['problem_func'],
                            solution_preamble=problem['solution_preamble'])

            solutions = env.random()
            res_sols_list = []
            solved = False
            for sol in solutions:
                res_sol_dict = {}
                res_sol_dict['program_str'] = sol.program_str
                res_sol_dict['result_obj'] = sol.result_obj
                res_sol_dict['error_code'] = sol.error_code

                fitness = env.fitness(sol)
                res_sol_dict['fitness'] = fitness
                res_sols_list.append(res_sol_dict)
                if not solved and fitness == 1.0:
                    solved = True # just want to count num problems solved
                    num_problems_solved += 1

            res_dict['solutions'] = res_sols_list
            res_list.append(res_dict)

        res = {}
        res['problems'] = res_list
        res['num_problems'] = len(problems)
        res['num_problems_solved'] = num_problems_solved

        return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='p3')
    args = parser.parse_args()

    # Run
    with open('config/elm_p3_cfg.yaml', 'r') as file:
        cfg = yaml.safe_load(file)
        p = P3(cfg)
        res = p.run()

    # Save results
    with open(f'logs/{time.time()}.json', 'w') as file:
        file.write(json.dumps(res))
 