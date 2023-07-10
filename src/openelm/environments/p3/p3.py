import json
import re
import warnings
from typing import Optional, Union

import numpy as np
import requests
from openai.embeddings_utils import cosine_similarity, get_embedding
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from transformers import pipeline

from openelm.configs import P3ProblemEnvConfig, P3ProbSolEnvConfig
from openelm.environments.base import BaseEnvironment, Genotype, Phenotype
from openelm.environments.p3 import (
    P3_IMPORTS,
    P3_PROBLEM_LONG_SEED,
    P3_PROBLEM_MED_SEED,
    P3_PROBSOL_LONG_SEED,
    P3_PROBSOL_MED_SEED,
)
from openelm.mutation_model import MutationModel
from openelm.sandbox.server.sandbox_codex_execute import ExecResult
from openelm.utils.code_eval import pass_at_k, pool_exec_processes, type_check


class P3Solution(Genotype):
    def __init__(self, program_str: str, result_obj: dict, config: P3ProblemEnvConfig):
        """
        Genotype for a programming puzzle solution.
        Args:
            program_str: the solution program string (the g6() function).
            result_obj: dict.
            config: environment config
        """
        self.program_str = program_str
        self.result_obj = result_obj
        self.config = config

        # When comparing for phenotype, just use import statement and new solution function
        baseline = '''from typing import List

def g1():
    """Find a string that when concatenated onto 'Hello ' gives 'Hello world'."""
    return "world")'''
        self.baseline_emb = np.array(
            get_embedding(baseline, engine=self.config.embedding_model_path)
        )

        if self.config.embedding_model_type == "hf":
            self.pl = pipeline(
                "feature-extraction", model=self.config.embedding_model_path
            )
            seed_features = np.array(self.pl(baseline))
            self.scaler = StandardScaler()
            seed_features_scaled = self.scaler.fit_transform(np.squeeze(seed_features))
            self.pca = PCA(0.95)
            self.pca.fit(seed_features_scaled)

    def to_phenotype(self) -> Optional[Phenotype]:
        if self.config.embedding_model_type == "openai":
            compare_str = self.program_str
            i_assert = compare_str.find("assert")
            if i_assert > -1:
                compare_str = compare_str[:i_assert]
            emb = np.array(
                get_embedding(compare_str, engine=self.config.embedding_model_path)
            )
            return cosine_similarity(emb, self.baseline_emb)
        elif self.config.embedding_model_type == "hf":
            features = np.array(self.pl(self.program_str))
            features_scaled = self.scaler.transform(np.squeeze(features))
            pca_features = self.pca.transform(features_scaled)
            return pca_features.max(axis=0).flatten()

    def __str__(self) -> str:
        return self.program_str

    def __getstate__(self):
        state = self.__dict__.copy()
        if "pl" in state:
            del state["pl"]
        if "scaler" in state:
            del state["scaler"]
        if "pca" in state:
            del state["pca"]
        return state


class P3Problem(BaseEnvironment[P3Solution]):
    def __init__(
        self,
        config: P3ProblemEnvConfig,
        mutation_model: MutationModel,
        problem_str: str = None,
        solution_preamble: str = None,
    ) -> None:
        """
        The objective is to generate solutions to a given programming puzzle problem.
        Args:
            seed: the seed dict.
            config: the config file path or dict.
            mutation_model: the diff model (or alternatives).
            problem_str: an optional puzzle problem
            solution_preamble: accompanies optional problem_str
        """
        self.mutation_model = mutation_model
        self.config = config
        self.batch_size = self.config.batch_size
        self.seed_index = self.config.starting_seed
        self.rng = None

        if self.config.prompt_size == "long":
            self.prompt_seed = P3_PROBLEM_LONG_SEED
        elif self.config.prompt_size == "med":
            self.prompt_seed = P3_PROBLEM_MED_SEED
        else:
            raise ValueError("No seed string found")

        # Get info for the puzzle that will be solved
        if problem_str is None:
            # This puzzle is at the index of the puzzles array specified by self.seed_index
            puzzles = requests.get(
                "https://raw.githubusercontent.com/microsoft/PythonProgrammingPuzzles/v0.2/puzzles/puzzles.json"
            ).json()
            puzzle = puzzles[self.seed_index]

            self.problem_func = puzzle["sat"].replace(
                "def sat(", "def f6("
            )  # prompt form is f6()
            self.solution_preamble = puzzle["sol_header"].replace(
                "def sol(", "def g6("
            )  # solution form is g6()
            if self.config.prompt_size == "long":
                self.solution_preamble += (
                    "\n" + puzzle["sol_docstring"]
                )  # add in the docstring
            self.ans_type = puzzle["ans_type"]
        else:
            self.problem_func = problem_str
            self.solution_preamble = solution_preamble
            # TODO: generate a docstring?
            self.ans_type = None

        # Use the first example in the prompt seed as basis for embedding sizes
        i_first = self.prompt_seed.find("assert")
        first_example = self.prompt_seed[:i_first].strip()

        if self.config.embedding_model_type == "openai":
            self.genotype_ndim: int = 1
            self.genotype_space = np.repeat([[0, 1]], self.genotype_ndim, axis=0).T
        elif self.config.embedding_model_type == "hf":
            # Dummy to get behavior space shape
            dummy_pl = pipeline(
                "feature-extraction", model=self.config.embedding_model_path
            )
            dummy_scaler = StandardScaler()
            dummy_features = np.array(dummy_pl(first_example))
            dummy_features_scaled = dummy_scaler.fit_transform(
                np.squeeze(dummy_features)
            )
            dummy_pca = PCA(0.95)
            dummy_pca_features = dummy_pca.fit_transform(
                np.squeeze(dummy_features_scaled)
            )
            self.genotype_ndim: int = dummy_pca_features.shape[-1]
            self.genotype_space = np.repeat([[-20, 20]], self.genotype_ndim, axis=0).T

    def get_rng_state(self) -> Optional[np.random._generator.Generator]:
        warnings.warn("WARNING: rng state not used in this environment")
        return None

    def set_rng_state(self, rng_state: Optional[np.random._generator.Generator]):
        warnings.warn("WARNING: rng state not used in this environment")
        pass

    def construct_prompt(
        self, code_batch: Optional[Union[list[str], str]] = None
    ) -> dict[str, str]:
        prompt_str = self.prompt_seed

        prompt_str += f"\n\n{self.problem_func}"  # add this particular problem, f6(), to the prompt
        if code_batch is None:
            prompt_str += "\n"
        else:
            prompt_str += (
                "\n\n# Old version of g6()\n# TODO: fix bugs in the code below\n"
            )
            if isinstance(code_batch, list):
                # TODO: get nearby genotypes
                prompt_str += code_batch[0]
            elif isinstance(code_batch, str):
                prompt_str += code_batch

            prompt_str += "\n\n# Fixed version of g6()"

        prompt_str += f"\n{self.solution_preamble}"

        template = f"{P3_IMPORTS}\n{self.solution_preamble}"
        return {"prompt": prompt_str, "template": template}

    def generate_programs(self, code_batch: list[str]) -> list[P3Solution]:
        """Generate new programs with a mutation model and evaluate them."""
        local_scope_exec = True
        generated_programs = self.mutation_model.generate_programs(
            code_batch, local_scope_exec, do_trunc=False
        )

        for i, gp in enumerate(generated_programs):
            i_assert = gp.find("assert")
            generated_programs[i] = gp[:i_assert].strip()

        if self.config.sandbox:
            results = []
            for code in generated_programs:
                resp = requests.post(
                    f"{self.sandbox_server}/eval_p3_solution",
                    json={"code": code, "timeout": self.config.timeout},
                    timeout=self.config.timeout,
                )
                if resp.status_code == 200:
                    return_dict = json.loads(resp.text)
                    results.append(return_dict)
        else:
            # TODO: handle (probably inside of pool_exec_processes) all cases
            # where the generated code returns a generator type. The multithreaded
            # execution pickles things and generators can't be pickled which
            # causes the whole thing to error out.
            # For now, try/except and re-try.
            try:
                results = pool_exec_processes(
                    generated_programs,
                    func_name="g6",
                    timeout=self.config.timeout,
                    processes=self.config.processes,
                    debug=self.config.debug,
                )
            except Exception:
                return self.generate_programs(code_batch)

        results = [
            {"program_str": gen_prog, "result_obj": res_obj, "config": self.config}
            for (gen_prog, res_obj) in zip(generated_programs, results)
        ]
        return [P3Solution(**p) for p in results]

    def evaluate_solution(self, sol: P3Solution) -> bool:
        """
        Returns whether or not the solution solves this problem
        """
        if self.ans_type is not None:
            return type_check(self.ans_type, sol.result_obj)

        eval_code = (
            f"{P3_IMPORTS}\n"
            f"{self.problem_func}\n"
            f"def run_eval():\n"
            f"    return f6({sol.result_obj})"
        )

        result = pool_exec_processes(
            eval_code,
            func_name="run_eval",
            timeout=self.config.timeout,
            processes=self.config.processes,
            debug=self.config.debug,
        )

        return result[0]

    def fitness(self, sol: P3Solution) -> float:
        """
        If passing the solution to the problem returns True, fitness is 1.0
            else -np.inf
        """
        result = self.evaluate_solution(sol)

        if result is True:
            return 1.0
        else:
            return -np.inf

    def random(self) -> list[P3Solution]:
        program_list = [self.construct_prompt() for _ in range(self.config.batch_size)]
        new_solutions = self.generate_programs(program_list)
        return new_solutions

    def mutate(self, sol_list: list[P3Solution]) -> list[P3Solution]:
        sols = [s.program_str for s in sol_list]
        program_list = list(map(self.construct_prompt, sols))
        new_sols = self.generate_programs(program_list)
        return new_sols


class P3ProbSolResult(Genotype):
    def __init__(self, program_str: str, result_obj: dict, config: P3ProbSolEnvConfig):
        """
        Genotype for a programming puzzle problem+solution pair.
        Args:
            program_str: the code for the pair.
            result_obj: the result of the solution.
            config: environment config
        """
        self.program_str = program_str
        self.result_obj = result_obj
        self.config = config

        i_f6 = program_str.find("def f6_2(")
        i_g6 = program_str.find("def g6_2(")
        i_assert = program_str.find("assert")
        self.problem_func = self.program_str[i_f6:i_g6].strip()
        self.solution_func = self.program_str[i_g6:i_assert].strip()

        # When comparing for phenotype, just use import statement and new probsol
        baseline = '''from typing import List

def f1_1(s: str):
    return "Hello " + s == "Hello world"

def g1_1():
    """Find a string that when concatenated onto 'Hello ' gives 'Hello world'."""
    return "world"'''
        self.baseline_emb = np.array(
            get_embedding(baseline, engine=self.config.embedding_model_path)
        )

        if self.config.embedding_model_type == "hf":
            self.pl = pipeline(
                "feature-extraction", model=self.config.embedding_model_path
            )
            seed_features = np.array(self.pl(baseline))
            self.scaler = StandardScaler()
            seed_features_scaled = self.scaler.fit_transform(np.squeeze(seed_features))
            self.pca = PCA(0.95)
            self.pca.fit(seed_features_scaled)

    def __str__(self) -> str:
        return self.program_str

    def to_phenotype(self) -> Optional[Phenotype]:
        if self.config.embedding_model_type == "openai":
            compare_str = (
                self.program_str
            )  # TODO: remove comments from f6_2 for diversity measurement
            i_assert = compare_str.find("assert")
            if i_assert > -1:
                compare_str = compare_str[:i_assert]
            emb = np.array(
                get_embedding(compare_str, engine=self.config.embedding_model_path)
            )
            return cosine_similarity(emb, self.baseline_emb)
        elif self.config.embedding_model_type == "hf":
            features = np.array(self.pl(self.program_str))
            features_scaled = self.scaler.transform(np.squeeze(features))
            pca_features = self.pca.transform(features_scaled)
            return pca_features.max(axis=0).flatten()

    def __getstate__(self):
        state = self.__dict__.copy()
        if "pl" in state:
            del state["pl"]
        if "scaler" in state:
            del state["scaler"]
        if "pca" in state:
            del state["pca"]
        return state


class P3ProbSol(BaseEnvironment[P3ProbSolResult]):
    def __init__(
        self,
        config: P3ProbSolEnvConfig,
        mutation_model: MutationModel,
    ) -> None:
        """
        The objective is to generate problem+solution pairs.
        Args:
            config: the config file path or dict.
            mutation_model: the diff model (or alternatives).
            ans_type: answer type
        """
        self.mutation_model = mutation_model
        self.config = config
        self.batch_size = self.config.batch_size
        self.seed_index = self.config.starting_seed
        self.rng = None

        if self.config.prompt_size == "long":
            self.prompt_seed = P3_PROBSOL_LONG_SEED
        elif self.config.prompt_size == "med":
            self.prompt_seed = P3_PROBSOL_MED_SEED
        else:
            raise ValueError("No seed string found")

        # Use the first example in the prompt seed as basis for embedding sizes
        i_first = self.prompt_seed.find("assert")
        first_example = self.prompt_seed[:i_first].strip()

        if self.config.embedding_model_type == "openai":
            self.genotype_ndim: int = 1
            self.genotype_space = np.repeat([[0, 1]], self.genotype_ndim, axis=0).T
        elif self.config.embedding_model_type == "hf":
            # Dummy to get behavior space shape
            dummy_pl = pipeline(
                "feature-extraction", model=self.config.embedding_model_path
            )
            dummy_features = np.array(dummy_pl(first_example))
            dummy_scaler = StandardScaler()
            dummy_features_scaled = dummy_scaler.fit_transform(
                np.squeeze(dummy_features)
            )
            dummy_pca = PCA(0.95)
            dummy_pca_features = dummy_pca.fit_transform(dummy_features_scaled)
            self.genotype_ndim: int = dummy_pca_features.shape[-1]
            self.genotype_space = np.repeat([[-20, 20]], self.genotype_ndim, axis=0).T

        # Get info for the seed puzzle that will be mutated
        # This puzzle is at the index of the puzzles array specified by self.seed_index
        # TODO: put this in a method or in construct_prompt()?
        puzzles = requests.get(
            "https://raw.githubusercontent.com/microsoft/PythonProgrammingPuzzles/v0.2/puzzles/puzzles.json"
        ).json()
        puzzle = puzzles[self.seed_index]
        if len(puzzle["sol_bodies"]) == 0:
            raise ValueError(
                f"No sample solution is provided for the puzzle at index {self.seed_index}"
            )

        f6_1 = puzzle["sat"].replace("def sat(", "def f6_1(")  # problem form is f6_1()
        g6_1 = puzzle["sol_header"].replace(
            "def sol(", "def g6_1("
        )  # solution form is g6_1()
        if self.config.prompt_size == "long":
            g6_1 += "\n" + puzzle["sol_docstring"]  # add in the docstring
        g6_1 += (
            "\n" + puzzle["sol_bodies"][0]
        )  # include the first example solution function body

        self.original_probsol = f6_1 + "\n\n" + g6_1 + "\n\n" + "assert f6_1(g6_1())"
        self.new_probsol_preamble = "def f6_2("

    def get_rng_state(self) -> Optional[np.random._generator.Generator]:
        warnings.warn("WARNING: rng state not used in this environment")
        return None

    def set_rng_state(self, rng_state: Optional[np.random._generator.Generator]):
        warnings.warn("WARNING: rng state not used in this environment")
        pass

    def construct_prompt(
        self, code_batch: Optional[Union[list[str], str]] = None
    ) -> dict[str, str]:
        prompt_str = self.prompt_seed

        if code_batch is None:
            # prompt with prob+sol from P3 dataset
            prompt_str += (
                f"\n\n{self.original_probsol}"  # add this particular probsol, f6_1() and g6_1(), to the prompt
                f"\n\n{self.new_probsol_preamble}"  # add f6_2() preamble to the prompt
            )
        else:
            # prompt with prob+sol that is given (one that was the output of a prev mutation)
            if isinstance(code_batch, list):
                # TODO: get nearby genotypes
                program_str = code_batch[0]
            elif isinstance(code_batch, str):
                program_str = code_batch

            # the prev output was f6_2 and g6_2, so now make it f6_1 and g6_1 for the prompt
            # and remove comments (which contain changes from prev f6_1) from new f6_1
            # TODO: pass in the whole object instead of the program_str since it already parsed some of this?
            i_f6 = program_str.find("def f6_2")
            program_str = program_str[i_f6:]  # remove import statement
            program_str = program_str.replace("f6_2(", "f6_1(")
            program_str = program_str.replace("g6_2(", "g6_1(")
            i_g6 = program_str.find("def g6_1(")
            # remove comments with """
            program_str = (
                re.sub('""".*"""', "", program_str[:i_g6]) + program_str[i_g6:]
            )
            # remove comments with # (and remove empty lines)
            i_g6 = program_str.find("def g6_1(")
            lines = program_str[:i_g6].strip().split("\n")
            new_lines = []
            for line in lines:
                if line.strip().startswith("#") or len(line.strip()) == 0:
                    continue
                new_lines.append(line)
            program_str = "\n".join(new_lines) + "\n\n" + program_str[i_g6:]
            program_str = program_str.strip()

            prompt_str += f"\n\n{program_str}" f"\n\n{self.new_probsol_preamble}"

        template = f"{P3_IMPORTS}\n{self.new_probsol_preamble}"
        return {"prompt": prompt_str, "template": template}

    def generate_programs(self, code_batch: list[str]) -> list[P3ProbSolResult]:
        """Generate new programs with a mutation model and evaluate them."""
        local_scope_exec = False
        generated_programs = self.mutation_model.generate_programs(
            code_batch, local_scope_exec
        )

        if self.config.sandbox:
            results = []
            for code in generated_programs:
                resp = requests.post(
                    f"{self.sandbox_server}/eval_p3_solution",
                    json={"code": code, "timeout": self.config.timeout},
                    timeout=self.config.timeout,
                )
                if resp.status_code == 200:
                    return_dict = json.loads(resp.text)
                    results.append(return_dict)
        else:
            # TODO: handle (probably inside of pool_exec_processes) all cases where the generated code returns
            # a generator type. The multithreaded execution pickles things and generators can't be pickled
            # which causes the whole thing to error out.
            # For now, try/except and re-try.
            try:
                results = pool_exec_processes(
                    generated_programs,
                    func_name="g6_2",
                    timeout=self.config.timeout,
                    processes=self.config.processes,
                    debug=self.config.debug,
                )
            except Exception:
                return self.generate_programs(code_batch)

        results = [
            {"program_str": gen_prog, "result_obj": res_obj, "config": self.config}
            for (gen_prog, res_obj) in zip(generated_programs, results)
        ]
        return [P3ProbSolResult(**p) for p in results]

    def fitness(self, probsol: P3ProbSolResult) -> float:
        """
        Fitness is the inverse of pass@k of the problem func.
        We want a pass@k of >0 so that the problem is reasonably solvable.
        So fitness=0 if unsolved (which is still better than -np.inf).
        Other than that, more difficult (lower pass@k) => higher fitness.
        """
        if isinstance(probsol.result_obj, ExecResult):
            return -np.inf

        # TODO: check type expected by f6_2 if any?
        # TODO: implement checks for absolute triviality of f6_2 requirements
        #   the fitness function being based on pass@k might take care of this though

        eval_code = (
            f"{P3_IMPORTS}\n"
            f"{probsol.problem_func}\n"
            f"def run_eval():\n"
            f"    return f6_2({probsol.result_obj})"
        )

        # Run code to see if g6_2 solves f6_2
        result = pool_exec_processes(
            eval_code,
            func_name="run_eval",
            timeout=self.config.timeout,
            processes=self.config.processes,
            debug=self.config.debug,
        )

        if result[0] is True:
            return -np.inf

        # Do pass@k eval

        # Get f6_2() and make it the new f6()
        problem_str = probsol.problem_func.replace("def f6_2(", "def f6(")
        # Remove comments with """
        problem_str = re.sub('""".*"""', "", problem_str)
        # Remove comments with # (and remove empty lines)
        lines = problem_str.strip().split("\n")
        new_lines = []
        for line in lines:
            if line.strip().startswith("#") or len(line.strip()) == 0:
                continue
            new_lines.append(line)
        problem_str = "\n".join(new_lines)
        # Get solution_preamble for g6()
        i_end_preamble = probsol.solution_func.find("):")
        solution_preamble = probsol.solution_func[: i_end_preamble + 2].replace(
            "def g6_2(", "def g6("
        )

        p3_problem = P3Problem(
            self.config,  # TODO: make an actual P3ProblemEnvConfig
            self.mutation_model,
            problem_str=problem_str,
            solution_preamble=solution_preamble,
        )
        solutions = []
        for _ in range(self.config.eval_k // self.config.batch_size + 1):
            solutions += p3_problem.random()

        c = 0
        for s in solutions:
            if p3_problem.evaluate_solution(s) is True:
                c += 1

        pak = pass_at_k(len(solutions), c, self.config.eval_k)
        return 1 / pak if pak > 0 else 0

    def random(self) -> list[P3ProbSolResult]:
        program_list = [self.construct_prompt() for _ in range(self.config.batch_size)]
        new_probsols = self.generate_programs(program_list)
        return new_probsols

    def mutate(self, probsol_list: list[P3ProbSolResult]) -> list[P3ProbSolResult]:
        probsols = [pb.program_str for pb in probsol_list]
        program_list = list(map(self.construct_prompt, probsols))
        new_probsols = self.generate_programs(program_list)
        return new_probsols
