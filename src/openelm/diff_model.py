import functools
import json
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import requests

from openelm.codegen import model_setup, sample, set_seed, truncate
from openelm.configs import SodaraceELMConfig
from openelm.environments.sodaracer import IMPORTS, SQUARE_PREREQ, Walker
from openelm.utils.code_eval import pool_exec_processes
from openelm.utils.diff_eval import apply_diff, split_diff


class MutationModel(ABC):
    """Base model class for all mutation models."""

    @abstractmethod
    def generate_program(self, code_batch: list[str]) -> list[dict]:
        pass


@dataclass
class FunctionTemplate:
    """
    A function template for a mutation model.

    Attributes:
        func_name: (str) The name of the function that we want to execute.
        import_line: (str) The import lines we add to the code.
        func_preamble: (str) The function definition, as well as potentially a
        few initial lines to generate code.
        instruction (str): The instruction we give to the model, before the
        preamble.
    """

    func_name: str
    import_line: str
    func_preamble: str
    instruction: str


class PromptMutationModel(MutationModel):
    """Mutation model that uses prompts to change a seed."""

    def __init__(
        self,
        cfg: SodaraceELMConfig,
        function_template: FunctionTemplate,
        sandbox_server: str = "http://localhost:5000",
    ) -> None:

        self.cfg: SodaraceELMConfig = cfg
        seed: int = set_seed(self.cfg.seed)
        # Use RNG to rotate random seeds during inference.
        self.rng = np.random.default_rng(seed=seed)
        self.sandbox_server = sandbox_server
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.model, self.tokenizer, self.device = model_setup(self.cfg)
        self.func_template: FunctionTemplate = function_template

    def construct_prompt(self, code: str) -> tuple[str, str]:
        """
        Construct a prompt from a code string.

        Args:
            code (str): The code string.

        Returns:
            A tuple of the prompt string and imports plus instruction.
        """
        prompt_str = (
            code + self.func_template.instruction + self.func_template.func_preamble
        )
        preamble_str = (
            self.func_template.import_line
            + self.func_template.instruction
            + self.func_template.func_preamble
        )
        return prompt_str, preamble_str

    def generate_program(self, code_batch: list[str]) -> list[dict]:
        """
        Generate a new program from a batch of programs.

        Given a piece of code, do prompt mutation, execute the code,
        and return the result.

        Args:
            code (str): The full code string.

        Returns:
            A numpy array (if successful) or the exception object.
        """
        prompts, preamble_strings = zip(*map(self.construct_prompt, code_batch))
        encodings = self.tokenizer(
            list(prompts),
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        completions: list[str] = sample(
            encodings,
            self.cfg,
            self.model,
            self.tokenizer,
            batch_size=1,
        )
        local_scope_exec: bool = len(self.func_template.func_preamble) > 0
        trunc = functools.partial(truncate, only_local_scope=local_scope_exec)
        self.truncations: list[str] = [
            preamble_strings[i] + trunc(completions[i]) for i in range(len(completions))
        ]
        if self.cfg.sandbox:
            results = []
            for code in self.truncations:
                resp = self._get_response(code, self.cfg.timeout)
                if resp.status_code == 200:
                    return_dict = json.loads(resp.text)
                    results.append(return_dict)
        else:
            results = pool_exec_processes(
                self.truncations,
                func_name=self.func_template.func_name,
                timeout=self.cfg.timeout,
                processes=self.cfg.processes,
                debug=self.cfg.debug,
            )
        return self._post_process(results)

    @abstractmethod
    def _get_response(self, code: str, timeout: float) -> requests.models.Response:
        raise NotImplementedError

    @abstractmethod
    def _post_process(self, results: list) -> list:
        raise NotImplementedError


class PromptMutationForSodarace(PromptMutationModel):
    def __init__(self, cfg, sandbox_server="http://localhost:5000") -> None:
        function_template = FunctionTemplate(
            func_name="make_walker",
            import_line=IMPORTS + SQUARE_PREREQ,
            instruction="",
            func_preamble="def make_walker():\n",
        )
        super().__init__(cfg, function_template, sandbox_server)

    def _get_response(self, code: str, timeout: float) -> requests.models.Response:
        return requests.post(
            f"{self.sandbox_server}/gen_racer",
            json={"code": code, "timeout": timeout},
            timeout=timeout,
        )

    def _post_process(self, results: list) -> list:
        if self.cfg.sandbox:
            return results
        else:
            result_list: list = []
            for i, result in enumerate(results):
                try:
                    if isinstance(result, Walker) and result.validate():
                        result_list.append(
                            {
                                "program_str": self.truncations[i],
                                "result_obj": result.to_dict(),
                            }
                        )
                    else:
                        if self.cfg.debug:
                            print("Failed execution, type:", result)
                            print(self.truncations[i])
                except Exception as e:
                    if self.cfg.debug:
                        print(type(e), e)
            return result_list


class PromptMutationForImgTask(PromptMutationModel):
    def __init__(self, cfg, sandbox_server="http://localhost:5000") -> None:
        func_name = "draw"
        func_preamble = (
            f'def {func_name}():\n\t"""Draw a yellow circle.\n'
            '\t"""\n\tpic = np.zeros((32, 32, 3))\n'
        )
        function_template = FunctionTemplate(
            func_name=func_name,
            import_line="import math\nimport numpy as np",
            func_preamble=func_preamble,
            instruction="",
        )
        super().__init__(cfg, function_template, sandbox_server)

    def reset_shape(self, shape: tuple):
        func_name = self.func_template.func_name
        self.func_preamble = f'def {func_name}():\n\t"""Draw a yellow circle.\n\t"""\n\tpic = np.zeros({shape})\n'

    def _get_response(self, code: str, timeout: float) -> requests.models.Response:
        func_name = self.func_template.func_name
        return requests.post(
            f"{self.sandbox_server}/eval_imageoptim_func",
            json={"code": code, "func_name": func_name, "timeout": timeout},
            timeout=timeout,
        )

    def _post_process(self, results: list) -> list:
        for i in range(len(results)):
            results[i]["result_obj"] = np.array(results[i]["result_obj"])
        return results


class DiffModel(PromptMutationModel):
    def __init__(
        self,
        cfg: SodaraceELMConfig,
        function_template: FunctionTemplate,
        sandbox_server: str = "http://localhost:5000",
    ) -> None:
        super().__init__(cfg, function_template, sandbox_server)

    def construct_prompt(self, code: str) -> tuple[str, str]:
        prompt_list = [
            "<NME> walker.py\n<BEF> ",
            code,
            "\n<MSG> Fixed bugs",
        ]
        prompt_str = "".join(prompt_list)
        prompt_str = (
            code + self.func_template.instruction + self.func_template.func_preamble
        )
        preamble_str = (
            self.func_template.import_line
            + self.func_template.instruction
            + self.func_template.func_preamble
        )
        return prompt_str, preamble_str

    def generate_program(self, code_batch: list[str]) -> list[dict]:
        """
        Generate a new program for a diff model from a batch of programs.

        Given a piece of code, do prompt mutation, execute the code,
        and return the result.

        Args:
            code (str): The full code string.

        Returns:
            A numpy array (if successful) or the exception object.
        """
        prompts, preamble_strings = zip(*map(self.construct_prompt, code_batch))
        encodings = self.tokenizer(
            list(prompts),
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        completions: list[str] = sample(
            encodings,
            self.cfg,
            self.model,
            self.tokenizer,
            batch_size=1,
        )

        local_scope_exec: bool = len(self.func_template.func_preamble) > 0
        end_of_diff = re.compile("\n[^ +-@]+")
        trunc = functools.partial(truncate, only_local_scope=local_scope_exec)
        self.truncations: list[str] = [
            preamble_strings[i] + trunc(completions[i]) for i in range(len(completions))
        ]
        outputs = []
        for i, code in enumerate(self.truncations):
            # split the diff text according to <NME>, <BEF>, <MSG>, <DFF>.
            parsed: dict = split_diff(code)
            # truncate the diff hunk at the first line not starting with " ",
            # "+", "-", or "@".
            if parsed and all(
                (s in parsed for s in ["name", "file", "message", "diff"])
            ):
                diff_hunk: str = end_of_diff.split(parsed["diff"])[0]
                nme_idx: int = diff_hunk.find("<NME>")
                if nme_idx != -1:
                    diff_hunk = diff_hunk[:nme_idx]
                outputs.append(apply_diff(prompts[i], diff_hunk))
        if self.cfg.sandbox:
            results = []
            for code in outputs:
                resp = self._get_response(code, self.cfg.timeout)
                if resp.status_code == 200:
                    return_dict = json.loads(resp.text)
                    results.append(return_dict)
        else:
            results = pool_exec_processes(
                outputs,
                func_name=self.func_template.func_name,
                timeout=self.cfg.timeout,
                processes=self.cfg.processes,
                debug=self.cfg.debug,
            )
        return self._post_process(results)


class DiffModelForSodarace(DiffModel):
    def __init__(self, cfg, sandbox_server="http://localhost:5000") -> None:
        function_template = FunctionTemplate(
            func_name="make_walker",
            import_line=IMPORTS + SQUARE_PREREQ,
            instruction="",
            func_preamble="def make_walker():\n",
        )
        super().__init__(cfg, function_template, sandbox_server)

    def _get_response(self, code: str, timeout: float) -> requests.models.Response:
        return requests.post(
            f"{self.sandbox_server}/gen_racer",
            json={"code": code, "timeout": timeout},
            timeout=timeout,
        )

    def _post_process(self, results: list) -> list:
        if self.cfg.sandbox:
            return results
        else:
            result_list: list = []
            for i, result in enumerate(results):
                try:
                    if isinstance(result, Walker) and result.validate():
                        result_list.append(
                            {
                                "program_str": self.truncations[i],
                                "result_obj": result.to_dict(),
                            }
                        )
                    else:
                        if self.cfg.debug:
                            print("Failed execution, type:", result)
                            print(self.truncations[i])
                except Exception as e:
                    if self.cfg.debug:
                        print(type(e), e)
            return result_list
