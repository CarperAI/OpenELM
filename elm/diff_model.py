import json
import os
import re
import shutil
from abc import abstractmethod, ABC
from typing import Dict

import requests
import torch

from elm.codegen.codegen_utilities import model_setup, sample, set_seed, truncate
from elm.codegen.codex_execute import (
    TimeoutException,
    create_tempdir,
    reliability_guard,
    swallow_io,
    time_limit,
)
from elm.environments.sodaracer import Walker


def reset_os_funcs(rmtree, rmdir, chdir):
    shutil.rmtree = rmtree
    os.rmdir = rmdir
    os.chdir = chdir


def unsafe_execute(code_str: str, timeout: int = 5):
    if len(code_str) == 0 or "def " not in code_str:
        return 6  # No code found or no function found.
    code_dct: Dict = {}
    func_match = re.search(r"def (\w+)\s*\((.*?)\):", code_str)
    if func_match:
        func_name = func_match.group(1)
    else:
        print("No function found")
        return 6  # No proper function found in code.
    with create_tempdir():

        # These system calls are needed when cleaning up tempdir.
        import os
        import shutil

        rmtree = shutil.rmtree
        rmdir = os.rmdir
        chdir = os.chdir

        # Disable functionalities that can make destructive changes to the test.
        # TODO: fix interaction between reliability guard and create_tempdir
        reliability_guard()
        try:
            # TODO: Check https://arxiv.org/pdf/2209.07753.pdf
            with swallow_io():
                with time_limit(timeout):
                    exec(code_str, code_dct, code_dct)
                    return code_dct["make_walker"]()
        except ValueError:
            reset_os_funcs(rmtree, rmdir, chdir)
            return 1  # Code fails validation check.
        except TimeoutException:
            reset_os_funcs(rmtree, rmdir, chdir)
            return 2  # Code takes too long to run.
        except RuntimeError:
            reset_os_funcs(rmtree, rmdir, chdir)
            return 3  # Code runs but crashes.
        except SyntaxError:
            reset_os_funcs(rmtree, rmdir, chdir)
            return 4  # Code does not run - syntax error.
        except TypeError:
            reset_os_funcs(rmtree, rmdir, chdir)
            return 5  # Code does not run - type error.
        except Exception as e:
            reset_os_funcs(rmtree, rmdir, chdir)
            print(e)
            return 6  # Code fails to run - other error.


class Model(ABC):
    @abstractmethod
    def generate_program(self, seed_str: str) -> dict:
        pass


class PromptMutationModel(Model):
    func_name: str  # the name of the function that we want to execute
    import_line: str  # the import lines we add to the code
    func_preamble: str  # the function definition plus possibly a few initial lines to generate codes
    return_line: str  # the return line we add to the end of the code

    def __init__(self, cfg, sandbox_server='http://localhost:5000') -> None:
        self.cfg = cfg
        set_seed(self.cfg.seed)
        self.sandbox_server = sandbox_server
        self.model, self.tokenizer = model_setup(self.cfg)

    def generate_prompt_str(self, seed: str, tokenizer=None, num=1,
                            append_return=True, without_trunc=False) -> list[str]:
        """
        Args:
            seed: the seed text.
            tokenizer: (Optional) assign only if you want to use a different tokenizer (default: None)
            num: (Optional) batch size.
            append_return: (Optional) append a return line to the code in the end.
            without_trunc: (Optional) True if we don't apply the `truncate` function.
        Returns:
            a list of code(s) generated by the model.
        """
        tokenizer = self.tokenizer if tokenizer is None else tokenizer

        encoding = tokenizer(
            [seed + '\n\n' + self.func_preamble],
            truncation=True,
            padding=True,
            max_length=self.cfg.gen_max_len,
            return_tensors="pt",
        )
        self.cfg.batch_size = num

        with torch.no_grad():
            completion = sample(self.cfg, self.model, self.tokenizer, encoding)

        if without_trunc:
            truncation = completion
        else:
            truncation = [truncate(code, print_num=float('inf'), only_local_scope=True) for code in completion]

        truncation = [self.import_line + '\n' + self.func_preamble + '\n' + code for code in truncation]

        if append_return:
            truncation = [code + '\n' + self.return_line for code in truncation]

        return truncation

    @abstractmethod
    def generate_program(self, code: str) -> dict:
        pass


class PromptMutationForSodarace(PromptMutationModel):
    func_name: str = 'make_walker'
    import_line: str = 'from .walker import walker_creator'
    func_preamble: str = f'def {func_name}():\n\twc = walker_creator()\n'
    return_line: str = '\treturn wc.get_walker()\n'

    def generate_program(self, code: str) -> dict:
        """
        Given a piece of code, do prompt mutation, call the sandbox server to execute the code and return the result.
        Args:
            code: the full code string.
        Returns:
            a numpy array (if successful) or the exception object.
        """
        code = self.generate_prompt_str(code)[0]

        resp = requests.post(
            f"{self.sandbox_server}/gen_racer",
            json={"code": code, "timeout": 5.0},
            timeout=5,
        )

        if resp.status_code == 200:
            return_dict = json.loads(resp.text)
            error_code = "0"
        elif resp.status_code == 500:  # Bad request
            try:
                msg = json.loads(resp.text)
                return_dict = {"program_str": code, "result_dict": msg["message"]}
                error_code = msg["unsafe_execute_error_code"]
            except Exception as e:
                return_dict = {"program_str": code, "result_dict": str(e)}
                error_code = 6
        else:
            return_dict = {"program_str": code, "result_dict": resp.text}
            error_code = 6

        result = {**return_dict, "error_code": error_code}

        return result


class DiffModel(Model):
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        set_seed(self.cfg.seed)
        self.model, self.tokenizer = model_setup(self.cfg)

    def generate_prompt_str(self, seed, tokenizer):
        if self.cfg.env_name == "sodarace":
            encoding = tokenizer(
                [seed],
                truncation=True,
                padding=True,
                max_length=2048,
                return_tensors="pt",
            )
        elif self.cfg.env_name == "imageoptim":
            encoding = tokenizer(
                [seed],
                truncation=True,
                padding=True,
                max_length=2048,
                return_tensors="pt",
            )
        return encoding

    def generate_program(self, seed_str: str) -> dict:
        encoding = self.generate_prompt_str(seed_str, self.tokenizer)
        while True:
            completion = sample(self.cfg, self.model, self.tokenizer, encoding)
            # truncation = truncate(completions[0])
            execution_result = unsafe_execute(completion, timeout=self.cfg.timeout)
            if isinstance(execution_result, Walker):
                if execution_result.validate():
                    sodaracer_dict: dict = execution_result.to_dict()
                    return {
                        "program_str": seed_str,
                        "result_dict": sodaracer_dict,
                    }
