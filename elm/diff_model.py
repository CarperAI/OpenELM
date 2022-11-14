import os
import re
import shutil
from typing import Dict

from elm.codegen.codegen_utilities import model_setup, sample, set_seed, truncate
from elm.codex_execute import (
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
        # reliability_guard()
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


class DiffModel:
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        set_seed(self.cfg.seed)
        self.model, self.tokenizer = model_setup(self.cfg)

    def generate_prompt_str(self, seed, tokenizer):
        if self.cfg.env_name == "Sodarace":
            encoding = tokenizer(
                [seed],
                truncation=True,
                padding=True,
                max_length=2048,
                return_tensors="pt",
            )
        elif self.cfg.env_name == "Imagegen":
            encoding = tokenizer(
                [seed],
                truncation=True,
                padding=True,
                max_length=2048,
                return_tensors="pt",
            )
        return encoding

    def generate_program(self, seed: str) -> dict:
        # encoding = self.generate_prompt_str(seed, self.tokenizer)
        while True:
            # completions = sample(self.cfg, self.model, self.tokenizer, encoding)
            # truncation = truncate(completions[0])
            execution_result = unsafe_execute(seed, timeout=self.cfg.timeout)
            if isinstance(execution_result, Walker):
                if execution_result.validate():
                    sodaracer_dict: dict = execution_result.to_dict()
                    return {
                        "program_str": seed,
                        "result_dict": sodaracer_dict,
                    }
