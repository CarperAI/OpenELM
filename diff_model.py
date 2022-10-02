import os
import re
import shutil
from typing import Dict

import torch

from codegen.codegen_utilities import model_setup, sample, set_seed, truncate
from codex_execute import (TimeoutException, create_tempdir, reliability_guard,
                           swallow_io, time_limit)
from walker.walk_creator import Walker


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
        return 6  # No proper function found in code.
    with create_tempdir():

        # These system calls are needed when cleaning up tempdir.
        import os
        import shutil

        rmtree = shutil.rmtree
        rmdir = os.rmdir
        chdir = os.chdir

        # Disable functionalities that can make destructive changes to the test.
        reliability_guard()
        try:
            # TODO: Check https://arxiv.org/pdf/2209.07753.pdf
            with swallow_io():
                with time_limit(timeout):
                    exec(code_str, {}, code_dct)
                    return code_dct["make_walker"]()
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
        except Exception:
            reset_os_funcs(rmtree, rmdir, chdir)
            return 6  # Code fails to run - other error.


class DiffModel():
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        set_seed(self.cfg.seed)

    def generate_prompt_str(self, seed, tokenizer):
        if self.cfg.task == "Sodarace":
            encoding = tokenizer([seed], truncation=True, padding=True,
                                max_length=2048,
                                return_tensors='pt')
        elif self.cfg.task == "Imagegen":
            encoding = tokenizer([seed], truncation=True, padding=True,
                                max_length=2048,
                                return_tensors='pt')
        return encoding

    def generate_program(self, seed: str) -> dict:
        model, tokenizer = model_setup(self.cfg)
        encoding = self.generate_prompt_str(seed, tokenizer)
        completion = sample(self.cfg, model, tokenizer, encoding)
        truncation = truncate(completion)
        execution_result = unsafe_execute(truncation, timeout=self.cfg.timeout)
        if isinstance(execution_result, Walker):
            if execution_result.validate():
                sodaracer_dict: dict = execution_result.serialize_walker_sodarace()
                return {
                    "program_str": truncation,
                    "sodaracer": sodaracer_dict,
                }
        return {}
