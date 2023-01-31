import itertools
import os
import re
import shutil
from typing import Union

from openelm.codegen.codex_execute import create_tempdir, reliability_guard, swallow_io, time_limit, TimeoutException


def eval_completions(
        eval_results: Union[str, list[str]], task: str = "parity", timeout: int = 5
) -> Union[int, list[int]]:
    """
    Evaluate (a batch of) the modified eval_results on a task.

    Args:
        eval_results: either a string or a batch of strings. The code(s) to be evaluated.
        task: (Optional) the task to be performed.
        timeout: (Optional) the timeout (in seconds).

    Returns:
        either the status code of the string or a list of status eval_results of the batch
        of strings (depending on whether `eval_results` is batched).
    """
    if task == "parity":
        _eval_results = (
            eval_results if isinstance(eval_results, list) else [eval_results]
        )
        results = []
        for code in _eval_results:
            results.append(eval_code_string(code, parity_test_data, timeout))
        if isinstance(eval_results, list):
            # Batch evaluation returns the batch
            return results
        else:
            # Single evaluation returns a single result
            return results[0]
    else:
        raise ValueError(f"Unknown task: {task}")


def mutate_code(n_bugs: int = 5, task: str = "parity", mutate_method='prompt') -> tuple:
    """
    Mutate code to create n bugs. Output the prompt in diff format.

    Args:
        n_bugs: number of bugs to introduce (from 1 to 5).
        task: (Optional) the task to be performed.
        mutate_method: (Optional) 'diff' or 'prompt', corresponding to diff mutation or prompt mutation.

    Returns:
        mutated_code, function_string
    """
    mutation_templates = {
        "diff": [
            f"<NME> {task}.py\n<BEF> ",
            "",  # placeholder for the context, e.g., the buggy code
            "\n<MSG> Fixed bugs",
        ],
        "prompt": [
            "# A buggy implementation\n#!/usr/bin/python3\n",
            "",  # placeholder for the context, e.g., the buggy code
            "\n# Fixed bugs\ndef",
        ]}
    mutation_template = mutation_templates[mutate_method]
    if task == "parity":
        variables = ["b", "b", "b", "b", 2]
        for i in range(n_bugs):
            variables[i] = "c" if i < 4 else 3
        func_str = (
            'def parity(b1,b2,b3,b4):\n    """Return binary parity of a sequence of input bits.'
            ' Return 0 for even parity, 1 for odd parity."""\n    bit_sum = sum(['
            "{}1,{}2,{}3,{}4])\n    return bit_sum % {}".format(*variables)
        )
        mutation_template[1] = func_str
        return "".join(mutation_template), func_str
    else:
        raise ValueError(f"Unknown task: {task}")


def parity_reference(b1, b2, b3, b4):
    """Return binary parity of a sequence of input bits. Return 0 for even parity, 1 for odd parity."""
    bit_sum = sum([b1, b2, b3, b4])
    return bit_sum % 2


def quadratic(a, b, c, x):
    """Return quadratic: a,b,c are coefficients and x is the independent variable."""
    return a * x ** 2 + b * x + c


def reset_os_funcs(rmtree, rmdir, chdir):
    shutil.rmtree = rmtree
    os.rmdir = rmdir
    os.chdir = chdir


def eval_code_string(code_str: str, ground_truth: dict, timeout: int = 5):
    if len(code_str) == 0 or "def " not in code_str:
        return 6  # No code found or no function found.
    code_dct: dict = {}
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
                    if not all(
                            [
                                code_dct[func_name](*i) == res
                                for i, res in ground_truth.items()
                            ]
                    ):
                        reset_os_funcs(rmtree, rmdir, chdir)
                        return 1  # Code runs but fails a test.
                    else:
                        reset_os_funcs(rmtree, rmdir, chdir)
                        return 0  # Passes all tests.
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


parity_test_data = {
    i: parity_reference(*i) for i in itertools.product(range(2), repeat=4)
}
