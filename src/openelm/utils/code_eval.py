import functools
import itertools
import multiprocessing as mp
from typing import Any, Optional, Union

from openelm.sandbox.server.sandbox_codex_execute import unsafe_execute


def pool_exec_processes(
    prompt: str,
    func_name: Optional[str] = None,
    args: Optional[dict] = None,
    timeout: float = 5.0,
    processes: int = 1,
    debug: bool = False,
) -> Any:
    """
    Execute code in separate process(s).

    This ensures that we avoid disabling system functions in the main process.

    Note that we cannot do this in another *thread*, since execute uses `signal`
    which only works in the main thread.

    Args:
        prompt (str): Prompt string.
        func_name (str): Name of function in prompt string to execute.
        args (dict): Arguments to pass to function.
        timeout (float): Timeout limit in seconds.
        processes (int): Number of processes to use.
        debug (bool): Whether to print debug messages.
    """
    with mp.Pool(processes=processes) as pool:
        eval_fn = functools.partial(
            unsafe_execute,
            func_name=func_name,
            args=args,
            timeout=timeout,
            debug=debug,
        )
        result = list(pool.map(eval_fn, [prompt]))[0]
    if debug:
        print(result)
    return result


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
            res_arr = []
            for args, res in parity_test_data:
                res_arr.append(
                    pool_exec_processes(code, func_name="parity", args=args) == res
                )
            if not all(res_arr):
                results.append(1)
            else:
                results.append(0)
        if isinstance(eval_results, list):
            # Batch evaluation returns the batch
            return results
        else:
            # Single evaluation returns a single result
            return results[0]
    else:
        raise ValueError(f"Unknown task: {task}")


def mutate_code(n_bugs: int = 5, task: str = "parity", mutate_method="prompt") -> tuple:
    """
    Mutate code to create n bugs. Output the prompt in diff format.

    Args:
        n_bugs: number of bugs to introduce (from 1 to 5).
        task: (Optional) the task to be performed.
        mutate_method: (Optional) 'diff' or 'prompt',
        corresponding to diff mutation or prompt mutation.

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
        ],
    }
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


parity_test_data = list(
    ({k: v for k, v in zip([f"b{i}" for i in range(1, 5)], i)}, parity_reference(*i))
    for i in itertools.product(range(2), repeat=4)
)


def quadratic(a, b, c, x):
    """Return quadratic: a,b,c are coefficients and x is the independent variable."""
    return a * x**2 + b * x + c
