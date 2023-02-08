import functools
import itertools
import multiprocessing as mp
from typing import Any, Iterable, Optional, Union

from openelm.sandbox.server.sandbox_codex_execute import ExecResult, unsafe_execute


def pool_exec_processes(
    prompts: Union[str, Iterable[str]],
    func_name: Optional[str] = None,
    args: Optional[dict[str, Any]] = None,
    ground_truth: Optional[dict[tuple, Any]] = None,
    timeout: float = 5.0,
    processes: int = 0,
    debug: bool = False,
) -> list[Any]:
    """
    Execute code in separate process(s).

    Args:
        prompts (str or Iterable): Prompt string(s) to execute.
        func_name (str): Name of function in prompt string to execute.
        args (dict): Arguments to pass to function.
        ground_truth (dict): Dict with args as keys and correct return values as
        values.
        timeout (float): Timeout limit in seconds.
        processes (int): Number of processes to use.
        debug (bool): Whether to print debug messages.
    """
    if isinstance(prompts, str):
        prompts = [prompts]

    eval_fn = functools.partial(
        unsafe_execute,
        func_name=func_name,
        args=args,
        ground_truth=ground_truth,
        timeout=timeout,
        debug=debug,
    )
    if processes <= 1:
        return list(map(eval_fn, prompts))
    with mp.Pool(processes=processes) as pool:
        results = list(pool.map(eval_fn, prompts))
    if debug:
        print(results)
    return results


def eval_completions(
    eval_results: Union[str, Iterable[str]],
    task: str = "parity",
    timeout: float = 5.0,
    processes: int = 1,
    debug: bool = False,
) -> list[Union[int, ExecResult]]:
    """
    Evaluate (a batch of) the modified eval_results on a task.

    Args:
        eval_results: either a string or a batch of strings. The code(s) to be evaluated.
        task: (Optional) the task to be performed.
        timeout: (Optional) the timeout (in seconds).
        processes: (Optional) the number of processes to use.

    Returns:
        A list of status eval_results of the batch of strings.
    """
    if task == "parity":
        if isinstance(eval_results, str):
            eval_results = [eval_results]
        results = pool_exec_processes(
            eval_results,
            func_name="parity",
            ground_truth=parity_test_data,
            timeout=timeout,
            processes=processes,
            debug=debug,
        )
        return results
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


parity_test_data = {
    i: parity_reference(*i) for i in itertools.product(range(2), repeat=4)
}


def quadratic(a, b, c, x):
    """Return quadratic: a,b,c are coefficients and x is the independent variable."""
    return a * x**2 + b * x + c
