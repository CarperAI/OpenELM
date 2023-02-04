import functools
import multiprocessing as mp
from typing import Any, Optional

from openelm.sandbox.server.utils import ErrorCode, sandbox_unsafe_execute


def exec_code_test(prompt: str, func_name: Optional[str] = None,
                   args: Optional[dict] = None, timeout: float = 5.0,
                   debug: bool = True) -> Any:
    """
    Execute code in separate process.

    This ensures that we avoid disabling system functions in the main process.

    Note that we cannot do this in another *thread*, since execute uses `signal`
    which only works in the main thread.

    Args:
        prompt (str): Prompt string.
        func_name (str): Name of function in prompt string to execute.
        args (dict): Arguments to pass to function.
        timeout (float): Timeout limit in seconds.
        debug (bool): Whether to print debug messages.
    """
    with mp.Pool(processes=1) as pool:
        eval_fn = functools.partial(sandbox_unsafe_execute,
                                    func_name=func_name,
                                    args=args,
                                    timeout=timeout,
                                    debug=debug)
        result = list(pool.map(eval_fn, [prompt]))[0]
    if debug:
        print(result)
    return result


def test_code_execute():
    # Correctness
    PARITY_PROMPT = """
def parity(b1,b2,b3,b4):
    \"\"\"Return binary parity of a sequence of input bits.
    Return 0 for even parity, 1 for odd parity.\"\"\"
    bit_sum = sum([b1,b2,b3,b4])
    return bit_sum % 2
"""
    test_input = {"b1": 1, "b2": 0, "b3": 1, "b4": 1}
    result = exec_code_test(PARITY_PROMPT, "parity", args=test_input)
    assert result == 1
    test_input = {"b1": 1, "b2": 1, "b3": 1, "b4": 1}
    result = exec_code_test(PARITY_PROMPT, "parity", args=test_input)
    assert result == 0

    # Timeout test
    PROMPT = """
import time
def test_func():
    print("Hello World!")
    time.sleep(1.1)
    return 0
"""
    result = exec_code_test(PROMPT, "test_func", timeout=1.0)
    assert result == ErrorCode.TIMEOUT_EXCEPTION

    # Syntax error test
    PROMPT = """
def test_func():
test_dct = {}
    return 0
"""
    result = exec_code_test(PROMPT, "test_func")
    assert result == ErrorCode.SYNTAX_ERROR

    # Type error test
    PROMPT = """
def test_func():
    result = 1 + "Hello World"
    return 0
"""
    result = exec_code_test(PROMPT, "test_func")
    assert result == ErrorCode.TYPE_ERROR

    # Other exception test
    PROMPT = """
def test_func():
    result = 5 / 0
    return 0
"""
    result = exec_code_test(PROMPT, "test_func")
    assert result == ErrorCode.EXCEPTION
