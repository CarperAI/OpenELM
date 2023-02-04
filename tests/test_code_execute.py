from openelm.sandbox.server.utils import ErrorCode
from openelm.utils.code_eval import eval_completions, pool_exec_processes


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
    result = pool_exec_processes(PARITY_PROMPT, "parity", args=test_input)
    assert result == 1
    test_input = {"b1": 1, "b2": 1, "b3": 1, "b4": 1}
    result = pool_exec_processes(PARITY_PROMPT, "parity", args=test_input)
    assert result == 0

    # Completion test
    result = eval_completions(eval_results=[PARITY_PROMPT])
    assert result == [0]

    # Timeout test
    PROMPT = """
import time
def test_func():
    print("Hello World!")
    time.sleep(1.1)
    return 0
"""
    result = pool_exec_processes(PROMPT, "test_func", timeout=1.0)
    assert result == ErrorCode.TIMEOUT_EXCEPTION

    # Syntax error test
    PROMPT = """
def test_func():
test_dct = {}
    return 0
"""
    result = pool_exec_processes(PROMPT, "test_func")
    assert result == ErrorCode.SYNTAX_ERROR

    # Type error test
    PROMPT = """
def test_func():
    result = 1 + "Hello World"
    return 0
"""
    result = pool_exec_processes(PROMPT, "test_func")
    assert result == ErrorCode.TYPE_ERROR

    # Other exception test
    PROMPT = """
def test_func():
    result = 5 / 0
    return 0
"""
    result = pool_exec_processes(PROMPT, "test_func")
    assert result == ErrorCode.EXCEPTION
