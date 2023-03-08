from openelm.sandbox.server.sandbox_codex_execute import ExecResult
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
    result = pool_exec_processes(PARITY_PROMPT, "parity", args=test_input)[0]
    assert result == 1
    test_input = {"b1": 1, "b2": 1, "b3": 1, "b4": 1}
    result = pool_exec_processes(PARITY_PROMPT, "parity", args=test_input)[0]
    assert result == 0

    # Completion test
    result = eval_completions(eval_results=[PARITY_PROMPT])
    assert result == [ExecResult.VALID]

    # Timeout test
    PROMPT = """
import time
def test_func():
    print("Hello World!")
    time.sleep(1.1)
    return 5
"""
    result = pool_exec_processes(PROMPT, "test_func", timeout=1.0, debug=False)[0]
    assert result == ExecResult.TIMEOUT_EXCEPTION

    # Syntax error test
    PROMPT = """
def test_func():
test_dct = {}
    return 0
"""
    result = pool_exec_processes(PROMPT, "test_func", debug=False)[0]
    assert result == ExecResult.SYNTAX_ERROR

    # Type error test
    PROMPT = """
def test_func():
    result = 1 + "Hello World"
    return 0
"""
    result = pool_exec_processes(PROMPT, "test_func", debug=False)[0]
    assert result == ExecResult.TYPE_ERROR

    # Other exception test
    PROMPT = """
def test_func():
    raise StopIteration
    result = 5 / 0
    return 0
"""
    result = pool_exec_processes(PROMPT, "test_func", debug=False)[0]
    assert result == ExecResult.EXCEPTION
