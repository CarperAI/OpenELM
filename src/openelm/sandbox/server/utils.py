import re
from enum import Enum
from typing import Any, Optional

from openelm.sandbox.server.sandbox_codex_execute import (
    TimeoutException,
    create_tempdir,
    reliability_guard,
    swallow_io,
    time_limit,
)


class ErrorCode(Enum):
    """An Enum to represent the errors for the execution of generated code."""

    VALID = 0
    TEST_FAILED = 1
    TIMEOUT_EXCEPTION = 2
    SYNTAX_ERROR = 3
    TYPE_ERROR = 4
    EXCEPTION = 5


def sandbox_unsafe_execute(
    code_str: str,
    func_name: Optional[str] = None,
    args: Optional[dict[str, Any]] = None,
    timeout: float = 5.0,
    debug: bool = False,
):
    if len(code_str) == 0 or "def " not in code_str:
        print("No code found or no function found.")
        return ErrorCode(5)  # No code found or no function found.
    func_match = re.search(r"def (\w+)\s*\((.*?)\):", code_str)
    if not func_match:
        print("No proper function found in code.")
        return ErrorCode(5)  # No proper function found in code.
    elif func_match and func_name is None:
        func_name = func_match.groups()[0]
    with create_tempdir():

        # These system calls are needed when cleaning up tempdir.
        import os
        import shutil

        rmtree = shutil.rmtree
        rmdir = os.rmdir
        chdir = os.chdir

        # Disable functionalities that can make destructive changes.
        reliability_guard()

        try:
            # TODO: Check https://arxiv.org/abs/2209.07753 code.
            code_dct: dict = {}
            with swallow_io():
                with time_limit(timeout):
                    exec(code_str, code_dct)
                    if args is None:
                        result = code_dct[func_name]()
                    else:
                        result = code_dct[func_name](**args)
        except TimeoutException as e:
            if debug:
                print(type(e), e.args)
                print(code_str)
            result = ErrorCode(2)
        except SyntaxError as e:
            if debug:
                print(type(e), e.args)
                print(code_str)
            result = ErrorCode(3)
        except TypeError as e:
            if debug:
                print(type(e), e.args)
                print(code_str)
            result = ErrorCode(4)
        except Exception as e:
            if debug:
                print(type(e), e.args)
                print(code_str)
            result = ErrorCode(5)

        shutil.rmtree = rmtree
        os.rmdir = rmdir
        os.chdir = chdir

        return result
