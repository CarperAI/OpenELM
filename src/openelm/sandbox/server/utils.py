import os
import re
import shutil

from openelm.sandbox.server.sandbox_codex_execute import (
    TimeoutException,
    create_tempdir,
    reliability_guard,
    swallow_io,
    time_limit,
)


def reset_os_funcs(rmtree, rmdir, chdir):
    shutil.rmtree = rmtree
    os.rmdir = rmdir
    os.chdir = chdir


def sandbox_unsafe_execute(
    code_str: str, func_name: str, timeout: float = 5.0, debug=False
):
    if len(code_str) == 0 or "def " not in code_str:
        print("No code found or no function found.")
        return 6  # No code found or no function found.
    code_dct: dict = {}
    func_match = re.search(r"def (\w+)\s*\((.*?)\):", code_str)
    if not func_match:
        print("No proper function found in code.")
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
                if timeout <= 0.0:
                    exec(code_str, globals(), code_dct)
                    for k, v in code_dct.items():
                        globals()[k] = v
                    res = code_dct[func_name]()
                    reset_os_funcs(rmtree, rmdir, chdir)
                    return res
                else:
                    with time_limit(timeout):
                        exec(code_str, globals(), code_dct)
                        for k, v in code_dct.items():
                            globals()[k] = v
                        res = code_dct[func_name]()
                        reset_os_funcs(rmtree, rmdir, chdir)
                        return res
        except TimeoutException as e:
            reset_os_funcs(rmtree, rmdir, chdir)
            if debug:
                print("Code takes too long to run.")
                print(e)
                print(code_str)
            return 2  # Code takes too long to run.
        except RuntimeError as e:
            reset_os_funcs(rmtree, rmdir, chdir)
            if debug:
                print("Code runs but crashes.")
                print(e)
                print(code_str)
            return 3  # Code runs but crashes.
        except SyntaxError as e:
            reset_os_funcs(rmtree, rmdir, chdir)
            if debug:
                print("Code does not run - syntax error.")
                print(e)
                print(code_str)
            return 4  # Code does not run - syntax error.
        except TypeError as e:
            reset_os_funcs(rmtree, rmdir, chdir)
            if debug:
                print("Code does not run - type error.")
                print(e)
                print(code_str)
            return 5  # Code does not run - type error.
        except Exception as e:
            reset_os_funcs(rmtree, rmdir, chdir)
            if debug:
                print("Code fails to run - other error.")
                print(e)
                print(code_str)
            return 6  # Code fails to run - other error.
