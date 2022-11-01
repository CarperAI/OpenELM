# %%
import os
import re
import shutil
from typing import Dict
from .codex_execute import (TimeoutException, create_tempdir, reliability_guard,
# from codex_execute import (TimeoutException, create_tempdir, reliability_guard,
                           swallow_io, time_limit)


test_code = """
from .walker.walk_creator import walker_creator
def make_square(wc, x0, y0, x1, y1):
    \"\"\"Make a square with top left x0,y0 and top right x1,y1.\"\"\"
    j0 = wc.add_joint(x0, y0)
    j1 = wc.add_joint(x0, y1)
    j2 = wc.add_joint(x1, y1)
    j3 = wc.add_joint(x1, y0)
    return j0, j1, j2, j3

def make_walker():
    wc = walker_creator()

    # the main body is a square
    sides = make_square(wc, 0, 0, 10, 10)
    center = wc.add_joint(5, 5)

    # connect the square with distance muscles
    for k in range(len(sides) - 1):
        wc.add_muscle(sides[k], sides[k + 1])
    wc.add_muscle(sides[3], sides[0])

    # one prong of the square is a distance muscle
    wc.add_muscle(sides[3], center)

    # the other prongs from the center of the square are active
    wc.add_muscle(sides[0], center, False, 5.0, 0.0)
    wc.add_muscle(sides[1], center, False, 10.0, 0.0)
    wc.add_muscle(sides[2], center, False, 2.0, 0.0)

    return wc.get_walker()


"""

def reset_os_funcs(rmtree, rmdir, chdir):
    shutil.rmtree = rmtree
    os.rmdir = rmdir
    os.chdir = chdir


def sandbox_unsafe_execute(code_str: str, func_name:str,  timeout: int = 5):
    if len(code_str) == 0 or "def " not in code_str:
        print("No code found or no function found.")
        return 6  # No code found or no function found.
    code_dct: Dict = {}
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
                # with time_limit(timeout):
                exec(code_str, globals(), code_dct)
                for k,v in code_dct.items():
                    globals()[k] = v
                res = code_dct[func_name]()
                reset_os_funcs(rmtree, rmdir, chdir)
                return res
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
            print("Code fails to run - other error.", code_dct)
            return 6  # Code fails to run - other error.
    
    # %%
# unsafe_execute(code_str, timeout=5)