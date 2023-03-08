from typing import Any

from openelm.environments.environments import (
    BaseEnvironment,
    FunctionOptim,
    Genotype,
    ImageOptim,
    MatchString,
    P3Problem,
    Sodarace,
)

P3_MED_SEED = {
    "program_str": """from typing import List

def f1(s: str):
    return "Hello " + s == "Hello world"

def g1():
    return "world"

assert f1(g1())

def f2(s: str):
    return "Hello " + s[::-1] == "Hello world"

def g2():
    return "world"[::-1]

assert f2(g2())

def f3(x: List[int]):
    return len(x) == 2 and sum(x) == 3

def g3():
    return [1, 2]

assert f3(g3())

def f4(s: List[str]):
    return len(set(s)) == 1000 and all(
        (x.count("a") > x.count("b")) and ('b' in x) for x in s)

def g4():
    return ["a"*(i+2)+"b" for i in range(1000)]

assert f4(g4())

def f5(n: int):
    return str(n * n).startswith("123456789")

def g5():
    return int(int("123456789" + "0"*9) ** 0.5) + 1

assert f5(g5())""",
    "result_obj": {},
}

P3_LONG_SEED = {
    "program_str": '''from typing import List

def f1(s: str):
    return "Hello " + s == "Hello world"

def g1():
    """Find a string that when concatenated onto 'Hello ' gives 'Hello world'."""
    return "world"

assert f1(g1())

def f2(s: str):
    return "Hello " + s[::-1] == "Hello world"

def g2():
    """Find a string that when reversed and concatenated onto 'Hello ' gives 'Hello world'."""
    return "world"[::-1]

assert f2(g2())

def f3(x: List[int]):
    return len(x) == 2 and sum(x) == 3

def g3():
    """Find a list of two integers whose sum is 3."""
    return [1, 2]

assert f3(g3())

def f4(s: List[str]):
    return len(set(s)) == 1000 and all(
        (x.count("a") > x.count("b")) and ('b' in x) for x in s)

def g4():
    """Find a list of 1000 distinct strings which each have more 'a's than 'b's and at least one 'b'."""
    return ["a"*(i+2)+"b" for i in range(1000)]

assert f4(g4())

def f5(n: int):
    return str(n * n).startswith("123456789")

def g5():
    """Find an integer whose perfect square begins with 123456789 in its decimal representation."""
    return int(int("123456789" + "0"*9) ** 0.5) + 1

assert f5(g5())''',
    "result_obj": {},
}

p3_med_init_args = {
    "seed": P3_MED_SEED,
    "config": "openelm/config/elm_p3_cfg.yaml",
    "diff_model": None,
}

p3_long_init_args = {
    "seed": P3_LONG_SEED,
    "config": "openelm/config/elm_p3_cfg.yaml",
    "diff_model": None,
}

ENVS_DICT: dict[str, Any] = {
    "sodarace": Sodarace,
    "image_evolution": ImageOptim,
    "p3": P3Problem,
}

__all__ = [
    "Genotype",
    "BaseEnvironment",
    "FunctionOptim",
    "ImageOptim",
    "MatchString",
    "Sodarace",
    "ENVS_DICT",
    "P3Problem",
]
