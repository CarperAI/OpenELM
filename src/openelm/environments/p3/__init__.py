P3_PROBLEM_MED_SEED = '''from typing import List

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

assert f5(g5())'''

P3_PROBLEM_LONG_SEED = '''from typing import List

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

assert f5(g5())'''

P3_PROBSOL_MED_SEED = '''from typing import List

def f1_1(s: str):
    return "Hello " + s == "Hello world"

def g1_1():
    return "world"

assert f1_1(g1_1())

def f1_2(s: str):
    """Changes the requirements of f1_1"""
    return "Hello " + s == "Hello to the world"

def g1_2():
    return "to the world"

assert f1_2(g1_2())

def f2_1(s: str):
    return "Hello " + s[::-1] == "Hello world"

def g2_1():
    return "world"[::-1]

assert f2_1(g2_1())

def f2_2(s: str):
    """Changes the requirements of f2_1"""
    return s[::-1].swapcase() + " world" == "Hello world"

def g2_2():
    return "Hello"[::-1].swapcase()

assert f2_2(g2_2())

def f3_1(x: List[int]):
    return len(x) == 2 and sum(x) == 3

def g3_1():
    return [1, 2]

assert f3_1(g3_1())

def f3_2(x: List[int]):
    """Changes the requirements of f3_1"""
    return len(x) == 2 and and x[0]+x[1] == 8 and x[0]*x[1] == 12

def g3_2():
    return [2, 6]

assert f3_2(g3_2())

def f4_1(s: List[str]):
    return len(set(s)) == 1000 and all(
        (x.count("a") > x.count("b")) and ('b' in x) for x in s)

def g4_1():
    return ["a"*(i+2)+"b" for i in range(1000)]

assert f4_1(g4_1())

def f4_2(s: List[str]):
    """Changes the requirements of f4_1"""
    return len(set(s)) == 1000 and all(
        (x.count("a") > x.count("b")) and ('b' in x) and (x.count('c')==2) and (x.startswith('cc')) for x in s)

def g4_2():
    return ["cc"+"a"*(i+2)+"b" for i in range(1000)]

assert f4_2(g4_2())

def f5_1(n: int):
    return str(n * n).startswith("123456789")

def g5_1():
    return int(int("123456789" + "0"*9) ** 0.5) + 1

assert f5_1(g5_1())

def f5_2(n: int):
    """Changes the requirements of f5_1"""
    return str((n-10) * (n-10)).startswith("123456789")

def g5_2():
    return (int(int("123456789" + "0"*9) ** 0.5) + 1) + 10

assert f5_2(g5_2())'''

P3_PROBSOL_LONG_SEED = '''from typing import List

def f1_1(s: str):
    return "Hello " + s == "Hello world"

def g1_1():
    """Find a string that when concatenated onto 'Hello ' gives 'Hello world'."""
    return "world"

assert f1_1(g1_1())

def f1_2(s: str):
    """Changes from f1_1: 'to the world' instead of 'world'."""
    return "Hello " + s == "Hello to the world"

def g1_2():
    """Find a string that when concatenated onto 'Hello ' gives 'Hello to the world'."""
    return "to the world"

assert f1_2(g1_2())

def f2_1(s: str):
    return "Hello " + s[::-1] == "Hello world"

def g2_1():
    """Find a string that when reversed and concatenated onto 'Hello ' gives 'Hello world'."""
    return "world"[::-1]

assert f2_1(g2_1())

def f2_2(s: str):
    """Changes from f2_1: swapcase and preprend the given string instead of concatenate to create the phrase"""
    return s[::-1].swapcase() + " world" == "Hello world"

def g2_2():
    """Find a string that when reversed, swapcased, and prepended onto ' world' gives 'Hello world'."""
    return "Hello"[::-1].swapcase()

assert f2_2(g2_2())

def f3_1(x: List[int]):
    return len(x) == 2 and sum(x) == 3

def g3_1():
    """Find a list of two integers whose sum is 3."""
    return [1, 2]

assert f3_1(g3_1())

def f3_2(x: List[int]):
    """Changes from f3_1: change sum to 8 and add requirement for product to equal 12"""
    return len(x) == 2 and and x[0]+x[1] == 8 and x[0]*x[1] == 12

def g3_2():
    """Find a list of two integers whose sum is 8 and product is 12."""
    return [2, 6]

assert f3_2(g3_2())

def f4_1(s: List[str]):
    return len(set(s)) == 1000 and all(
        (x.count("a") > x.count("b")) and ('b' in x) for x in s)

def g4_1():
    """Find a list of 1000 distinct strings which each have more 'a's than 'b's and at least one 'b'."""
    return ["a"*(i+2)+"b" for i in range(1000)]

assert f4_1(g4_1())

def f4_2(s: List[str]):
    """Changes from f4_1: add requirement for exactly two 'c's in the front"""
    return len(set(s)) == 1000 and all(
        (x.count("a") > x.count("b")) and ('b' in x) and (x.count('c')==2) and (x.startswith('cc')) for x in s)

def g4_2():
    """Find a list of 1000 distinct strings which each have more 'a's than 'b's and at least one 'b' and exactly two 'c's which are in the front."""
    return ["cc"+"a"*(i+2)+"b" for i in range(1000)]

assert f4_2(g4_2())

def f5_1(n: int):
    return str(n * n).startswith("123456789")

def g5_1():
    """Find an integer whose perfect square begins with 123456789 in its decimal representation."""
    return int(int("123456789" + "0"*9) ** 0.5) + 1

assert f5_1(g5_1())

def f5_2(n: int):
    """Changes from f5_1: 10 must be subtracted from the integer first"""
    return str((n-10) * (n-10)).startswith("123456789")

def g5_2():
    """Find an integer for which the output of subtracting 10 and squaring the result begins with 123456789 in its decimal representation."""
    return (int(int("123456789" + "0"*9) ** 0.5) + 1) + 10

assert f5_2(g5_2())'''

P3_IMPORTS = "from typing import List\n" # The only import that's necessary as of P3 v0.2

__all__ = [
    "P3_PROBLEM_MED_SEED",
    "P3_PROBLEM_LONG_SEED",
    "P3_PROBSOL_LONG_SEED",
    "P3_IMPORTS"
]
