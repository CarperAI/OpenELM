# This file steals codes from https://github.com/moshesipper/tiny_gp and modifies it into one of ELM benchmarks.
# Original header below:

# tiny genetic programming plus, by © moshe sipper, www.moshesipper.com
# graphic output, dynamic progress display, bloat-control option 
# need to install https://pypi.org/project/graphviz/

import itertools
from copy import deepcopy
from random import random, randint
from statistics import mean
from typing import Iterable, List, Tuple

from IPython.display import Image, display
from graphviz import Digraph, Source

MIN_DEPTH = 0  # minimal initial random tree depth
PROB_MUTATION = 0.2  # per-node mutation probability
NUM_MUTATION_PER_LOOP = 10000  # the number of mutations to be run per step of the experiments


# TODO: modularize functions and FUNCTIONS, TERMINALS lists so that they can be more easily swapped for different uses.
def add(x, y): return x + y


def mod(x, y): return x % y


def mul(x, y): return x * y


def subtract(x, y): return x - y


FUNCTIONS = [add, mod]
TERMINALS = [f'b{i}' for i in range(1, 5)] + [f'c{i}' for i in range(1, 5)] + [-2, -1, 0, 1, 2]


def four_parity_reference(b1, b2, b3, b4):
    bit_sum = sum([b1, b2, b3, b4])
    return bit_sum % 2


def quadratic_references(a, b, c, x):
    return a * pow(x, 2) + b * x + c


def generate_dataset(target_fn=four_parity_reference):
    inputs = [i for i in itertools.product(range(2), repeat=4)]
    ground_truth = [target_fn(*i) for i in inputs]
    return inputs, ground_truth


class GPTree:
    def __init__(self, data=None, left=None, right=None):
        self.data = data
        self.left = left
        self.right = right

    def node_label(self):  # return string label
        if (self.data in FUNCTIONS):
            return self.data.__name__
        else:
            return str(self.data)

    def draw(self, dot, count):  # dot & count are lists in order to pass "by reference"
        node_name = str(count[0])
        dot[0].node(node_name, self.node_label())
        if self.left:
            count[0] += 1
            dot[0].edge(node_name, str(count[0]))
            self.left.draw(dot, count)
        if self.right:
            count[0] += 1
            dot[0].edge(node_name, str(count[0]))
            self.right.draw(dot, count)

    def draw_tree(self, fname, footer):
        dot = [Digraph(fname, comment=footer)]
        count = [0]
        self.draw(dot, count)
        dot[0].format = 'png'
        dot[0].render(directory='')
        display(Image(filename = f'{fname}.gv.png'))
        return dot[0].source

    def compute_tree(self, b, arg_names=('b1', 'b2', 'b3', 'b4')):
        """
        Parameters:
            b: a list/tuple of inputs (b1, b2, b3, b4)
            arg_names: argument names.
        Returns:
            the evaluation at this node.
        """
        if not isinstance(b, (List, Tuple)):
            raise TypeError(f"Input b must be a list or tuple. Got {type(b)} instead.")

        arg_dict = {name: value for name, value in zip(arg_names, b)}

        if self.data in FUNCTIONS:
            return self.data(self.left.compute_tree(b), self.right.compute_tree(b))
        elif isinstance(self.data, str):
            return arg_dict[self.data]  # it will take care of errors like unrecognized variable 'c1', etc.
        else:
            return self.data  # self.data is a number

    def random_tree(self, grow, max_depth, depth=0):  # create random tree using either grow or full method
        if depth < MIN_DEPTH or (depth < max_depth and not grow):
            self.data = FUNCTIONS[randint(0, len(FUNCTIONS) - 1)]
        elif depth >= max_depth:
            self.data = TERMINALS[randint(0, len(TERMINALS) - 1)]
        else:  # intermediate depth, grow
            if random() > 0.5:
                self.data = TERMINALS[randint(0, len(TERMINALS) - 1)]
            else:
                self.data = FUNCTIONS[randint(0, len(FUNCTIONS) - 1)]
        if self.data in FUNCTIONS:
            self.left = GPTree()
            self.left.random_tree(grow, max_depth, depth=depth + 1)
            self.right = GPTree()
            self.right.random_tree(grow, max_depth, depth=depth + 1)

    def mutation(self):
        if random() < PROB_MUTATION:  # mutate at this node
            self.random_tree(grow=True, max_depth=2)
        elif self.left:
            self.left.mutation()
        elif self.right:
            self.right.mutation()

    def size(self):  # tree size in nodes
        if self.data in TERMINALS: return 1
        l = self.left.size() if self.left else 0
        r = self.right.size() if self.right else 0
        return 1 + l + r

    def build_subtree(self):  # count is list in order to pass "by reference"
        t = GPTree()
        t.data = self.data
        if self.left:  t.left = self.left.build_subtree()
        if self.right: t.right = self.right.build_subtree()
        return t

    def scan_tree(self, count, second):  # note: count is list, so it's passed "by reference"
        count[0] -= 1
        if count[0] <= 1:
            if not second:  # return subtree rooted here
                return self.build_subtree()
            else:  # glue subtree here
                self.data = second.data
                self.left = second.left
                self.right = second.right
        else:
            ret = None
            if self.left and count[0] > 1: ret = self.left.scan_tree(count, second)
            if self.right and count[0] > 1: ret = self.right.scan_tree(count, second)
            return ret

# end class GPTree


def error(individual, dataset):
    return mean([abs(individual.compute_tree(ds[0]) - ds[1]) for ds in dataset])


def generate_four_parity():
    """
    Hand-coded four_parity.
    Returns:
        the GPTree object that implements four_parity.
    """
    root = GPTree(data=mod, right=GPTree(data=2))
    prev_node = GPTree(data=add, left=GPTree(data='b3'), right=GPTree(data='b4'))
    for i in range(2, 0, -1):
        node = GPTree(data=add, left=GPTree(data=f'b{i}'), right=prev_node)
        prev_node = node
    root.left = prev_node

    return root


def generate_quadratic():
    """
    Hand-coded quadratic.
    Returns:
        the GPTree object that implements quadratic.
    """
    root = GPTree(data=add, right=GPTree(data='c'))

    node_1 = GPTree(data=pow, left=GPTree(data='x'), right=GPTree(data=2))
    node_2 = GPTree(data=mul, left=GPTree(data='a'), right=node_1)
    node_3 = GPTree(data=add, left=node_2, right=GPTree(data=mul,
                                                        left=GPTree(data='b'),
                                                        right=GPTree(data='x')))
    root.left = node_3

    return root


def swap_node(tree: GPTree, var_name: str, target_name: str) -> bool:
    """
    Swap the name of a variable into another one (only apply to the first encounter of a DFS).
    Parameters:
        tree: the GPTree node.
        var_name: the variable name.
        target_name: the target variable name.
    Returns:
        True if the variable is found and replaced.
    """
    if tree is None:
        return False

    if tree.data == var_name:
        tree.data = target_name
        return True

    if not swap_node(tree.left, var_name, target_name):
        return swap_node(tree.right, var_name, target_name)
    return True


def eval_tree(tree: GPTree, dataset: Iterable) -> list:
    """
    Test the correctness of a GPTree against a dataset.
    Parameters:
        tree: the tree to test against.
        dataset: (inputs, ground_truth)
    Returns
        the list of evaluation results.
        (I'm trying to match the error codes here:
            0: pass.
            1: error in code, but runs.
            2: fails to run, e.g., when it has unrecognized variable name like 'c1'.)
    """
    results = []
    for data in zip(*dataset):
        try:
            output = tree.compute_tree(data[0])
            results.append(0 if output == data[1] else 1)  # right or wrong, but no error.
        except Exception:
            results.append(2)  # Fails to run.

    return results


def list_equal(l1, l2):
    return all([x == y for x, y in zip(l1, l2)])


def main():
    tree = generate_four_parity()
    dataset = generate_dataset()

    # According to the paper, the error is gradually introduced by replacing b-variables to c-variables step-by-step.
    # At the very end, modulo 2 is replaced by modulo 3. We carry the experiments out here and collect the percentage
    #   of mutations that fix the problem.

    for i in range(4):
        swap_node(tree, f'b{i+1}', f'c{i+1}')

        corrected = 0
        for j in range(NUM_MUTATION_PER_LOOP):
            tree_copy = deepcopy(tree)
            tree_copy.mutation()
            corrected += list_equal(eval_tree(tree_copy, dataset), [0] * len(dataset[1]))

        print(f'Step {i+1}, percentage of successful mutations: {corrected / NUM_MUTATION_PER_LOOP}')


if __name__ == "__main__":
    main()
