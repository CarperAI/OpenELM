"""
This file contains some helper functions to parse python programs and visualize. They are not essential to the
    reimplementation of ELM, but if one wants to generalize the traditional GP experiments to arbitrary functions,
    hopefully this provides the code base for syntax parsing.
For example, if we expand the TinyGP experiment to arbitrary functions, we can modify `GPTree` to non-binary trees,
    and use the `tree.walk()` cursor demonstrated below to traverse (will also need to filter out some unwanted nodes).
    The visualization code below will also be necessary for development and testing. But I am not sure if we want to
    generalize the non-LM GP benchmarks. So I am leaving it here until the need arises in the future.
"""

import inspect
import os
from typing import Callable

import pygraphviz as pgv
from tree_sitter import Language, Parser, TreeCursor

PY_LANGUAGE = Language(os.path.dirname(__file__) + "/src/python_lib.so", "python")


def draw_func_tree(func, save_path="tree.png"):
    """
    Draws the naiive syntax tree parsed by pytree-sitter and save to a png file.
    Parameters:
        func: The function to be parsed. Can be either an actual function or a string.
        save_path: The path to the saved png image.
    Returns:
        the pygraphviz AGraph object (note that in the meantime a picture will be saved).
    """
    parser = Parser()
    parser.set_language(PY_LANGUAGE)

    if isinstance(func, Callable):
        tree = parser.parse(bytes(inspect.getsource(func), "utf8"))
    elif isinstance(func, (bytes, str)):
        tree = parser.parse(bytes(func, "utf8"))
    else:
        raise TypeError(
            f"func must be a function, a string or a bytes object. Got {type(func)}."
        )

    cursor = tree.walk()

    graph = pgv.AGraph()

    construct_graph(graph, None, cursor)
    graph.layout("dot")
    graph.draw(save_path)

    return graph


def construct_graph(graph: pgv.AGraph, node_id: int, cursor: TreeCursor):
    """
    Recursively construct the pygraphviz graph according to the TreeCursor.
    Parameters:
        graph: the pygraphviz AGraph object.
        node_id: current node id.
        cursor: the pytree-sitter TreeCursor object.
    """
    if not cursor.goto_first_child():
        return

    while True:
        new_id = graph.number_of_nodes()
        graph.add_node(new_id, label=f"{cursor.node.type}, {shorten(cursor.node.text)}")
        if node_id is not None:
            graph.add_edge(node_id, new_id)

        construct_graph(graph, new_id, cursor)

        if not cursor.goto_next_sibling():
            break

    cursor.goto_parent()


def shorten(st, length=5):
    if isinstance(st, bytes):
        st = st.decode("utf-8")
    return st[:5] + "..." if len(st) > 5 else st
