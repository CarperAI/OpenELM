"""
This file contains some helper functions to parse python programs. They might not be absolutely essential to the reimplementation of ELM, but
    hopefully they simplify some processes (turning 4-parity/quadratic to GPTree) and provide a code base for future generalizations.
"""

from tree_sitter import Language, Parser
from typing import Callable
import inspect
import pygraphviz as pgv
import os

PY_LANGUAGE = Language(os.path.dirname(__file__) + '/src/python_parsing_lib.so', 'python')


def draw_func_tree(func, save_path='tree.png'):
    """
    Draws the naiive syntax tree parsed by pytree-sitter and save to a png file.
    Parameters:
        func: The function to be parsed. Can be either an actual function or a string.
        save_path: The path to the saved png image.
    """
    parser = Parser()
    parser.set_language(PY_LANGUAGE)

    if isinstance(func, Callable):
        tree = parser.parse(bytes(inspect.getsource(func), 'utf8'))
    elif isinstance(func, (bytes, str)):
        tree = parser.parse(bytes(func, 'utf8'))

    cursor = tree.walk()

    graph = pgv.AGraph()

    construct_graph(graph, None, cursor)
    graph.layout('dot')
    graph.draw(save_path)

    return graph


def construct_graph(graph, node_id, cursor):
    if not cursor.goto_first_child():
        return

    while True:
        new_id = graph.number_of_nodes()
        graph.add_node(new_id, label=f'{cursor.node.type}, {shorten(cursor.node.text)}')
        if node_id is not None:
            graph.add_edge(node_id, new_id)

        construct_graph(graph, new_id, cursor)
        
        if not cursor.goto_next_sibling():
            break

    cursor.goto_parent()


def shorten(st, length=5):
    if isinstance(st, bytes):
        st = st.decode('utf-8')
    return st[:5] + '...' if len(st) > 5 else st

