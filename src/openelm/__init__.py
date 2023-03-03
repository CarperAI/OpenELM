from importlib.metadata import version as importlib_version

from openelm.elm import ELM

__version__ = importlib_version("openelm")

__all__ = ["ELM"]
