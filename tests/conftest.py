import json
from pathlib import Path

import pytest


@pytest.fixture
def square_walker_dict():
    file = Path(Path.cwd() / 'tests' / 'test_data' / 'square.json')
    with file.open() as f:
        contents = json.load(f)
    return contents


@pytest.fixture
def cppn_fixed_walker_dict():
    file = Path(Path.cwd() / 'tests' / 'test_data' / 'cppn_fixed.json')
    with file.open() as f:
        contents = json.load(f)
    return contents
