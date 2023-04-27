import json
from pathlib import Path

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow to run")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        # --runslow given in cli: do not skip slow tests
        return
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


@pytest.fixture
def square_walker_dict():
    file = Path(Path.cwd() / "tests" / "test_data" / "square.json")
    with file.open() as f:
        contents = json.load(f)
    return contents


@pytest.fixture
def cppn_fixed_walker_dict():
    file = Path(Path.cwd() / "tests" / "test_data" / "cppn_fixed.json")
    with file.open() as f:
        contents = json.load(f)
    return contents
