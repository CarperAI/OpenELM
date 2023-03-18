import pytest

from openelm.configs import ImageEnvConfig, StringEnvConfig
from openelm.environments.environments import (
    BaseEnvironment,
    FunctionOptim,
    ImageOptim,
    MatchString,
)
from openelm.map_elites import MAPElites


@pytest.mark.slow
def test_string_matching():
    env: BaseEnvironment = MatchString(StringEnvConfig())
    elites = MAPElites(env, map_grid_size=(3,), history_length=10)
    result = elites.search(init_steps=10_000, total_steps=100_000)
    # elites.plot()
    assert result == "MAPElites"


@pytest.mark.skip(reason="unimplemented")
def test_function_optim():
    env: BaseEnvironment = FunctionOptim(ndim=2)
    elites = MAPElites(env, map_grid_size=(128,), history_length=10)
    assert elites.search(init_steps=5_000, total_steps=50_000, atol=0) == 1.0

    # elites.plot()


@pytest.mark.skip(reason="unimplemented")
def test_image_optim():
    env = ImageOptim(ImageEnvConfig())
    elites = MAPElites(env, n_bins=2, history_length=10)

    print("Best image", elites.search(initsteps=5, totalsteps=10))
