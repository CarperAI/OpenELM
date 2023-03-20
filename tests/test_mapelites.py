import numpy as np
import pytest

from openelm.configs import ImageEnvConfig, StringEnvConfig
from openelm.environments.environments import (
    BaseEnvironment,
    FunctionOptim,
    ImageOptim,
    MatchString,
)
from openelm.map_elites import GridMap, MAPElites


def test_map():
    behavior_ndim = 2
    map_grid_size = [3]
    history_length = 4
    fitnesses = GridMap(
        dims=map_grid_size * behavior_ndim,
        fill_value=-np.inf,
        dtype=float,
        history_length=history_length,
    )
    fitnesses[0, 0] = -1.
    fitnesses[0, 0] = -2.

    fitnesses[1, 1] = 1.
    fitnesses[1, 1] = 2.
    fitnesses[1, 1] = 3.
    fitnesses[1, 1] = 4.
    fitnesses[1, 1] = 5.  # should wrap around

    fitnesses[2, 2] = 1.

    fitnesses[1, 2] = 1.
    fitnesses[1, 2] = -1.
    fitnesses[1, 2] = 2.
    fitnesses[1, 2] = -2.

    assert fitnesses.shape == (4, 3, 3)
    assert fitnesses.top.shape == (3, 3)
    assert fitnesses.top[0, 0] == 1.
    assert fitnesses.top[1, 1] == 0.
    assert fitnesses.top[2, 2] == 0.
    assert fitnesses.top[1, 2] == 3.
    assert fitnesses.top[2, 1] == 3.

    latest = fitnesses.latest
    assert latest.shape == (3, 3)
    assert latest[0, 0] == -2.
    assert latest[1, 1] == 5.
    assert latest[2, 2] == 1.
    assert latest[1, 2] == -2.
    assert latest[2, 1] == -np.inf

    assert fitnesses.min == -2.
    assert fitnesses.max == 5.
    assert fitnesses.mean == (-2. + 5 + 1 - 2) / 4


@pytest.mark.slow
def test_string_matching():
    target_string = "AAA"
    env: BaseEnvironment = MatchString(StringEnvConfig(target=target_string, batch_size=1))
    elites = MAPElites(env, map_grid_size=(2,), history_length=10)
    result = elites.search(init_steps=100, total_steps=3000)
    elites.plot()
    assert result == target_string

    # ordering of characters is abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ
    # the splitting point for the bins is at A (26.0)
    # we expect all niches to converge around this point

    print('genes', *[g.to_phenotype() for g in elites.genomes.latest.flatten().tolist()])
    print('bins', elites.bins)
    genomes = [str(g) for g in elites.genomes.latest.flatten().tolist()]
    assert len(genomes) == 8
    for g in genomes:
        assert g == target_string
    # print(genomes)


@pytest.mark.slow
def test_string_matching2():
    target_string = "Evolve"
    env: BaseEnvironment = MatchString(StringEnvConfig(target=target_string, batch_size=8))
    elites = MAPElites(env, map_grid_size=(3,), history_length=1)
    result = elites.search(init_steps=2_000, total_steps=20_000)
    elites.plot()
    assert result == target_string


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
