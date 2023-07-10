import shutil
from pathlib import Path

import numpy as np
import pytest

from openelm.algorithms.map_elites import CVTMAPElites, Map, MAPElites
from openelm.configs import (
    CVTMAPElitesConfig,
    ImageEnvConfig,
    MAPElitesConfig,
    PromptModelConfig,
    StringEnvConfig,
)
from openelm.environments.base import (
    BaseEnvironment,
    FunctionOptim,
    ImageOptim,
    MatchString,
)
from openelm.mutation_model import PromptModel


def test_map():
    behavior_ndim = 2
    map_grid_size = [3]
    history_length = 4
    fitnesses = Map(
        dims=map_grid_size * behavior_ndim,
        fill_value=-np.inf,
        dtype=float,
        history_length=history_length,
    )
    fitnesses[0, 0] = -1.0
    fitnesses[0, 0] = -2.0

    fitnesses[1, 1] = 1.0
    fitnesses[1, 1] = 2.0
    fitnesses[1, 1] = 3.0
    fitnesses[1, 1] = 4.0
    fitnesses[1, 1] = 5.0  # should wrap around

    fitnesses[2, 2] = 1.0

    fitnesses[1, 2] = 1.0
    fitnesses[1, 2] = -1.0
    fitnesses[1, 2] = 2.0
    fitnesses[1, 2] = -2.0
    fitnesses[0, 2] = np.inf

    assert fitnesses.shape == (4, 3, 3)
    assert fitnesses.top.shape == (3, 3)
    assert fitnesses.top[0, 0] == 1.0
    assert fitnesses.top[1, 1] == 0.0
    assert fitnesses.top[2, 2] == 0.0
    assert fitnesses.top[1, 2] == 3.0
    assert fitnesses.top[2, 1] == 3.0
    assert fitnesses.top[0, 2] == 0.0

    latest = fitnesses.latest
    assert latest.shape == (3, 3)
    assert latest[0, 0] == -2.0
    assert latest[1, 1] == 5.0
    assert latest[2, 2] == 1.0
    assert latest[1, 2] == -2.0
    assert latest[2, 1] == -np.inf
    assert latest[0, 2] == np.inf

    assert fitnesses.min == -np.inf
    assert fitnesses.min_finite == -2.0
    assert fitnesses.max == np.inf
    assert fitnesses.max_finite == 5.0
    assert fitnesses.mean == (-2.0 + 5 + 1 - 2) / 4


def test_empty_map():
    behavior_ndim = 2
    map_grid_size = [3]
    history_length = 4
    fitnesses = Map(
        dims=map_grid_size * behavior_ndim,
        fill_value=-np.inf,
        dtype=float,
        history_length=history_length,
    )

    assert fitnesses.min == -np.inf
    assert np.isnan(fitnesses.min_finite)
    assert fitnesses.max == -np.inf
    assert np.isnan(fitnesses.max_finite)


def test_empty_mapelites():
    target_string = "AAA"
    env: BaseEnvironment = MatchString(
        StringEnvConfig(target=target_string, batch_size=1)
    )
    elites = MAPElites(env, MAPElitesConfig(map_grid_size=(2,), history_length=10))
    assert elites.fitnesses.min == -np.inf
    assert np.isnan(elites.fitnesses.min_finite)
    assert elites.fitnesses.max == -np.inf
    assert np.isnan(elites.fitnesses.max_finite)
    # elites.plot_fitness()


@pytest.mark.slow
def test_cvt():
    target_string = "AAA"
    env: BaseEnvironment = MatchString(
        StringEnvConfig(target=target_string, batch_size=1)
    )
    elites = CVTMAPElites(env, CVTMAPElitesConfig(n_niches=5, history_length=100))
    result = elites.search(init_steps=10, total_steps=3000)

    assert result == target_string

    # Since the CVT is random, results may be less consistent than grid
    # TODO: figure out a test for this


@pytest.mark.slow
def test_cvt2():
    target_string = "Evolve"
    env: BaseEnvironment = MatchString(
        StringEnvConfig(target=target_string, batch_size=10)
    )
    elites = CVTMAPElites(env, CVTMAPElitesConfig(n_niches=10, history_length=100))
    result = elites.search(init_steps=20, total_steps=5000)

    elites.plot_fitness()
    elites.plot_behaviour_space()
    assert result == target_string


@pytest.mark.slow
def test_string_matching():
    target_string = "AAA"
    env: BaseEnvironment = MatchString(
        StringEnvConfig(target=target_string, batch_size=1)
    )
    elites = MAPElites(env, MAPElitesConfig(map_grid_size=(2,), history_length=10))
    result = elites.search(init_steps=100, total_steps=3000)
    elites.plot_fitness()
    assert result == target_string

    # ordering of characters is abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ
    # the splitting point for the bins is at A (26.0)
    # we expect all niches to converge around this point

    genomes = [str(g) for g in elites.genomes.latest.flatten().tolist()]
    assert len(genomes) == 8
    for g in genomes:
        assert g == target_string
    # print(genomes)


@pytest.mark.slow
def test_string_matching2():
    target_string = "Evolve"
    env: BaseEnvironment = MatchString(
        StringEnvConfig(target=target_string, batch_size=8)
    )
    elites = MAPElites(env, MAPElitesConfig(map_grid_size=(3,), history_length=1))
    result = elites.search(init_steps=2_000, total_steps=20_000)
    elites.plot_fitness()
    assert result == target_string


@pytest.mark.skip(reason="unimplemented")
def test_function_optim():
    env: BaseEnvironment = FunctionOptim(ndim=2)
    elites = MAPElites(env, MAPElitesConfig(map_grid_size=(128,), history_length=10))
    assert elites.search(init_steps=5_000, total_steps=50_000, atol=0) == 1.0

    # elites.plot()


@pytest.mark.slow
def test_image_optim():
    env = ImageOptim(
        ImageEnvConfig(debug=False),
        PromptModel(PromptModelConfig(model_path="Salesforce/codegen-2B-mono", gpus=2)),
    )
    elites = MAPElites(
        env, MAPElitesConfig(map_grid_size=(2,), history_length=10, save_history=True)
    )
    result = elites.search(init_steps=10, total_steps=50)

    elites.plot_fitness()
    elites.visualize_individuals()
    print(f"Best image\n{result}")


@pytest.mark.slow
def test_load_snapshot():
    test_dir = Path(__file__).parent / "test_data" / "test_snapshot"
    # clean up test generated directory before running test
    if test_dir.exists() and test_dir.is_dir():
        shutil.rmtree(test_dir)

    target_string = "GaV"

    # run map-elites for 30 steps, while saving intermediate checkpoint
    env: BaseEnvironment = MatchString(
        StringEnvConfig(target=target_string, batch_size=8)
    )
    elites_1 = MAPElites(
        env,
        MAPElitesConfig(
            map_grid_size=(3,),
            history_length=1,
            save_snapshot_interval=15,
            output_dir=test_dir,
            seed=42,
            save_np_rng_state=True,
        ),
    )
    result_1 = elites_1.search(init_steps=20, total_steps=30)

    # resume map-elites run from snapshot at step 15, then search up to step 30 as before
    snapshot_to_load = test_dir / "step_15"
    elites_2 = MAPElites(
        env,
        MAPElitesConfig(
            map_grid_size=(3,),
            history_length=1,
            save_snapshot_interval=15,
            log_snapshot_dir=snapshot_to_load,
            output_dir=test_dir,
            load_np_rng_state=True,
        ),
    )
    result_2 = elites_2.search(init_steps=20, total_steps=30)

    assert (
        result_1 == result_2
    ), f"deterministic runs lead to different results from search ({result_1}), ({result_2})"
    assert (
        elites_1.fitnesses.array == elites_2.fitnesses.array
    ).all(), "fitness map from resumed run not same as run from scratch"


@pytest.mark.slow
def test_load_snapshot_with_history():
    test_dir = Path(__file__).parent / "test_data" / "test_snapshot"
    # clean up test generated directory before running test
    if test_dir.exists() and test_dir.is_dir():
        shutil.rmtree(test_dir)

    target_string = "GaV"

    # run map-elites for 30 steps, while saving intermediate checkpoint
    env: BaseEnvironment = MatchString(
        StringEnvConfig(target=target_string, batch_size=8)
    )
    elites_1 = MAPElites(
        env,
        MAPElitesConfig(
            map_grid_size=(3,),
            history_length=5,
            save_snapshot_interval=15,
            output_dir=test_dir,
            seed=42,
            save_np_rng_state=True,
        ),
    )
    result_1 = elites_1.search(init_steps=20, total_steps=30)

    # resume map-elites run from snapshot at step 15, then search up to step 30 as before
    snapshot_to_load = test_dir / "step_15"
    elites_2 = MAPElites(
        env,
        MAPElitesConfig(
            map_grid_size=(3,),
            history_length=5,
            save_snapshot_interval=15,
            log_snapshot_dir=snapshot_to_load,
            output_dir=test_dir,
            load_np_rng_state=True,
        ),
    )
    result_2 = elites_2.search(init_steps=20, total_steps=30)

    assert (
        result_1 == result_2
    ), f"deterministic runs lead to different results from search ({result_1}), ({result_2})"
    assert (
        elites_1.fitnesses.array == elites_2.fitnesses.array
    ).all(), "fitness map from resumed run not same as run from scratch"
