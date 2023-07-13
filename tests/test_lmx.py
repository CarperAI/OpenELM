import numpy as np
import pytest
from pathlib import Path
import shutil

from openelm.configs import (
    APIModelConfig,
    LMXMapElitesConfig,
    LMXGenerationEnvConfig,
)
from openelm.environments.environments import (
    LMXGenerationEnvironment,
)
from openelm.map_elites import LMXMapElites
from openelm.mutation_model import AlephAlphaLLM


def test_lmx_environment():
    test_dir = Path(__file__).parent / "test_data" / "test_snapshot"
    # clean up test generated directory before running test
    if test_dir.exists() and test_dir.is_dir():
        shutil.rmtree(test_dir)

    env = LMXGenerationEnvironment(
        LMXGenerationEnvConfig(),
        AlephAlphaLLM(APIModelConfig(model_used="luminous-base", output_dir=test_dir)),
    )
    elites_1 = LMXMapElites(
        env,
        LMXMapElitesConfig(
            map_grid_size=(5,),
            history_length=100,
            save_history=True,
            custom_ticks=None,
            output_dir=test_dir,
        ),
    )
    elites_2 = LMXMapElites(
        env,
        LMXMapElitesConfig(
            map_grid_size=(5,),
            history_length=100,
            save_history=True,
            custom_ticks=[0.005, 0.01, 0.015, 0.02],
            output_dir=test_dir,
        ),
    )
    assert (
        elites_1.bins.shape == elites_2.bins.shape
    ), f"expected number (and dim) of bins between uniform and custom bins to be equal, got ({elites_1.bins.shape}), ({elites_2.bins.shape})"

    phenotype_0 = np.repeat([0.007], env.genotype_ndim)
    placed_idx_1 = elites_1.to_mapindex(phenotype_0)
    placed_idx_2 = elites_2.to_mapindex(phenotype_0)
    assert (
        placed_idx_1[0] == 0
    ), f"expected placed index in uniform bin to be 0, got ({placed_idx_1[0]})"
    assert (
        placed_idx_2[0] == 1
    ), f"expected placed index in uniform bin to be 1, got ({placed_idx_1[0]})"

    result_1 = elites_1.search(init_steps=1, total_steps=10)
    result_2 = elites_2.search(init_steps=1, total_steps=10)


@pytest.mark.parametrize("use_alt_depth_method", [True, False])
@pytest.mark.parametrize("add_only_improved_completions_to_prompt_pool", [True, False])
@pytest.mark.parametrize("solution_init_method", ["seed", "generated"])
@pytest.mark.parametrize("batch_size", [1, 2])
def test_lmx_replace(
    use_alt_depth_method,
    add_only_improved_completions_to_prompt_pool,
    solution_init_method,
    batch_size,
):
    mutation_method = "replace"

    test_dir = Path(__file__).parent / "test_data" / "test_snapshot"
    # clean up test generated directory before running test
    if test_dir.exists() and test_dir.is_dir():
        shutil.rmtree(test_dir)

    env = LMXGenerationEnvironment(
        LMXGenerationEnvConfig(
            add_only_improved_completions_to_prompt_pool=add_only_improved_completions_to_prompt_pool,
            mutation_method=mutation_method,
            solution_init_method=solution_init_method,
            batch_size=batch_size,
        ),
        AlephAlphaLLM(APIModelConfig(model_used="luminous-base", output_dir=test_dir)),
    )
    elites = LMXMapElites(
        env,
        LMXMapElitesConfig(
            map_grid_size=(5,),
            history_length=100,
            save_history=True,
            custom_ticks=None,
            output_dir=test_dir,
            use_alt_depth_method=use_alt_depth_method,
        ),
    )
    result = elites.search(init_steps=1, total_steps=10)


@pytest.mark.parametrize("use_alt_depth_method", [True, False])
@pytest.mark.parametrize("solution_init_method", ["seed", "generated"])
@pytest.mark.parametrize("batch_size", [1, 2])
def test_lmx_near(use_alt_depth_method, solution_init_method, batch_size):
    mutation_method = "lmx_near"
    test_dir = Path(__file__).parent / "test_data" / "test_snapshot"
    # clean up test generated directory before running test
    if test_dir.exists() and test_dir.is_dir():
        shutil.rmtree(test_dir)

    env = LMXGenerationEnvironment(
        LMXGenerationEnvConfig(
            mutation_method=mutation_method,
            solution_init_method=solution_init_method,
            batch_size=batch_size,
        ),
        AlephAlphaLLM(APIModelConfig(model_used="luminous-base", output_dir=test_dir)),
    )
    elites = LMXMapElites(
        env,
        LMXMapElitesConfig(
            map_grid_size=(5,),
            history_length=100,
            save_history=True,
            custom_ticks=None,
            output_dir=test_dir,
            use_alt_depth_method=use_alt_depth_method,
        ),
    )
    result = elites.search(init_steps=1, total_steps=10)


def test_prompt_pool_initialization():
    num_fewshot = 5
    init_size = 2
    with pytest.raises(AssertionError):
        env = LMXGenerationEnvironment(
            LMXGenerationEnvConfig(
                solution_init_method="generated",
                init_size_prompt_pool=init_size,
                mutation_method="replace",
            ),
            AlephAlphaLLM(APIModelConfig(model_used="luminous-base")),
            num_fewshot=num_fewshot,
        )

    init_size = 10
    env = LMXGenerationEnvironment(
        LMXGenerationEnvConfig(
            solution_init_method="generated",
            init_size_prompt_pool=init_size,
            mutation_method="replace",
        ),
        AlephAlphaLLM(APIModelConfig(model_used="luminous-base")),
        num_fewshot=num_fewshot,
    )
    assert (
        len(env.prompt_pool) == init_size
    ), "expected to create prompt pool of size {init_size}"
    env = LMXGenerationEnvironment(
        LMXGenerationEnvConfig(
            solution_init_method="seed",
            prompt_pool_path="tests/test_data/test_seed_pool.txt",
            mutation_method="replace",
        ),
        AlephAlphaLLM(APIModelConfig(model_used="luminous-base")),
    )
    assert (
        env.prompt_pool[0] == "test 1"
    ), "prompt pool should be 'test 1' for init method seed"


def test_invalid_config_args():
    # Test failure cases for values of LMXGenerationEnvConfig
    with pytest.raises(AssertionError):
        env = LMXGenerationEnvironment(
            LMXGenerationEnvConfig(classifier_model="luminous-base"),
            AlephAlphaLLM(APIModelConfig(model_used="luminous-base")),
        )
    with pytest.raises(AssertionError):
        env = LMXGenerationEnvironment(
            LMXGenerationEnvConfig(solution_init_method="random"),
            AlephAlphaLLM(APIModelConfig(model_used="luminous-base")),
        )
    with pytest.raises(AssertionError):
        env = LMXGenerationEnvironment(
            LMXGenerationEnvConfig(mutation_method="random"),
            AlephAlphaLLM(APIModelConfig(model_used="luminous-base")),
        )
    # Test failure cases for values of APIModelConfig
    with pytest.raises(AssertionError):
        env = LMXGenerationEnvironment(
            LMXGenerationEnvConfig(),
            AlephAlphaLLM(APIModelConfig(model_used="luminous-tiny")),
        )
