from openelm.environments.environments import (
    BaseEnvironment,
    FunctionOptim,
    ImageOptim,
    MatchString,
)
from openelm.map_elites import MAPElites
from openelm.environments import image_init_args


def test_string_matching():
    env: BaseEnvironment = MatchString(target="MAPElites")
    elites = MAPElites(env, n_bins=3, history_length=10)
    assert elites.search(initsteps=10_000, totalsteps=100_000) == "MAPElites"

    elites.plot()


def test_function_optim():
    env: BaseEnvironment = FunctionOptim(ndim=2)
    elites = MAPElites(env, n_bins=128, history_length=10)
    assert elites.search(initsteps=5_000, totalsteps=50_000, atol=0) == 1.0

    elites.plot()


def test_image_optim():
    env = ImageOptim(**image_init_args)
    elites = MAPElites(env, n_bins=2, history_length=10)

    print("Best image", elites.search(initsteps=5, totalsteps=10))
