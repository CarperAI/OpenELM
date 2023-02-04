import multiprocessing as mp
from time import time

from openelm.environments.sodaracer.simulator import IESoRWorld, SodaraceSimulator
from openelm.environments.sodaracer.walker import Walker
from openelm.environments.sodaracer.walker.CPPN_fixed import (
    make_walker as make_walker_cppn_fixed,
)
from openelm.environments.sodaracer.walker.CPPN_mutable import (
    make_walker as make_walker_cppn_mutable,
)
from openelm.environments.sodaracer.walker.radial import (
    make_walker as make_walker_radial,
)
from openelm.environments.sodaracer.walker.square import (
    make_walker as make_walker_square,
)


def run_simulator(timesteps):
    square_walker = make_walker_square()
    simulator = SodaraceSimulator(body=square_walker.to_dict())
    res = simulator.evaluate(timesteps)
    # print(f"Square took {time() - start} seconds for {timesteps} timesteps")
    return res


def benchmark_sodarace():
    square_walker: Walker = make_walker_square()

    start = time()
    times = [350] * 10_000
    ress = []
    for timesteps in times:
        simulator = SodaraceSimulator(body=square_walker.to_dict())
        ress.append(simulator.evaluate(timesteps))
    print("Total time was", time() - start)
    print("Sequential simulations over.")
    processes = [1, 2, 4, 8, 16, 32]
    for p in processes:
        start = time()
        with mp.Pool(processes=p) as pool:
            _ = pool.map(run_simulator, times)
        print(f"Total time was {time() - start} with {p} processes.")


if __name__ == "__main__":
    benchmark_sodarace()
