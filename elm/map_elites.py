from collections import defaultdict
from typing import Optional

import numpy as np
from tqdm import trange

from elm.environments.environments import BaseEnvironment

Phenotype = Optional[np.ndarray]
Mapindex = Optional[tuple]


class Map:
    def __init__(
        self,
        dims: tuple,
        fill_value: float,
        dtype: type = np.float32,
        history_length: int = 1,
    ):
        self.history_length: int = history_length
        self.dims: tuple = dims
        if self.history_length == 1:
            self.array: np.ndarray = np.full(dims, fill_value, dtype=dtype)
        else:
            # Set starting top of buffer to 0 (% operator)
            self.top = np.full(dims, self.history_length - 1, dtype=int)
            self.array = np.full((history_length,) + dims, fill_value, dtype=dtype)

    def __getitem__(self, map_ix):
        """If history length > 1, the history dim is an n-dim circular buffer."""
        if self.history_length == 1:
            return self.array[map_ix]
        else:
            return self.array[(self.top[map_ix], *map_ix)]

    def __setitem__(self, map_ix, value):
        if self.history_length == 1:
            self.array[map_ix] = value
        else:
            top_val = self.top[map_ix]
            top_val = (top_val + 1) % self.history_length
            self.top[map_ix] = top_val
            self.array[(self.top[map_ix], *map_ix)] = value

    @property
    def shape(self) -> tuple:
        return self.array.shape

    @property
    def map_size(self) -> int:
        if self.history_length == 1:
            return self.array.size
        else:
            return self.array[0].size


class MAPElites:
    def __init__(
        self, env, n_bins: int, history_length: int, save_history: bool = False
    ):
        self.env: BaseEnvironment = env
        self.n_bins = n_bins
        self.history_length = history_length
        self.save_history = save_history

        # discretization of space
        self.bins = np.linspace(*env.behavior_space, n_bins + 1)[1:-1].T
        # perfomance of niches
        self.fitnesses: Map = Map(
            dims=(n_bins,) * env.behavior_ndim,
            fill_value=-np.inf,
            dtype=float,
            history_length=history_length,
        )
        # niches' sources
        self.genomes: Map = Map(
            dims=self.fitnesses.dims,
            fill_value=0.0,
            dtype=object,
            history_length=history_length,
        )
        # index over explored niches to select from
        self.nonzero: Map = Map(dims=self.fitnesses.dims, fill_value=False, dtype=bool)
        # bad mutations that ended up with invalid output.
        self.recycled = [None] * 1000
        self.recycled_count = 0

        print(f"MAP of size: {self.fitnesses.dims} = {self.fitnesses.map_size}")

    def to_mapindex(self, b: Phenotype) -> Mapindex:
        return (
            None
            if b is None
            else tuple(np.digitize(x, bins) for x, bins in zip(b, self.bins))
        )

    def random_selection(self) -> Mapindex:
        ix = np.random.choice(np.flatnonzero(self.nonzero.array))
        return np.unravel_index(ix, self.nonzero.dims)

    def search(self, initsteps: int, totalsteps: int, atol=1, batch_size=32):
        tbar = trange(int(totalsteps))
        max_fitness = -np.inf
        max_genome = None
        if self.save_history:
            self.history: dict = defaultdict(list)

        config = {"batch_size": batch_size}

        for n_steps in tbar:
            if n_steps < initsteps:
                # Initialise by generating initsteps random solutions.
                x = self.env.random(**config)
            else:
                # Randomly select an elite from the map
                map_ix = self.random_selection()
                x = self.genomes[map_ix]
                # Mutate the elite
                x = self.env.mutate(x, **config)

            # Now that `x` is a list, we put them into the behavior space one-by-one.
            for individual in x:
                map_ix = self.to_mapindex(self.env.to_behavior_space(individual))
                # if the return is None, the individual is invalid and is thrown into the recycle bin.
                if map_ix is None:
                    self.recycled[self.recycled_count % len(self.recycled)] = individual
                    self.recycled_count += 1
                    continue

                if self.save_history:
                    self.history[map_ix].append(individual)
                self.nonzero[map_ix] = True

                f = self.env.fitness(individual)
                # If new fitness greater than old fitness in niche, replace.
                if f > self.fitnesses[map_ix]:
                    self.fitnesses[map_ix] = f
                    self.genomes[map_ix] = individual
                # If new fitness is the highest so far, update the tracker.
                if f > max_fitness:
                    max_fitness = f
                    max_genome = individual

                    tbar.set_description(f"{max_fitness=:.4f}")
                # If best fitness is within atol of the maximum possible fitness, stop.
                if np.isclose(max_fitness, self.env.max_fitness, atol=atol):
                    break

        return str(max_genome)

    def plot(self):
        import matplotlib
        from matplotlib import pyplot

        matplotlib.rcParams["font.family"] = "Futura"
        matplotlib.rcParams["figure.dpi"] = 100
        matplotlib.style.use("ggplot")

        ix = tuple(np.zeros(self.fitnesses.array.ndim - 2, int))
        print(ix)
        map2d = self.fitnesses[ix]
        print(f"{map2d.shape=}")

        pyplot.pcolor(map2d, cmap="inferno")
        pyplot.show()


if __name__ == "__main__":
    from elm.environments.environments import (
        BaseEnvironment,
        FunctionOptim,
        ImageOptim,
        MatchString,
    )

    env: BaseEnvironment = MatchString(target="MAPElites")
    elites = MAPElites(env, n_bins=3, history_length=10)
    print("Best string match:", elites.search(initsteps=10_000, totalsteps=100_000))
    elites.plot()

    env = FunctionOptim(ndim=2)
    elites = MAPElites(env, n_bins=128, history_length=10)
    print(
        "Function's maximum:", elites.search(initsteps=5_000, totalsteps=50_000, atol=0)
    )
    elites.plot()

    seed = """def draw_blue_rectangle() -> np.ndarray:
\tpic = np.zeros((32, 32, 3))
\tfor x in range(2, 30):
\t\tfor y in range(2, 30):
\t\t\tpic[x, y] = np.array([0, 0, 255])
\treturn pic
"""
    target = np.zeros((32, 32, 3))
    for y in range(32):
        for x in range(32):
            if (y - 16) ** 2 + (x - 16) ** 2 <= 100:  # a radius-10 circle
                target[y, x] = np.array([1, 1, 0])
    env = ImageOptim(seed, "elm_image_cfg.yaml", target_img=target, func_name="draw")
    elites = MAPElites(env, n_bins=2, history_length=10)
    print("Best image", elites.search(initsteps=5, totalsteps=10))
