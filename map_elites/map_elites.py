from typing import Union

import numpy as np
from tqdm import trange

Genotype = Union[str, np.ndarray, dict, tuple]
Phenotype = np.ndarray
Mapindex = tuple


class MAPElites:
    def __init__(self, env, n_bins: int, task: str = None):
        self.task = task
        self.env = env
        self.n_bins = n_bins

        # discretization of behaviour space
        self.bins = np.linspace(*env.behaviour_space, n_bins + 1)[1:-1].T
        # perfomance of niches
        self.fitnesses = np.full(np.repeat(n_bins, env.behaviour_ndim), -np.inf)
        # niches' sources
        self.genomes = np.zeros(self.fitnesses.shape, dtype=object)
        # index over explored niches to select from
        self.nonzero = np.full(self.fitnesses.shape, False)

        print(f"MAP of size: {self.fitnesses.shape} = {self.fitnesses.size}")

    def to_mapindex(self, b: Phenotype) -> Mapindex:
        return tuple(np.digitize(x, bins) for x, bins in zip(b, self.bins))

    def random_selection(self) -> Mapindex:
        ix = np.random.choice(np.flatnonzero(self.nonzero))
        return np.unravel_index(ix, self.nonzero.shape)

    def search(self, initsteps: int, totalsteps: int, atol=1):
        tbar = trange(int(totalsteps))
        max_fitness = -np.inf
        max_genome = None

        for n_steps in tbar:
            if self.task == "Sodarace":
                # Batch generate initsteps programs
                pass
            if n_steps < initsteps:
                # Initialise by generating initsteps random solutions.
                x = self.env.random()
            else:
                # Randomly select an elite from the map
                map_ix = self.random_selection()
                x = self.genomes[map_ix]
                # Mutate the elite
                x = self.env.mutate(x)

            map_ix = self.to_mapindex(self.env.to_behaviour_space(x))
            self.nonzero[map_ix] = True

            f = self.env.fitness(x)
            # If new fitness greater than old fitness in niche, replace.
            if f > self.fitnesses[map_ix]:
                self.fitnesses[map_ix] = f
                self.genomes[map_ix] = x
            # If new fitness is the highest so far, update the tracker.
            if f > max_fitness:
                max_fitness = f
                max_genome = x

                tbar.set_description(f'{max_fitness=:.4f} of "{self.env.to_string(max_genome)}"')
            # If best fitness is within atol of the maximum possible fitness, stop.
            if np.isclose(max_fitness, self.env.max_fitness, atol=atol):
                break

        return self.env.to_string(self.genomes[np.unravel_index(self.fitnesses.argmax(), self.fitnesses.shape)])

    def plot(self):
        import matplotlib
        from matplotlib import pyplot

        matplotlib.rcParams['font.family'] = 'Futura'
        matplotlib.rcParams['figure.dpi'] = 100
        matplotlib.style.use('ggplot')

        ix = tuple(np.zeros(self.fitnesses.ndim - 2, int))
        print(ix)
        map2d = self.fitnesses[ix]
        print(f'{map2d.shape=}')

        pyplot.pcolor(map2d, cmap='inferno')
        pyplot.show()


if __name__ == '__main__':
    from map_elites.environments import FunctionOptim, MatchString

    env = MatchString(target='MAPElites')
    elites = MAPElites(env, n_bins=3)
    print('Best string match:', elites.search(initsteps=10_000, totalsteps=50_000))
    elites.plot()

    env = FunctionOptim(ndim=2)
    elites = MAPElites(env, n_bins=128)
    print("Function's maximum:", elites.search(initsteps=10_000, totalsteps=1_000_000, atol=0))
    elites.plot()
