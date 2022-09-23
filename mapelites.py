import numpy as np
from tqdm import trange
from typing import Union

Genotype = Union[str, np.ndarray]
Phenotype = np.ndarray
Mapindex = tuple

class MAPElites:
    def __init__(self, env, nbins: int):
        self.env = env
        self.nbins = nbins

        # discretization of behaviour space
        self.bins = np.linspace(*env.behaviour_space, nbins+1)[1:-1].T
        # perfomance of niches
        self.fitnesses = np.full(np.repeat(nbins, env.behaviour_ndim), -np.inf)
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
            if n_steps < initsteps:
                x = self.env.random()
            else:
                map_ix = self.random_selection()
                x = self.genomes[map_ix]
                x = self.env.mutate(x)

            map_ix = self.to_mapindex(self.env.to_behaviour_space(x))
            self.nonzero[map_ix] = True

            f = self.env.fitness(x)

            if f > self.fitnesses[map_ix]:
                self.fitnesses[map_ix] = f
                self.genomes[map_ix] = x

            if f > max_fitness:
                max_fitness = f
                max_genome = x

                tbar.set_description(f'{max_fitness=:.4f} of "{self.env.to_string(max_genome)}"')

            if np.isclose(max_fitness, self.env.max_fitness, atol=atol):
                break

        return self.env.to_string(self.genomes[np.unravel_index(self.fitnesses.argmax(), self.fitnesses.shape)])

    def plot(self):
        from matplotlib import pyplot
        import matplotlib

        matplotlib.rcParams['font.family'] = 'Futura'
        matplotlib.rcParams['figure.dpi'] = 100
        matplotlib.style.use('ggplot')

        ix = tuple(np.zeros(elites.fitnesses.ndim-2, int))
        print(ix)
        map2d = elites.fitnesses[ix]
        print(f'{map2d.shape=}')

        pyplot.pcolor(map2d, cmap='inferno')
        pyplot.show()

if __name__ == '__main__':
    from examples import MatchString, FunctionOptim

    env = MatchString(target='MAPElites')
    elites = MAPElites(env, nbins=3)
    print('Best string match:', elites.search(initsteps=1e4, totalsteps=1e6))
    elites.plot()


    env = FunctionOptim(ndim=2)
    elites = MAPElites(env, nbins=128)
    print("Function's maximum:", elites.search(initsteps=1e4, totalsteps=1e6, atol=0))
    elites.plot()
