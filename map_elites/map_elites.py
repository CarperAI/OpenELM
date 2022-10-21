from typing import Union

import numpy as np
from tqdm import trange

Genotype = Union[str, np.ndarray, dict, tuple]
Phenotype = Union[np.ndarray, None]
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

        # bad mutations that ended up with invalid output.
        self.recycled = [None] * 1000
        self.recycled_count = 0

        print(f"MAP of size: {self.fitnesses.shape} = {self.fitnesses.size}")

    def to_mapindex(self, b: Phenotype) -> Mapindex:
        return None if b is None else tuple(np.digitize(x, bins) for x, bins in zip(b, self.bins))

    def random_selection(self) -> Mapindex:
        ix = np.random.choice(np.flatnonzero(self.nonzero))
        return np.unravel_index(ix, self.nonzero.shape)

    def search(self, initsteps: int, totalsteps: int, atol=1, batch_size=32):
        tbar = trange(int(totalsteps))
        max_fitness = -np.inf
        max_genome = None

        use_batch_op = self.env.use_batch_op
        config = {'batch_size': batch_size} if use_batch_op else {}

        for n_steps in tbar:
            if self.task == "Sodarace":
                # Batch generate initsteps programs
                pass
            if n_steps < initsteps:
                # Initialise by generating initsteps random solutions.
                x = self.env.random(**config)
            else:
                # Randomly select an elite from the map
                map_ix = self.random_selection()
                x = self.genomes[map_ix]
                # Mutate the elite
                x = self.env.mutate(x, **config)

            if not use_batch_op:
                x = [x]

            # Now that `x` is a list, we put them into the behaviour space one-by-one.
            for individual in x:
                map_ix = self.to_mapindex(self.env.to_behaviour_space(individual))
                # if the return is None, the individual is invalid and is thrown into the recycle bin.
                if map_ix is None:
                    self.recycled[self.recycled_count % len(self.recycled)] = individual
                    self.recycled_count += 1
                    continue

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
    from map_elites.environments import FunctionOptim, MatchString, ImageOptim

    env = MatchString(target='MAPElites')
    elites = MAPElites(env, n_bins=3)
    print('Best string match:', elites.search(initsteps=10_000, totalsteps=50_000))
    elites.plot()

    env = FunctionOptim(ndim=2)
    elites = MAPElites(env, n_bins=128)
    print("Function's maximum:", elites.search(initsteps=10_000, totalsteps=1_000_000, atol=0))
    elites.plot()

    seed = """def draw_blue_rectangle() -> np.ndarray:
    pic = np.zeros((32, 32, 3))
    for x in range(2, 30):
        for y in range(2, 30):
            pic[x, y] = np.array([0, 0, 255])
    return pic
    """
    target = np.zeros((32, 32, 3))
    for y in range(32):
        for x in range(32):
            if (y - 16)**2 + (x - 16)**2 <= 100:  # a radius-10 circle
                target[y, x] = np.array([1, 1, 0])
    env = ImageOptim(seed, 'elm_cfg.yaml', target_img=target, func_name='draw')
    elites = MAPElites(env, n_bins=2)
    print("Best image", elites.search(initsteps=5, totalsteps=10))
