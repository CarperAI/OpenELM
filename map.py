import string
import numpy as np
from numpy import array
from tqdm import trange

Genotype = np.ndarray
Phenotype = np.ndarray
Mapindex = tuple

# find a string by mutating one character at a time
class MatchString:
    def __init__(self, target: str):
        self.alphabet = string.ascii_letters

        self.target = array([self.alphabet.index(ch) for ch in target])
        self.nparams = self.target.shape[0]
        self.genotype_search_space = np.repeat([[0, len(self.alphabet)]], self.nparams, axis=0).T

    def random(self) -> Genotype:
        return np.random.uniform(*self.genotype_search_space)

    def mutate(self, x: Genotype) -> Genotype:
        x = x.copy()
        ix = np.random.randint(self.nparams)
        x[ix] = x[ix] + np.random.uniform(-5, 5)
        return x

    def fitness(self, x: Genotype) -> float:
        return -np.abs(x - self.target).sum()

    def to_feature_space(self, x: Genotype) -> Phenotype:
        return x

    def to_string(self, x: Genotype) -> str:
        return ''.join(self.alphabet[ix] for ix in np.clip(np.round(x).astype(int), 0, len(self.alphabet)-1))

    @property
    def max_fitness(self):
        return 0

    @property
    # [starts, endings) of search intervals
    def phenotype_search_space(self):
        return self.genotype_search_space

class MAPElites:
    def __init__(self, env, nbins: int):
        self.env = env
        self.nbins = nbins

        self.binmarkers = np.linspace(*env.phenotype_search_space, nbins+1)[1:-1].T

        self.fitnesses = np.full(np.repeat(nbins, env.nparams), -np.inf)
        self.genomes = np.zeros(self.fitnesses.shape, dtype=object)
        self.nonzero = np.full(self.fitnesses.shape, False)

        print(f"MAP of size: {self.fitnesses.shape} = {self.fitnesses.size}")

    def to_mapindex(self, b: Phenotype) -> Mapindex:
        return tuple(np.digitize(v, bins) for v, bins in zip(b, self.binmarkers))

    def random_selection(self) -> Mapindex:
        ix = np.random.choice(np.flatnonzero(self.nonzero))
        return np.unravel_index(ix, self.nonzero.shape)

    def search(self, initsteps: int, totalsteps: int):
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

            map_ix = self.to_mapindex(self.env.to_feature_space(x))
            self.nonzero[map_ix] = True

            f = self.env.fitness(x)

            if f > self.fitnesses[map_ix]:
                self.fitnesses[map_ix] = f
                self.genomes[map_ix] = x

            if f > max_fitness:
                max_fitness = f
                max_genome = x

                tbar.set_description(f'{max_fitness=:.2f} of "{self.env.to_string(max_genome)}"')

            if np.isclose(max_fitness, self.env.max_fitness, atol=1):
                break

        return self.env.to_string(self.genomes[np.unravel_index(self.fitnesses.argmax(), self.fitnesses.shape)])

if __name__ == '__main__':
    env = MatchString(target='MAPElites')
    elites = MAPElites(env, nbins=2)
    print(elites.search(initsteps=100, totalsteps=100000))
