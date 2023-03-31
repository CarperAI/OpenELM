from collections import defaultdict
from typing import Optional

import numpy as np
from sklearn.cluster import KMeans
from tqdm import trange

from openelm.configs import CVTMAPElitesConfig, MAPElitesConfig, QDConfig
from openelm.environments import BaseEnvironment, Genotype

Phenotype = Optional[np.ndarray]
MapIndex = Optional[tuple]


class Map:
    """
    Class to represent a map of any dimensionality, backed by a numpy array.

    This class is necessary to handle the circular buffer for the history dimension.
    """

    def __init__(
        self,
        dims: tuple,
        fill_value: float,
        dtype: type = np.float32,
        history_length: int = 1,
    ):
        """
        Class to represent a map of any dimensionality, backed by a numpy array.

        This class is a wrapper around a numpy array that handles the circular
        buffer for the history dimension. We use it for the array of fitnesses,
        genomes, and to track whether a niche in the map has been explored or not.

        Args:
            dims (tuple): Tuple of ints representing the dimensions of the map.
            fill_value (float): Fill value to initialize the array.
            dtype (type, optional): Type to pass to the numpy array to initialize
                it. For example, we initialize the map of genomes with type `object`.
                Defaults to np.float32.
            history_length (int, optional): Length of history to store for each
                niche (cell) in the map. This acts as a circular buffer, so after
                storing `history_length` items, the buffer starts overwriting the
                oldest items. Defaults to 1.
        """
        self.history_length: int = history_length
        self.dims: tuple = dims
        if self.history_length == 1:
            self.array: np.ndarray = np.full(dims, fill_value, dtype=dtype)
        else:
            # Set starting top of buffer to 0 (% operator)
            self.top = np.full(dims, self.history_length - 1, dtype=int)
            self.array = np.full(
                (history_length,) + tuple(dims), fill_value, dtype=dtype
            )
        self.empty = True

    def __getitem__(self, map_ix):
        """If history length > 1, the history dim is an n-dim circular buffer."""
        if self.history_length == 1:
            return self.array[map_ix]
        else:
            return self.array[(self.top[map_ix], *map_ix)]

    def __setitem__(self, map_ix, value):
        self.empty = False
        if self.history_length == 1:
            self.array[map_ix] = value
        else:
            top_val = self.top[map_ix]
            top_val = (top_val + 1) % self.history_length
            self.top[map_ix] = top_val
            self.array[(self.top[map_ix], *map_ix)] = value

    @property
    def latest(self) -> np.ndarray:
        """Returns the latest values in the history buffer."""
        if self.history_length == 1:
            return self.array
        else:
            # should be equivalent to np.choose(self.top, self.array), but without limit of 32 choices
            return np.take_along_axis(
                arr=self.array, indices=self.top[np.newaxis, ...], axis=0
            ).squeeze(axis=0)

    @property
    def shape(self) -> tuple:
        """Wrapper around the shape of the numpy array."""
        return self.array.shape

    @property
    def map_size(self) -> int:
        """Returns the product of the dimension sizes, not including history."""
        if self.history_length == 1:
            return self.array.size
        else:
            return self.array[0].size

    @property
    def qd_score(self) -> float:
        """Returns the quality-diversity score of the map."""
        return self.array[np.isfinite(self.array)].sum()

    @property
    def max(self) -> float:
        """Returns the maximum value in the map, not including history."""
        return self.latest.max()

    @property
    def min(self) -> float:
        """Returns the minimum value in the map, not including history."""
        return self.latest.min()

    @property
    def max_finite(self) -> float:
        """Returns the maximum finite value in the map, not including history."""
        if not np.isfinite(self.latest).any():
            return np.NaN
        else:
            return self.latest[np.isfinite(self.latest)].max()

    @property
    def min_finite(self) -> float:
        """Returns the minimum finite value in the map, not including history."""
        if not np.isfinite(self.latest).any():
            return np.NaN
        else:
            return self.latest[np.isfinite(self.latest)].min()

    @property
    def mean(self) -> np.ndarray:
        """Returns the mean finite value in the map."""
        return np.mean(self.latest[np.isfinite(self.latest)])

    @property
    def niches_filled(self) -> int:
        """Returns the number of niches in the map that have been explored."""
        return np.count_nonzero(np.isfinite(self.array))


class MAPElitesBase:
    """
    Class implementing MAP-Elites, a quality-diversity algorithm.

    MAP-Elites creates a map of high perfoming solutions at each point in a
    discretized behavior space. First, the algorithm generates some initial random
    solutions, and evaluates them in the environment. Then, it  repeatedly mutates
    the solutions in the map, and places the mutated solutions in the map if they
    outperform the solutions already in their niche.
    """

    def __init__(
        self,
        env,
        config: QDConfig,
        init_map: Optional[Map] = None,
    ):
        """
        The base class for MAP-Elites and variants, implementing common functions and search.

        Args:
            env (BaseEnvironment): The environment to evaluate solutions in. This
            should be a subclass of `BaseEnvironment`, and should implement
            methods to generate random solutions, mutate existing solutions,
            and evaluate solutions for their fitness in the environment.
            config (QDConfig): The configuration for the algorithm.
            init_map (Map, optional): A map to use for the algorithm. If not passed,
            a new map will be created. Defaults to None.
        """
        self.env: BaseEnvironment = env
        self.config: QDConfig = config
        self.history_length = self.config.history_length
        self.save_history = self.config.save_history
        # self.history will be set/reset each time when calling `.search(...)`
        self.history: dict = defaultdict(list)
        self.fitness_history: dict = defaultdict(list)

        # bad mutations that ended up with invalid output.
        self.recycled = [None] * 1000
        self.recycled_count = 0

        self._init_discretization()
        self._init_maps(init_map)
        print(f"MAP of size: {self.fitnesses.dims} = {self.fitnesses.map_size}")

    def _init_discretization(self):
        """Initializes the discretization of the behavior space."""
        raise NotImplementedError

    def _get_map_dimensions(self):
        """Returns the dimensions of the map."""
        raise NotImplementedError

    def to_mapindex(self, b: Phenotype) -> MapIndex:
        """Converts a phenotype (position in behaviour space) to a map index."""
        raise NotImplementedError

    def visualize(self):
        """Visualizes the map."""
        pass

    def _init_maps(self, init_map: Optional[Map] = None):
        # TODO: abstract all maps out to a single class.
        # perfomance of niches
        if init_map is None:
            self.map_dims = self._get_map_dimensions()
            self.fitnesses: Map = Map(
                dims=self.map_dims,
                fill_value=-np.inf,
                dtype=float,
                history_length=self.history_length,
            )
        else:
            self.map_dims = init_map.dims
            self.fitnesses = init_map

        # niches' sources
        self.genomes: Map = Map(
            dims=self.map_dims,
            fill_value=0.0,
            dtype=object,
            history_length=self.history_length,
        )
        # index over explored niches to select from
        self.nonzero: Map = Map(dims=self.map_dims, fill_value=False, dtype=bool)

    def random_selection(self) -> MapIndex:
        """Randomly select a niche (cell) in the map that has been explored."""
        ix = np.random.choice(np.flatnonzero(self.nonzero.array))
        return np.unravel_index(ix, self.nonzero.dims)

    def search(self, init_steps: int, total_steps: int, atol: float = 1.0) -> str:
        """
        Run the MAP-Elites search algorithm.

        Args:
            initsteps (int): Number of initial random solutions to generate.
            totalsteps (int): Total number of steps to run the algorithm for,
                including initial steps.
            atol (float, optional): Tolerance for how close the best performing
                solution has to be to the maximum possible fitness before the
                search stops early. Defaults to 1.

        Returns:
            str: A string representation of the best perfoming solution. The
                best performing solution object can be accessed via the
                `current_max_genome` class attribute.
        """
        tbar = trange(int(total_steps))
        max_fitness = -np.inf
        max_genome = None
        if self.save_history:
            self.history = defaultdict(list)

        for n_steps in tbar:
            if n_steps < init_steps or self.genomes.empty:
                # Initialise by generating initsteps random solutions.
                # If map is still empty: force to do generation instead of mutation.
                # TODO: use a separate sampler, move batch size to qd config.
                new_individuals: list[Genotype] = self.env.random()
            else:
                # Randomly select a batch of elites from the map.
                batch: list[Genotype] = []
                for _ in range(self.env.batch_size):
                    map_ix = self.random_selection()
                    batch.append(self.genomes[map_ix])
                # Mutate the elite.
                new_individuals = self.env.mutate(batch)

            # `new_individuals` is a list of generation/mutation. We put them
            # into the behavior space one-by-one.
            # TODO: account for the case where multiple new individuals are
            # placed in the same niche, for saving histories.
            for individual in new_individuals:
                fitness = self.env.fitness(individual)
                if np.isinf(fitness):
                    continue
                map_ix = self.to_mapindex(individual.to_phenotype())
                # if the return is None, the individual is invalid and is thrown
                # into the recycle bin.
                if map_ix is None:
                    self.recycled[self.recycled_count % len(self.recycled)] = individual
                    self.recycled_count += 1
                    continue

                if self.save_history:
                    # TODO: thresholding
                    self.history[map_ix].append(individual)
                self.nonzero[map_ix] = True

                # If new fitness greater than old fitness in niche, replace.
                if fitness > self.fitnesses[map_ix]:
                    self.fitnesses[map_ix] = fitness
                    self.genomes[map_ix] = individual
                # If new fitness is the highest so far, update the tracker.
                if fitness > max_fitness:
                    max_fitness = fitness
                    max_genome = individual

                    tbar.set_description(f"{max_fitness=:.4f}")
                # Stop if best fitness is within atol of maximum possible fitness.
                if np.isclose(max_fitness, self.env.max_fitness, atol=atol):
                    break

            self.fitness_history["max"].append(max_fitness)
            self.fitness_history["min"].append(self.fitnesses.min)
            self.fitness_history["mean"].append(self.fitnesses.mean)

        self.current_max_genome = max_genome
        self.visualize()
        return str(max_genome)

    def niches_filled(self):
        """Get the number of niches that have been explored in the map."""
        return self.fitnesses.niches_filled

    def max_fitness(self):
        """Get the maximum fitness value in the map."""
        return self.fitnesses.max_finite

    def mean_fitness(self):
        """Get the mean fitness value in the map."""
        return self.fitnesses.mean

    def min_fitness(self):
        """Get the minimum fitness value in the map."""
        return self.fitnesses.min_finite

    def qd_score(self):
        """
        Get the quality-diversity score of the map.

        The quality-diversity score is the sum of the performance of all solutions
        in the map.
        """
        return self.fitnesses.qd_score

    def plot_fitness(self):
        import matplotlib.pyplot as plt

        plt.figure()
        plt.plot(self.fitness_history["max"], label="max fitness")
        plt.plot(self.fitness_history["mean"], label="mean fitness")
        plt.plot(self.fitness_history["min"], label="min fitness")
        plt.legend()
        plt.savefig("logs/elm/MAPElites_fitness_history.png")

        if len(self.map_dims) > 1:
            ix = tuple(np.zeros(max(1, len(self.fitnesses.dims) - 2), int))
            map2d = self.fitnesses.latest[ix]
            print(
                "plotted genes:",
                *[str(g) for g in self.genomes.latest[ix].flatten().tolist()],
            )

            plt.figure()
            plt.pcolor(map2d, cmap="inferno")
            plt.savefig("logs/elm/MAPElites_vis.png")

    def visualize_individuals(self):
        """Visualize the genes of the best performing solution."""
        import matplotlib.pyplot as plt

        tmp = self.genomes.array.reshape(self.genomes.shape[0], -1)

        # if we're tracking history, rows will be the history dimension
        # otherwise, just the first dimension of the map
        plt.figure()
        _, axs = plt.subplots(nrows=tmp.shape[0], ncols=tmp.shape[1])
        for genome, ax in zip(tmp.flatten(), axs.flatten()):
            # keep the border but remove the ticks
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])
            try:
                genome.visualize(ax=ax)
            except AttributeError:
                pass

        plt.savefig("logs/elm/MAPElites_individuals.png")


class MAPElites(MAPElitesBase):
    """
    Class implementing MAP-Elites, a quality-diversity algorithm.

    MAP-Elites creates a map of high perfoming solutions at each point in a
    discretized behavior space. First, the algorithm generates some initial random
    solutions, and evaluates them in the environment. Then, it  repeatedly mutates
    the solutions in the map, and places the mutated solutions in the map if they
    outperform the solutions already in their niche.
    """

    def __init__(
        self,
        env,
        config: MAPElitesConfig,
        *args,
        **kwargs,
    ):
        """
        Class implementing MAP-Elites, a quality-diversity algorithm.

        Args:
            env (BaseEnvironment): The environment to evaluate solutions in. This
            should be a subclass of `BaseEnvironment`, and should implement
            methods to generate random solutions, mutate existing solutions,
            and evaluate solutions for their fitness in the environment.
            config (MAPElitesConfig): The configuration for the algorithm.
        """
        self.map_grid_size = config.map_grid_size
        super().__init__(env=env, config=config, *args, **kwargs)

    def _init_discretization(self):
        """Set up the discrete behaviour space for the algorithm."""
        # TODO: make this work for any number of dimensions
        self.bins = np.linspace(*self.env.behavior_space, self.map_grid_size[0] + 1)[1:-1].T  # type: ignore

    def _get_map_dimensions(self):
        """Returns the dimensions of the map."""
        return self.map_grid_size * self.env.behavior_ndim

    def to_mapindex(self, b: Phenotype) -> MapIndex:
        """Converts a phenotype (position in behaviour space) to a map index."""
        return (
            None
            if b is None
            else tuple(np.digitize(x, bins) for x, bins in zip(b, self.bins))
        )

    def visualize(self):
        """Visualize the map."""
        self.plot_fitness()


class CVTMAPElites(MAPElitesBase):
    """
    Class implementing CVT-MAP-Elites, a variant of MAP-Elites.

    This replaces the grid of niches in MAP-Elites with niches generated using a Centroidal Voronoi Tessellation.
    Unlike in MAP-Elites, we have a fixed number of total niches rather than a fixed number of subdivisions per dimension.
    """

    def __init__(
        self,
        env,
        config: CVTMAPElitesConfig,
        *args,
        **kwargs,
    ):
        """
        Class implementing CVT-MAP-Elites, a variant of MAP-Elites.

        Args:
            env (BaseEnvironment): The environment to evaluate solutions in. This
            should be a subclass of `BaseEnvironment`, and should implement
            methods to generate random solutions, mutate existing solutions,
            and evaluate solutions for their fitness in the environment.
            config (CVTMAPElitesConfig): The configuration for the algorithm.
        """
        self.cvt_samples: int = config.cvt_samples
        self.n_niches: int = config.n_niches
        super().__init__(env=env, config=config, *args, **kwargs)

    def _init_discretization(self):
        """Discretize behaviour space using CVT."""
        # lower and upper bounds for each dimension
        low = self.env.behavior_space[0]
        high = self.env.behavior_space[1]

        points = np.zeros((self.cvt_samples, self.env.behavior_ndim))
        for i in range(self.env.behavior_ndim):
            points[:, i] = np.random.uniform(low[i], high[i], size=self.cvt_samples)

        k_means = KMeans(init="k-means++", n_init="auto", n_clusters=self.n_niches)
        k_means.fit(points)
        self.centroids = k_means.cluster_centers_

        self.plot_centroids(points, k_means)

    def _get_map_dimensions(self):
        """Returns the dimensions of the map."""
        return (self.n_niches,)

    def to_mapindex(self, b: Phenotype) -> MapIndex:
        """Maps a phenotype (position in behaviour space) to the index of the closest centroid."""
        return (
            None
            if b is None
            else (np.argmin(np.linalg.norm(b - self.centroids, axis=1)),)
        )

    def visualize(self):
        """Visualize the map."""
        self.plot_fitness()
        self.plot_behaviour_space()

    def plot_centroids(self, points, k_means):
        """
        Plot the CVT centroids and the points used to generate them.

        Args:
            points (np.ndarray, int): the points used to generate the centroids
            k_means (sklearn.cluster.KMeans): the k-means object used to generate the centroids
        """
        import matplotlib.pyplot as plt

        plt.figure()
        labels = k_means.labels_
        if self.env.behavior_ndim == 2:
            for i in range(self.centroids.shape[0]):
                color = plt.cm.tab10(
                    i % 10
                )  # choose a color based on the cluster index
                plt.scatter(
                    self.centroids[i, 0],
                    self.centroids[i, 1],
                    s=150,
                    marker="x",
                    color=color,
                    label=f"Niche {i}",
                )
                plt.scatter(
                    points[labels == i, 0],
                    points[labels == i, 1],
                    s=10,
                    marker=".",
                    color=color,
                )
        elif self.env.behavior_ndim >= 3:
            ax = plt.axes(projection="3d")

            for i in range(self.centroids.shape[0]):
                color = plt.cm.tab10(
                    i % 10
                )  # choose a color based on the cluster index
                ax.scatter(
                    self.centroids[i, 0],
                    self.centroids[i, 1],
                    self.centroids[i, 2],
                    s=150,
                    marker="x",
                    c=[color],
                    label=f"Niche {i}",
                )
                ax.scatter(
                    points[labels == i, 0],
                    points[labels == i, 1],
                    points[labels == i, 2],
                    s=10,
                    marker=".",
                    c=[color],
                )
        else:
            print("Not enough dimensions to plot centroids")
            return

        plt.savefig("logs/elm/MAPElites_centroids.png")

    def plot_behaviour_space(self):
        """Plot the first two dimensions (or three if available) of the behaviour space, along with the CVT centroids."""
        import matplotlib.pyplot as plt

        if self.env.behavior_ndim == 2:
            plt.figure()
            for i in range(self.centroids.shape[0]):
                color = plt.cm.tab10(i % 10)
                plt.scatter(
                    self.centroids[i, 0],
                    self.centroids[i, 1],
                    s=150,
                    marker="x",
                    color=color,
                    label=f"Niche {i}",
                )

                # get the first two dimensions for each behaviour in the history
                if self.genomes.history_length > 1:
                    phenotypes = [
                        g.to_phenotype()[:2]
                        for g in self.genomes.array[:, i]
                        if hasattr(g, "to_phenotype")
                    ]
                    if phenotypes:
                        hist = np.stack(phenotypes)
                        plt.scatter(
                            hist[:, 0], hist[:, 1], s=10, marker=".", color=color
                        )
                else:
                    g = self.genomes.array[i]
                    if hasattr(g, "to_phenotype"):
                        plt.scatter(
                            g.to_phenotype()[0],
                            g.to_phenotype()[1],
                            s=10,
                            marker=".",
                            color=color,
                        )

            plt.xlim([0, self.env.behavior_space[1, 0]])
            plt.ylim([0, self.env.behavior_space[1, 1]])

        elif self.env.behavior_ndim >= 3:
            plt.figure()
            ax = plt.axes(projection="3d")

            for i in range(self.centroids.shape[0]):
                color = plt.cm.tab10(i % 10)
                ax.scatter(
                    self.centroids[i, 0],
                    self.centroids[i, 1],
                    self.centroids[i, 2],
                    s=150,
                    marker="x",
                    c=[color],
                    label=f"Niche {i}",
                )

                # get the first three dimensions for each behaviour in the history
                if self.genomes.history_length > 1:
                    phenotypes = [
                        g.to_phenotype()[:3]
                        for g in self.genomes.array[:, i]
                        if hasattr(g, "to_phenotype")
                    ]
                    if phenotypes:
                        hist = np.stack(phenotypes)
                        ax.scatter(
                            hist[:, 0],
                            hist[:, 1],
                            hist[:, 2],
                            s=10,
                            marker=".",
                            c=[color],
                        )
                else:
                    g = self.genomes.array[i]
                    if hasattr(g, "to_phenotype"):
                        ax.scatter(
                            g.to_phenotype()[0],
                            g.to_phenotype()[1],
                            g.to_phenotype()[2],
                            s=10,
                            marker=".",
                            c=[color],
                        )

            ax.set_xlim([0, self.env.behavior_space[1, 0]])
            ax.set_ylim([0, self.env.behavior_space[1, 1]])
            ax.set_zlim([0, self.env.behavior_space[1, 2]])

        else:
            print("Not enough dimensions to plot behaviour space history")
            return

        plt.savefig("logs/elm/MAPElites_behaviour_history.png")
