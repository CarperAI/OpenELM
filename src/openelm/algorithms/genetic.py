import os
import pickle
import random
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from openelm.configs import QDConfig
from openelm.environments import BaseEnvironment, Genotype

Phenotype = Optional[np.ndarray]
MapIndex = Optional[tuple]
Individual = Tuple[np.ndarray, float]


class Pool:
    """The pool stores a set of solutions or individuals."""

    def __init__(self, pool_size: int):
        """Initializes an empty pool.

        Args:
            pool_size (int): The number of solutions to store in the pool.
            history_length (int): The number of historical solutions
                to maintain in the pool.
        """
        self.pool_size = pool_size
        self.pool = []

    def add(self, solution, fitness):
        """Adds a solution to the pool.

        If the pool is full, the oldest solution is removed. The solution
        is also added to the history.

        Args:
            solution: The solution to add to the pool.
        """
        # if new fitness is better than the worst, add it to the pool
        if fitness > self.pool[-1][1]:
            if len(self.pool) >= self.pool_size:
                self.pool.pop(0)
            self.pool.append((solution, fitness))
            # sort the pool by fitness
            self.pool.sort(key=lambda x: x[1], reverse=True)


class MAPElitesBase:
    """
    Base class for a genetic algorithm
    """

    def __init__(
        self,
        env,
        config: QDConfig,
        init_pool: Optional[Pool] = None,
    ):
        """
        The base class for a genetic algorithm, implementing common functions and search.

        Args:
            env (BaseEnvironment): The environment to evaluate solutions in. This
            should be a subclass of `BaseEnvironment`, and should implement
            methods to generate random solutions, mutate existing solutions,
            and evaluate solutions for their fitness in the environment.
            config (QDConfig): The configuration for the algorithm.
            init_pool (Pool, optional): A pool to use for the algorithm. If not passed,
            a new pool will be created. Defaults to None.
        """
        self.env: BaseEnvironment = env
        self.config: QDConfig = config
        self.save_history = self.config.save_history
        self.save_snapshot_interval = self.config.save_snapshot_interval
        self.start_step = 0
        self.save_np_rng_state = self.config.save_np_rng_state
        self.load_np_rng_state = self.config.load_np_rng_state
        self.rng = np.random.default_rng(self.config.seed)
        self.rng_generators = None

        self._init_pool(init_pool, self.config.log_snapshot_dir)

    def to_mapindex(self, b: Phenotype) -> MapIndex:
        """Converts a phenotype (position in behaviour space) to a map index."""
        raise NotImplementedError

    def _init_pool(
        self, init_map: Optional[Pool] = None, log_snapshot_dir: Optional[str] = None
    ):
        if init_map is None and log_snapshot_dir is None:
            self.pool = Pool(self.config.pool_size)
        elif init_map is not None and log_snapshot_dir is None:
            self.pool = init_map
        elif init_map is None and log_snapshot_dir is not None:
            self.pool = Pool(self.config.pool_size)
            log_path = Path(log_snapshot_dir)
            if log_snapshot_dir and os.path.isdir(log_path):
                stem_dir = log_path.stem

                assert (
                    "step_" in stem_dir
                ), f"loading directory ({stem_dir}) doesn't contain 'step_' in name"
                self.start_step = (
                    int(stem_dir.replace("step_", "")) + 1
                )  # add 1 to correct the iteration steps to run

                snapshot_path = log_path / "pool.pkl"
                assert os.path.isfile(
                    snapshot_path
                ), f'{log_path} does not contain map snapshot "pool.pkl"'
                # first, load arrays and set them in Maps
                # Load maps from pickle file
                with open(snapshot_path, "rb") as f:
                    self.pool = pickle.load(f)

        print("Loading finished")

    def random_selection(self) -> MapIndex:
        """Randomly select a niche (cell) in the map that has been explored."""
        return random.choice(self.pool.pool)

    def search(self, init_steps: int, total_steps: int, atol: float = 0.0) -> str:
        """
        Run the genetic algorithm.

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
        total_steps = int(total_steps)
        for n_steps in range(total_steps):
            if n_steps < init_steps:
                # Initialise by generating initsteps random solutions
                new_individuals: list[Genotype] = self.env.random()
            else:
                # Randomly select a batch of individuals
                batch: list[Genotype] = []
                for _ in range(self.env.batch_size):
                    item = self.random_selection()
                    batch.append(item)
                # Mutate
                new_individuals = self.env.mutate(batch)

            for individual in new_individuals:
                # Evaluate fitness
                fitness = self.env.fitness(individual)
                if np.isinf(fitness):
                    continue
                self.pool.add(individual, fitness)
