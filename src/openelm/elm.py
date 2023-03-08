from typing import Optional

from openelm.configs import DiffModelConfig, ELMConfig, PromptModelConfig
from openelm.environments import ENVS_DICT
from openelm.map_elites import MAPElites
from openelm.mutation_model import DiffModel, MutationModel, PromptModel


class ELM:
    def __init__(self, config: ELMConfig) -> None:
        """
        The main class of ELM.

        This class will load a diff model, an environment, and a QD algorithm
        from the passed config.

        Args:
            config: The config containing the diff model, environment, and QD algorithm.
        """
        self.config: ELMConfig = config
        env_name: str = self.config.env.env_name
        if isinstance(self.config.model, PromptModelConfig):
            self.mutation_model: MutationModel = PromptModel(self.config.model)
        elif isinstance(self.config.model, DiffModelConfig):
            self.mutation_model = DiffModel(self.config.model)

        self.environment = ENVS_DICT[env_name](
            config=self.config.env,
            mutation_model=self.mutation_model,
        )
        self.qd_algorithm = MAPElites(
            self.environment,
            map_grid_size=self.config.qd.map_grid_size,
            history_length=self.config.qd.history_length,
            save_history=self.config.qd.save_history,
        )

    def run(
        self, init_steps: Optional[int] = None, total_steps: Optional[int] = None
    ) -> str:
        """
        Run the ELM algorithm to evolve the population in the environment.

        Args:
            init_steps: The number of steps to run the initialisation phase.
            total_steps: The number of steps to run the QD algorithm in total,
            including init_steps.

        Returns:
            str: A string representing the maximum fitness genotype. The
            `qd_algorithm` class attribute will be updated.
        """
        if init_steps is None:
            init_steps = self.config.qd.init_steps
        if total_steps is None:
            total_steps = self.config.qd.total_steps
        return self.qd_algorithm.search(init_steps=init_steps, total_steps=total_steps)
