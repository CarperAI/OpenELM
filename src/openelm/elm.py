from typing import Optional

from openelm.configs import DiffModelConfig, ELMConfig, PromptModelConfig
from openelm.environments import ENVS_DICT
from openelm.map_elites import MAPElites
from openelm.mutation_model import DiffModel, MutationModel, PromptModel
from openelm.utils import validate_config


class ELM:
    def __init__(self, cfg: ELMConfig) -> None:
        """
        The main class of ELM.

        This class will load a diff model, an environment, and a QD algorithm
        from the passed config.

        Args:
            cfg: The config containing the diff model, environment, and QD algorithm.
        """
        self.cfg: ELMConfig = cfg
        # TODO: Make sure all seeds are available to choose
        # TODO: rename mutation_model.py to mutation_models.py
        env_name: str = self.cfg.env.env_name
        if isinstance(self.cfg.model, PromptModelConfig):
            self.mutation_model: MutationModel = PromptModel(self.cfg.model)
        elif isinstance(self.cfg.model, DiffModelConfig):
            self.mutation_model: MutationModel = DiffModel(self.cfg.model)

        self.environment = ENVS_DICT[env_name](
            config=self.cfg.env,
            mutation_model=self.mutation_model,
        )
        self.qd_algorithm = MAPElites(
            self.environment,
            map_grid_size=self.cfg.qd.map_grid_size,
            history_length=self.cfg.qd.history_length,
            save_history=self.cfg.qd.save_history,
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
            init_steps = self.cfg.qd.init_steps
        if total_steps is None:
            total_steps = self.cfg.qd.total_steps
        return self.qd_algorithm.search(init_steps=init_steps, total_steps=total_steps)
