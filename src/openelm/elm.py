from typing import Optional

from hydra.core.hydra_config import HydraConfig

from openelm.configs import (
    DiffModelConfig,
    ELMConfig,
    PromptModelConfig,
    APIModelConfig,
)
from openelm.environments import ENVS_DICT, QD_DICT
from openelm.mutation_model import DiffModel, MutationModel, PromptModel, AlephAlphaLLM


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
        self.config.qd.output_dir = HydraConfig.get().runtime.output_dir
        env_name: str = self.config.env.env_name
        qd_name: str = self.config.qd.qd_name
        if isinstance(self.config.model, PromptModelConfig):
            self.mutation_model: MutationModel = PromptModel(self.config.model)
        elif isinstance(self.config.model, DiffModelConfig):
            self.mutation_model = DiffModel(self.config.model)
        elif isinstance(self.config.model, APIModelConfig):
            self.mutation_model = AlephAlphaLLM(self.config.model)
        else:
            raise NotImplementedError()

        self.environment = ENVS_DICT[env_name](
            config=self.config.env,
            mutation_model=self.mutation_model,
        )
        self.qd_algorithm = QD_DICT[qd_name](
            env=self.environment,
            config=self.config.qd,
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
