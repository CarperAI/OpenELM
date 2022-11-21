from elm.diff_model import DiffModel
from elm.environments import IMAGE_SEED, ImageOptim, Sodarace
from elm.environments.sodaracer import SQUARE_SEED
from elm.map_elites import MAPElites

ENVS_DICT = {"sodarace": Sodarace, "imageoptim": ImageOptim}


class ELM:
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.diff_model = DiffModel(self.cfg)
        if self.cfg.env_name == "sodarace":
            self.seed = SQUARE_SEED
        elif self.cfg.env_name == "imageoptim":
            self.seed = IMAGE_SEED
        self.environment = ENVS_DICT[self.cfg.env_name](
            seed=self.seed,
            diff_model=self.diff_model,
            eval_steps=self.cfg.evaluation_steps,
        )
        self.map_elites = MAPElites(
            self.environment,
            n_bins=self.cfg.behavior_n_bins,
            history_length=self.cfg.evo_history_length,
        )

    def run(self) -> str:
        return self.map_elites.search(
            initsteps=self.cfg.evo_init_steps, totalsteps=self.cfg.evo_n_steps
        )