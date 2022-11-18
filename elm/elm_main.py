import hydra
from omegaconf import OmegaConf

from elm.constants import SRC_PATH
from elm.diff_model import DiffModel
from elm.environments import IMAGE_SEED, ImageOptim, Sodarace
from elm.environments.sodaracer import SQUARE_SEED
from elm.map_elites import MAPElites


class ELM:
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.diff_model = DiffModel(self.cfg)
        if self.cfg.env_name == "sodarace":
            self.seed = SQUARE_SEED
            self.environment = Sodarace(
                seed=self.seed,
                diff_model=self.diff_model,
                eval_steps=self.cfg.evaluation_steps,
            )
        elif self.cfg.env_name == "imageoptim":
            self.seed = IMAGE_SEED
            self.environment = ImageOptim(
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


# Load hydra config from yaml files and command line arguments.
@hydra.main(
    config_path=str(SRC_PATH / "config"), config_name="elm_cfg", version_base="1.2"
)
def main(cfg):
    print("----------------- Config ---------------")
    print(OmegaConf.to_yaml(cfg))
    print("-----------------  End -----------------")
    elm = ELM(cfg)
    print("Best Sodaracer: ", elm.run())


if __name__ == "__main__":
    main()
