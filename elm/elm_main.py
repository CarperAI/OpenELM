from elm.environments import ImageOptim, Sodarace, image_init_args, sodarace_init_args
from elm.map_elites import MAPElites

ENVS_DICT = {"sodarace": Sodarace, "imageoptim": ImageOptim}
ARG_DICT = {"sodarace": sodarace_init_args, "imageoptim": image_init_args}


class ELM:
    def __init__(self, cfg, diff_model_cls=None, env_args: dict = None) -> None:
        """
        Args:
            cfg: the config (e.g. OmegaConf who uses dot to access members).
            diff_model_cls: (Optional) The class of diff model. One can apply alternative models here for comparison.
            env_args: (Optional) The argument dict for Environment.
        """
        self.cfg = cfg

        # Get the defaults if `env_args` is not specified.
        if env_args is None:
            env_args = ARG_DICT[self.cfg.env_name]
        env_args["config"] = self.cfg  # Override default environment config

        # Override diff model if `diff_model_cls` is specified.
        if diff_model_cls is not None:
            self.diff_model = diff_model_cls(self.cfg)
            env_args = {**env_args, "diff_model": self.diff_model}
        else:
            self.diff_model = None

        self.seed = env_args["seed"]

        self.environment = ENVS_DICT[self.cfg.env_name](**env_args)
        self.map_elites = MAPElites(
            self.environment,
            n_bins=self.cfg.behavior_n_bins,
            history_length=self.cfg.evo_history_length,
        )

    def run(self) -> str:
        return self.map_elites.search(
            initsteps=self.cfg.evo_init_steps, totalsteps=self.cfg.evo_n_steps
        )
