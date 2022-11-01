from dataclasses import dataclass

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, DictConfig, OmegaConf


@dataclass
class ConfigClass:
    """Config class for ELM."""
    model: str = MISSING
    checkpoints_dir: str = MISSING
    cuda: bool = MISSING
    gpus: int = MISSING
    seed: int = MISSING
    fp16: bool = MISSING
    deterministic: bool = MISSING
    top_p: float = MISSING
    temp: float = MISSING
    pad_token: int = MISSING
    timeout: float = MISSING
    gen_max_length: int = MISSING
    batch_size: int = MISSING
    evo_init_steps: int = MISSING
    evo_n_steps: int = MISSING
    behavior_n_bins: int = MISSING
    evo_history_length: int = MISSING
    evaluation_steps: int = MISSING
    env_name: str = MISSING
    learning_rate: float = MISSING
    run_name: str = MISSING


cs = ConfigStore.instance()
cs.store(name="elm_cfg", node=ConfigClass)
