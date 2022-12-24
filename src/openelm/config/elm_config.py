from dataclasses import dataclass

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING


@dataclass
class ConfigClass:
    model: str = MISSING
    checkpoints_dir: str = MISSING
    epochs: int = MISSING
    batch_size: int = MISSING
    fp16: bool = MISSING
    cuda: bool = MISSING
    gpus: int = MISSING
    seed: int = MISSING
    deterministic: bool = MISSING
    top_p: float = MISSING
    temp: float = MISSING
    timeout: float = MISSING
    gen_max_len: int = MISSING
    evo_init_steps: int = MISSING
    evo_n_steps: int = MISSING
    behavior_n_bins: int = MISSING
    evo_history_length: int = MISSING
    evaluation_steps: int = MISSING
    pad_token: str = MISSING
    env_name: str = MISSING
    run_name: str = MISSING


cs = ConfigStore.instance()
cs.store(name="elm_cfg", node=ConfigClass)
