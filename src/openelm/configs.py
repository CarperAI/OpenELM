from dataclasses import dataclass, field
from typing import Any, Optional

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING


@dataclass
class BaseConfig:
    pass


@dataclass
class ConfigClass(BaseConfig):
    model: str = MISSING
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
    env_name: str = MISSING
    run_name: str = MISSING


@dataclass
class SodaraceELMConfig(BaseConfig):
    hydra: Any = field(
        default_factory=lambda: {
            "run": {"dir": "logs/elm/sodarace/${hydra.job.override_dirname}"}
        }
    )
    model: str = "codegen-350M-mono"
    batch_size: int = 32
    fp16: bool = True
    cuda: bool = True
    gpus: int = 1
    seed: Optional[int] = None
    debug: bool = False
    deterministic: bool = False
    top_p: float = 0.95
    temp: float = 0.85
    timeout: float = 5.0  # Seconds
    evaluation_steps: int = 1000  # Milliseconds
    gen_max_len: int = 768
    evo_init_steps: int = 10
    evo_n_steps: int = 20
    behavior_n_bins: int = 12
    evo_history_length: int = 1
    processes: int = 12
    run_name: Optional[str] = None
    sandbox: bool = True


@dataclass
class ImageELMConfig(BaseConfig):
    hydra: Any = field(
        default_factory=lambda: {
            "run": {"dir": "logs/elm/image/${hydra.job.override_dirname}"}
        }
    )
    model: str = "codegen-350M-mono"
    batch_size: int = 32
    fp16: bool = True
    cuda: bool = True
    gpus: int = 1
    seed: Optional[int] = None
    debug: bool = False
    deterministic: bool = False
    top_p: float = 0.95
    temp: float = 0.85
    timeout: float = 5.0  # Seconds
    evaluation_steps: int = 1000  # Milliseconds
    gen_max_len: int = 1024
    evo_init_steps: int = 10
    evo_n_steps: int = 15
    behavior_n_bins: int = 12
    evo_history_length: int = 1
    processes: int = 12
    run_name: Optional[str] = None
    sandbox: bool = True


# TODO: Hierarchy of configs
# e.g. ModelConfig, QDConfig, EnvConfig, etc.
# Also add base ELMConfig(BaseConfig)


cs = ConfigStore.instance()
cs.store(name="elm_cfg", node=ConfigClass)
