from dataclasses import dataclass, field
from time import time
from typing import Any, Optional

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from tqdm import trange

from openelm.codegen import model_setup, sample
from openelm.configs import BaseConfig
from openelm.environments import SQUARE_SEED


@dataclass
class BenchmarkSpeedConfig(BaseConfig):
    hydra: Any = field(
        default_factory=lambda: {
            "run": {"dir": "logs/benchmarks/lm_speed/${now:%Y-%m-%d-%H-%M-%S}"}
        }
    )
    seed: Optional[int] = None
    fp16: bool = True
    cuda: bool = True
    model: str = "Salesforce/codegen-2B-mono"
    gpus: int = 1
    temp: float = 0.85
    top_p: float = 0.95
    gen_max_len: int = 512
    batch_size: int = 16
    num_iters: int = 1


cs = ConfigStore.instance()
cs.store(name="config", node=BenchmarkSpeedConfig)


@hydra.main(version_base="1.2", config_name="config")
def main(cfg):
    print(OmegaConf.to_yaml(cfg))
    cfg = OmegaConf.to_object(cfg)
    model, tokenizer, device = model_setup(cfg)

    encoding = tokenizer(
        [SQUARE_SEED["program_str"]],
        padding=True,
        truncation=True,
        return_tensors="pt",
    ).to(device)
    print(f"Prompt length: {encoding['input_ids'].shape[1]}")
    start = time()
    for _ in trange(cfg.num_iters):

        _ = sample(encoding, cfg, model, tokenizer, decode=False)

    print(f"Average time: {(time() - start) / cfg.num_iters} seconds")


if __name__ == "__main__":
    main()
