from dataclasses import dataclass, field
from typing import Any, Optional

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING


@dataclass
class BaseConfig:
    output_dir: str = "logs/"


@dataclass
class ModelConfig(BaseConfig):
    fp16: bool = True
    cuda: bool = True
    gpus: int = 1
    seed: Optional[int] = None
    deterministic: bool = False
    top_p: float = 0.95
    temp: float = 1.1
    gen_max_len: int = 512
    batch_size: int = 10
    model_type: str = "hf"  # Can be "hf", "openai", etc
    model_path: str = MISSING  # Can be HF model name or path to local model
    logits_only: bool = False
    do_sample: bool = True
    num_return_sequences: int = 1
    trust_remote_code: bool = True  # needed for mosaicml/mpt-7b-instruct


@dataclass
class PromptModelConfig(ModelConfig):
    model_name: str = "prompt"
    model_path: str = "Salesforce/codegen-350M-mono"


@dataclass
class DiffModelConfig(ModelConfig):
    model_name: str = "diff"
    model_path: str = "CarperAI/diff-codegen-350m-v2"


@dataclass
class QDConfig(BaseConfig):
    init_steps: int = 250
    total_steps: int = 2500
    history_length: int = 1
    save_history: bool = False
    save_snapshot_interval: int = 10
    log_snapshot_dir: str = ""
    seed: Optional[int] = 42
    save_np_rng_state: bool = False
    load_np_rng_state: bool = False
    crossover: bool = False
    crossover_parents: int = 2


@dataclass
class MAPElitesConfig(QDConfig):
    qd_name: str = "mapelites"
    map_grid_size: tuple[int, ...] = field(default_factory=lambda: (5,))


@dataclass
class CVTMAPElitesConfig(QDConfig):
    qd_name: str = "cvtmapelites"
    n_niches: int = 12
    cvt_samples: int = 10000


@dataclass
class EnvConfig(BaseConfig):
    timeout: float = 5.0  # Seconds
    sandbox: bool = False
    sandbox_server: str = "http://localhost:5000"
    processes: int = 1
    batch_size: int = 10  # Batch size of MAP-Elites
    env_name: str = MISSING
    debug: bool = False
    seed: Optional[int] = 42


@dataclass
class SodaraceEnvConfig(EnvConfig):
    env_name: str = "sodarace"
    eval_ms: int = 1000  # Milliseconds
    behavior_space: list[list[float]] = field(
        default_factory=lambda: [
            # Height, Width, Mass dimensions
            [0, 500],
            [0, 500],
            [0, 1000],
        ]
    )
    starting_seeds: list[str] = field(default_factory=lambda: ["square"])
    instruction: int = 2
    crossover: bool = False


@dataclass
class ImageEnvConfig(EnvConfig):
    env_name: str = "image_evolution"
    behavior_mode: str = "3-channel"
    target: str = "circle"


@dataclass
class StringEnvConfig(EnvConfig):
    env_name: str = "string_evolution"
    target: str = "MapElites"


@dataclass
class P3ProblemEnvConfig(EnvConfig):
    env_name: str = "p3_problem"
    prompt_size: str = "long"  # med or long
    timeout: float = 1.0  # timeout for running a solution
    starting_seed: int = field(
        default_factory=lambda: 3
    )  # index of p3 dataset to use as puzzle to mutate
    embedding_model_type: str = "hf"  # openai or hf
    embedding_model_path: str = MISSING  # e.g. hf: Salesforce/codegen-350M-mono ; openai: text-embedding-ada-002


@dataclass
class P3ProbSolEnvConfig(EnvConfig):
    env_name: str = "p3_probsol"
    prompt_size: str = "long"  # med or long
    timeout: float = 1.0  # timeout for running a solution
    starting_seed: int = field(
        default_factory=lambda: 3
    )  # index of p3 dataset to use as puzzle to mutate
    eval_k: int = 100  # k for pass@k for fitness
    embedding_model_type: str = "hf"  # openai or hf
    embedding_model_path: str = MISSING  # e.g. hf: Salesforce/codegen-350M-mono ; openai: text-embedding-ada-002


@dataclass
class QDEnvConfig(EnvConfig):
    env_name: str = "qdaif"
    behavior_space: list[list[float]] = field(
        default_factory=lambda: [
            [0, 5],
            [0, 5],
        ]
    )


@dataclass
class PromptEnvConfig(EnvConfig):
    env_name: str = "prompt_evolution"
    task_name: str = "antonym"  # toy or antonym or animal or cot
    evals_per_prompt: int = 10


defaults_elm = [
    {"model": "prompt"},
    {"qd": "mapelites"},
    {"env": "sodarace"},
    "_self_",
]


@dataclass
class ELMConfig(BaseConfig):
    hydra: Any = field(
        default_factory=lambda: {
            "run": {
                "dir": "logs/elm/${hydra.job.override_dirname}/${now:%y-%m-%d_%H:%M}"
            }
        }
    )
    defaults: list[Any] = field(default_factory=lambda: defaults_elm)
    model: Any = MISSING
    qd: Any = MISSING
    env: Any = MISSING
    run_name: Optional[str] = None


defaults_p3 = [
    {"model": "prompt"},
    {"env": "p3"},
    "_self_",
]


@dataclass
class P3Config(BaseConfig):
    hydra: Any = field(
        default_factory=lambda: {
            "run": {
                "dir": "logs/p3/${hydra.job.override_dirname}/${now:%y-%m-%d_%H:%M}"
            }
        }
    )
    defaults: list[Any] = field(default_factory=lambda: defaults_p3)
    model: Any = MISSING
    env: Any = MISSING
    run_name: Optional[str] = None
    # --- The below are for run_p3.py
    iterations_per_puzzle: int = 128
    starting_seeds: list[int] = field(
        default_factory=lambda: [3]
    )  # indices of selection of puzzles to evaluate with
    save_results: bool = True
    save_result_obj: bool = False  # if saving results, include the whole output
    # text from model for each iteration (which can get long)
    probsol: bool = True  # generate new problem+solution pairs from given
    # problems instead of just solutions to given problems
    # set eval_k >0 to evaluate pass@k of previous runs using this k, instead of
    # doing a new run
    eval_k: int = -1
    eval_timestamp: str = ""  # optionally provide timestamp of run to eval
    # pass@k, otherwise eval with latest run of every problem


def register_configstore() -> ConfigStore:
    """Register configs with Hydra's ConfigStore."""
    cs = ConfigStore.instance()
    cs.store(group="env", name="sodarace", node=SodaraceEnvConfig)
    cs.store(group="env", name="image_evolution", node=ImageEnvConfig)
    cs.store(group="env", name="string_evolution", node=StringEnvConfig)
    cs.store(group="env", name="p3_probsol", node=P3ProbSolEnvConfig)
    cs.store(group="env", name="p3_problem", node=P3ProblemEnvConfig)
    cs.store(group="env", name="prompt_evolution", node=PromptEnvConfig)
    cs.store(group="env", name="qdaif", node=QDEnvConfig)
    cs.store(group="qd", name="mapelites", node=MAPElitesConfig)
    cs.store(group="qd", name="cvtmapelites", node=CVTMAPElitesConfig)
    cs.store(group="model", name="prompt", node=PromptModelConfig)
    cs.store(group="model", name="diff", node=DiffModelConfig)
    cs.store(name="elmconfig", node=ELMConfig)
    cs.store(name="p3config", node=P3Config)
    return cs


CONFIGSTORE = register_configstore()
