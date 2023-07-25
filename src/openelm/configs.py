from dataclasses import dataclass, field
from typing import Any, Optional

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

AVAILABLE_GENERATION_MODELS_AA = [
    "luminous-base",
    "luminous-extended",
    "luminous-supreme",
]
AVAILABLE_CLASSIFIER_MODEL_AA = ["luminous-supreme-qdaif", "luminous-supreme-control"]
VALID_SOLUTION_INIT_METHODS = ["seed", "generated"]
VALID_MUTATION_METHODS = ["lmx_near", "replace"]


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
    temp: float = 0.8
    gen_max_len: int = 100
    batch_size: int = 32
    model_type: str = "hf"  # Can be "hf", "openai", etc
    model_path: str = MISSING  # Can be HF model name or path to local model
    logits_only: bool = False
    do_sample: bool = True
    num_return_sequences: int = 1
    trust_remote_code: bool = True  # needed for mosaicml/mpt-7b-instruct
    frequency_penalty: float = 0.0  # for more open-ended domains (note: maybe penalty doesn't work well experimentally)


@dataclass
class PromptModelConfig(ModelConfig):
    model_name: str = "prompt"
    model_path: str = "Salesforce/codegen-350M-mono"


@dataclass
class DiffModelConfig(ModelConfig):
    model_name: str = "diff"
    model_path: str = "CarperAI/diff-codegen-350m-v2"


@dataclass
class APIModelConfig(ModelConfig):
    model_name: str = "api"
    model_path: str = ""
    model_used: str = "luminous-base"
    api_token_file: str = "aa_client_token.txt"
    stop_sequences: list[Any] = field(
        default_factory=lambda: [
            "\n#",
            "\n##",
            "\n###",
            "###",
            "\n####",
            "\n#####",
            "####",
            "#####",
            "\n",
            "\n\n",
            "\n\n\n",
            "\n\n\n\n",
            "@@@",
            "#",
            "##",
            "\nHere",
            "\n\nHere",
        ]
    )

    def __post_init__(self):
        # assert use of valid model
        assert (
            self.model_used in AVAILABLE_GENERATION_MODELS_AA
        ), f"Model {self.model_used} is not supported. Please select one of the following models: {AVAILABLE_GENERATION_MODELS_AA}"


@dataclass
class QDConfig(BaseConfig):
    init_steps: int = 50
    total_steps: int = 2000
    history_length: int = 100
    save_history: bool = True
    save_snapshot_interval: Optional[int] = 500
    log_snapshot_dir: str = ""
    seed: Optional[int] = 42
    save_np_rng_state: bool = False
    load_np_rng_state: bool = False


@dataclass
class MAPElitesConfig(QDConfig):
    qd_name: str = "mapelites"
    map_grid_size: tuple[int, ...] = field(default_factory=lambda: (20,))


@dataclass
class LMXMapElitesConfig(QDConfig):
    qd_name: str = "lmx_mapelites"
    map_grid_size: tuple[int, ...] = field(default_factory=lambda: (20,))
    write_all_individuals_to_jsonl: bool = True
    append_bin_idx_to_batch: bool = True
    use_alt_depth_method: bool = True
    custom_ticks: Optional[list[float]] = field(
        default_factory=lambda: [
            0.005,
            0.01,
            0.015,
            0.02,
            0.03,
            0.04,
            0.05,
            0.10,
            0.20,
            0.50,
            0.80,
            0.90,
            0.95,
            0.96,
            0.97,
            0.98,
            0.985,
            0.99,
            0.995,
        ]
    )

    def __post_init__(self):
        if self.custom_ticks:
            assert self.map_grid_size[0] - 1 == len(
                self.custom_ticks
            ), f"the number of custom ticks {len(self.custom_ticks)} should be the number of bins {self.map_grid_size[0]} - 1"


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
    processes: int = 12
    batch_size: int = 1
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
            [0, 1000],
            [0, 1000],
            [0, 2000],
        ]
    )
    starting_seeds: list[str] = field(default_factory=lambda: ["square"])
    instruction: int = 1
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
class PromptEnvConfig(EnvConfig):
    env_name: str = "prompt_evolution"
    task_name: str = "antonym"  # toy or antonym or animal or cot
    evals_per_prompt: int = 10


@dataclass
class LMXGenerationEnvConfig(EnvConfig):
    env_name: str = "lmx_generation"
    behavior_measure: str = "ai_feedback"
    solution_init_method: str = "seed"  # seed, generated
    mutation_method: str = "lmx_near"  # replace, lmx_near
    fitness_query: str = "A fantasy story about a suspicious spy and a rich politician"
    few_shot_template: str = "Here is a random example of a fantasy story about a suspicious spy and a rich politician:"
    instruction_prompt: str = "Determine the sentiment of the text by writing 'positive' or 'negative' in the output."
    quality_feedback_prompt: = """Determine if the input text contains a high-quality short story containing two characters, a suspicious spy, and a rich politician. Answer "yes" if the input contains a high-quality short story about a suspicious spy and a rich politician, otherwise answer "no"."""
    max_prompt_pool_size: int = 100  # for storage of few-shot pool, based on accepted fit solutions set to 3*map size for depth search
    init_size_prompt_pool: int = (
        5  # with generated init method, should be slightly more than few-shot size
    )
    prompt_pool_path: str = (
        "src/openelm/environments/lmx_seed_pools/short_story_seed_pool.txt"
    )
    add_only_improved_completions_to_prompt_pool: bool = True
    classifier_model: str = "luminous-supreme-qdaif"
    fitness_method: str = "ai_feedback"

    def __post_init__(self):
        # assert valid config values
        assert (
            self.classifier_model in AVAILABLE_CLASSIFIER_MODEL_AA
        ), f"Model {self.classifier_model} is currently not supported. Please pick one of the following models: {AVAILABLE_CLASSIFIER_MODEL_AA}"
        assert (
            self.solution_init_method in VALID_SOLUTION_INIT_METHODS
        ), f"Please select one of the following options as init method: {VALID_SOLUTION_INIT_METHODS}"
        assert (
            self.mutation_method in VALID_MUTATION_METHODS
        ), f"Please select one of the following mutation methods: {VALID_MUTATION_METHODS}"

        # set some classifier model specific paramters
        if self.classifier_model == "luminous-supreme-qdaif":
            extra_prefix = " \n"
            extra_suffix = ""
        elif self.classifier_model == "luminous-supreme-control":
            extra_prefix = ""
            extra_suffix = " Please respond with a single word only."
        else:
            raise NotImplementedError

        self.ai_feedback_entries = {  # entries to setup ai feedback.
            "sentiment": {
                "answer_space": [
                    f"{extra_prefix}positive",
                    f"{extra_prefix}negative",
                ],
                "feedback_prompt_template": f"### Instruction:\n{self.instruction_prompt}{extra_suffix}\n\n### Input:\n{{genotype}}\n\n### Response:",
            },
        }
        self.quality_ai_feedback_entries = {
            "quality": {
                "answer_space": [
                    f"{extra_prefix}yes",
                    f"{extra_prefix}no",
                ],
                "feedback_prompt_template": f"### Instruction:\n{self.quality_feedback_prompt}{extra_suffix}\n\n### Input:\n{{genotype}}\n\n### Response:",
            },
        }


defaults_elm = [
    {"model": "api"},
    {"qd": "lmx_mapelites"},
    {"env": "lmx_generation"},
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
    save_result_obj: bool = False  # if saving results, include the whole output text from model for each iteration (which can get long)
    probsol: bool = True  # generate new problem+solution pairs from given problems instead of just solutions to given problems
    eval_k: int = (
        -1
    )  # set >0 to evaluate pass@k of previous runs using this k, instead of doing a new run
    eval_timestamp: str = ""  # optionally provide timestamp of run to eval pass@k, otherwise eval with latest run of every problem


def register_configstore() -> ConfigStore:
    """Register configs with Hydra's ConfigStore."""
    cs = ConfigStore.instance()
    cs.store(group="env", name="sodarace", node=SodaraceEnvConfig)
    cs.store(group="env", name="image_evolution", node=ImageEnvConfig)
    cs.store(group="env", name="string_evolution", node=StringEnvConfig)
    cs.store(group="env", name="lmx_generation", node=LMXGenerationEnvConfig)
    cs.store(group="env", name="p3_probsol", node=P3ProbSolEnvConfig)
    cs.store(group="env", name="p3_problem", node=P3ProblemEnvConfig)
    cs.store(group="env", name="prompt_evolution", node=PromptEnvConfig)
    cs.store(group="qd", name="mapelites", node=MAPElitesConfig)
    cs.store(group="qd", name="cvtmapelites", node=CVTMAPElitesConfig)
    cs.store(group="qd", name="lmx_mapelites", node=LMXMapElitesConfig)
    cs.store(group="model", name="prompt", node=PromptModelConfig)
    cs.store(group="model", name="diff", node=DiffModelConfig)
    cs.store(group="model", name="api", node=APIModelConfig)
    cs.store(name="elmconfig", node=ELMConfig)
    cs.store(name="p3config", node=P3Config)
    return cs


CONFIGSTORE = register_configstore()
