from typing import List

import torch
from torch import nn
from transformers import pipeline

import wandb
from trlx.data.configs import TRLConfig
from trlx.model.accelerate_ppo_model import AcceleratePPOModel
from trlx.model.accelerate_ppo_softprompt_model import AcceleratePPOSoftpromptModel
from trlx.orchestrator.ppo_orchestrator import PPOOrchestrator
from trlx.pipeline.ppo_pipeline import PPOPipeline
from trlx.utils.loading import get_model, get_orchestrator, get_pipeline

from trlx.data.method_configs import register_method, PPOConfig
from dataclasses import dataclass


@dataclass
@register_method
class PPOSoftpromptConfig(PPOConfig):
    n_soft_tokens: int = None  # soft prompt support
    initialize_from_vocab: bool = True  # soft prompt support


if __name__ == "__main__":
    cfg = TRLConfig.load_yaml(
        "configs/ppo_softprompt_config.yml"
    )  # load softprompt config instead of original ppo one

    sentiment_pipe = pipeline(
        "sentiment-analysis", "lvwerra/distilbert-imdb", device=-1
    )

    def reward_fn(samples: List[str]):
        sent_kwargs = {
            "return_all_scores": True,
            "function_to_apply": None,
            "batch_size": cfg.method.chunk_size,
        }
        pipe_outputs = sentiment_pipe(samples, **sent_kwargs)
        scores = torch.tensor([output[1]["score"] for output in pipe_outputs])
        return scores

    model = get_model(cfg.model.model_type)(cfg)

    if model.accelerator.is_main_process:
        wandb.watch(model.model)

    pipeline: PPOPipeline = get_pipeline(cfg.train.pipeline)(model.tokenizer, cfg)
    orch: PPOOrchestrator = get_orchestrator(cfg.train.orchestrator)(
        model, pipeline, reward_fn=reward_fn, chunk_size=cfg.method.chunk_size
    )
    orch.make_experience(cfg.method.num_rollouts)
    model.learn()

    print("DONE!")
