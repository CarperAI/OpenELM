from dataclasses import dataclass

import trlx
from datasets import load_dataset

# to register the added softprompt model, and supported orchestrator, need to import here
from model.accelerate_ppo_softprompt_model import AcceleratePPOSoftpromptModel
from orchestrator.ppo_softprompt_orchestrator import PPOSoftpromptOrchestrator
from transformers import pipeline
from trlx.data.configs import TRLConfig
from trlx.data.method_configs import register_method
from trlx.model.nn.ppo_models import PPOConfig


@dataclass
@register_method
class PPOSoftpromptConfig(PPOConfig):
    n_soft_tokens: int = None
    initialize_from_vocab: bool = True  # of softprompt
    tune_v_head: bool = True  # set in case whole model is frozen (except softprompt)
    measure_soft_embedding_drift: bool = True  # for debugging purposes


if __name__ == "__main__":
    sentiment_fn = pipeline("sentiment-analysis", "lvwerra/distilbert-imdb", device=-1)

    def reward_fn(samples):
        outputs = sentiment_fn(samples, return_all_scores=True)
        sentiments = [output[1]["score"] for output in outputs]
        return sentiments

    # Take few words off of movies reviews as prompts
    imdb = load_dataset("imdb", split="train+test")
    prompts = [" ".join(review.split()[:4]) for review in imdb["text"]]

    config = TRLConfig.load_yaml("configs/ppo_softprompt_config.yml")

    model = trlx.train(
        "lvwerra/gpt2-imdb",
        reward_fn=reward_fn,
        prompts=prompts,
        eval_prompts=["I don't know much about Hungarian underground"] * 64,
        config=config,
    )

    print("DONE!")
