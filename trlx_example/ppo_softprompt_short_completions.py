from datasets import load_dataset
from transformers import pipeline
from transformers import AutoTokenizer

import trlx
from trlx.data.configs import TRLConfig
from trlx.model.accelerate_ppo_softprompt_model import AcceleratePPOSoftpromptModel
from trlx.data.method_configs import register_method, PPOConfig
from dataclasses import dataclass


@dataclass
@register_method
class PPOSoftpromptConfig(PPOConfig):
    n_soft_tokens: int = None
    initialize_from_vocab: bool = True  # of softprompt
    tune_v_head: bool = True  # set in case whole model is frozen (except softprompt)


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("lvwerra/gpt2-imdb")
    config = TRLConfig.load_yaml("configs/ppo_softprompt_config.yml")

    max_gen_length = config.method.gen_kwargs['max_length'] # set reward as 0 if max length is reached.

    def reward_fn(samples):
        samples_tokenized = tokenizer(samples)
        samples_token_ids = samples_tokenized.data['input_ids']
        # reward = [(1 - (len(item_ids)/max_gen_length)) for item_ids in samples_token_ids]
        reward = [(1 - len(item_ids)/max_gen_length) for item_ids in samples_token_ids]

        return reward # list of scalar reward scores for each response
    
    model = trlx.train(
        "lvwerra/gpt2-imdb",
        reward_fn=reward_fn,
        prompts=["The following is a movie quote:"] * 64,
        eval_prompts=["The following is a movie quote:"] * 64,
        config=config
    )

    print("DONE!")
