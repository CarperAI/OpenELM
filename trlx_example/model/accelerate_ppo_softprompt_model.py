import os
from abc import abstractmethod
from typing import Dict, Iterable, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from accelerate import Accelerator
from torch.utils.data import DataLoader
from torchtyping import TensorType
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer

import wandb
from trlx.data.accelerate_base_datatypes import PromptBatch
from trlx.data.configs import TRLConfig
from trlx.model import BaseRLModel, register_model
from trlx.model.accelerate_base_model import AccelerateRLModel
from trlx.model.accelerate_ppo_model import AcceleratePPOModel
from trlx.model.nn.ppo_models import GPTHeadWithValueModel, GPTHydraHeadWithValueModel
from trlx.pipeline.ppo_pipeline import PPORolloutStorage
from trlx.utils import Clock, rampup_decay, safe_mkdir, topk_mask
from trlx.utils.modeling import clip_by_value, logprobs_from_logits, whiten


class SoftEmbedding(nn.Module):
    def __init__(
        self,
        wte: nn.Embedding,
        n_tokens: int = 10,
        random_range: float = 0.5,
        initialize_from_vocab: bool = True,
    ):
        """appends learned embedding as prefix

        From: https://github.com/kipgparker/soft-prompt-tuning

        Args:
            wte (nn.Embedding): original transformer word embedding
            n_tokens (int, optional): number of tokens for task. Defaults to 10.
            random_range (float, optional): range to init embedding (if not initialize from vocab). Defaults to 0.5.
            initialize_from_vocab (bool, optional): initalizes from default vocab. Defaults to True.
        """
        super().__init__()
        self.wte = wte
        self.n_tokens = n_tokens
        self.padding_token_id = 50256 # used when input tensors are prefix padded
        self.learned_embedding = nn.parameter.Parameter(
            self.initialize_embedding(
                wte, n_tokens, random_range, initialize_from_vocab
            )
        )

    def initialize_embedding(
        self,
        wte: nn.Embedding,
        n_tokens: int = 10,
        random_range: float = 0.5,
        initialize_from_vocab: bool = True,
    ):
        """initializes learned embedding
        Args:
            same as __init__
        Returns:
            torch.float: initialized using original schemes
        """
        if initialize_from_vocab:
            return self.wte.weight[:n_tokens].clone().detach()
        return torch.FloatTensor(n_tokens, wte.weight.size(1)).uniform_(
            -random_range, random_range
        )

    def forward(self, tokens):
        """run forward pass
        Args:
            tokens (torch.long): input tokens before encoding
        Returns:
            torch.float: encoding of text concatenated with learned task specifc embedding
        """
        prompt_tokens = tokens[:, self.n_tokens :] # without extra soft prompt padding
        if self.padding_token_id in prompt_tokens[:, 0]: # padding is applied as prefix
            seq_embedding = self.wte(tokens)
            padding_tensor = torch.tensor([self.padding_token_id]).to(seq_embedding.device)
            
            # index in each sequence in tokens just after last prefix padding
            # this would be where the (first) soft prompt embedding should be set
            first_prompt_indices = (prompt_tokens == padding_tensor).int().argmin(axis=1)
            
            # replace embeddings at soft prompt indices with correct soft embeddings
            for batch_idx, first_prompt_idx in enumerate(first_prompt_indices.tolist()):
                seq_embedding[batch_idx, first_prompt_idx:first_prompt_idx+self.n_tokens] = self.learned_embedding

            return seq_embedding
        else:
            input_embedding = self.wte(prompt_tokens)
            learned_embedding = self.learned_embedding.repeat(input_embedding.size(0), 1, 1)
            return torch.cat([learned_embedding, input_embedding], 1)


@register_model
class AcceleratePPOSoftpromptModel(AcceleratePPOModel):
    def __init__(self, config, train_mode=True):
        super().__init__(config)

        assert (
            config.method.n_soft_tokens > 0
        ), "Number of soft prompt tokens should be >=1"

        self.soft_dummy_token_id = 50256  # dummy token for padding soft prompts

        # account for extra prefix tokens
        self.config.method.gen_kwargs["max_length"] += self.n_soft_tokens
        self.config.method.gen_kwargs["min_length"] += self.n_soft_tokens

    def get_arch(self, config: TRLConfig):
        # TODO: set only self.learned_embedding as learnable parameter in case of fully frozen layers model
        model = GPTHydraHeadWithValueModel(
            self.config.model.model_path, self.config.model.num_layers_unfrozen
        )

        # here, we setup softprompts by initializing learned softprompt embedding(s)
        # and the model's input embeddings.
        # the model will always concatenate learned softprompt embeddings as prefix to the prompt/query after it's set
        # use config option to initialize embedding from existing vocab, or random
        self.n_soft_tokens = (
            config.method.n_soft_tokens
        )  # number of prefix tokens added to prompt, with learned embeddings

        s_wte = SoftEmbedding(
            model.gpt.get_input_embeddings(),
            n_tokens=self.n_soft_tokens,
            initialize_from_vocab=config.method.initialize_from_vocab,
        )

        model.gpt.set_input_embeddings(s_wte)

        # if all layers are frozen, freeze other non-learned-embedding params in addition to layer params
        if self.config.model.num_layers_unfrozen == 0:
            transformer_layers = model.gpt.transformer
            transformer_layers.wte.requires_grad_(False)
            transformer_layers.wpe.requires_grad_(False)

        return model
    
    def generate(self, input_ids, attention_mask=None, **kwargs):
        """Wraps hf's `generate` adding some specific method's defaults"""
        # pad for soft prompts (using same token as for padding)
        input_ids = torch.cat(
            [
                torch.full(
                    (input_ids.shape[0], self.n_soft_tokens), self.soft_dummy_token_id
                ).to(input_ids.device),
                input_ids,
            ],
            1,
        )
        input_ids = input_ids.to(self.accelerator.device)
        if attention_mask is not None:
            # extend for soft prompts (by extending mask at the end of tensor)
            attention_mask = torch.cat(
                [
                    attention_mask,
                    torch.full(
                        (attention_mask.shape[0], self.n_soft_tokens), 1
                    ).to(attention_mask.device),
                ],
                1,
            )
            attention_mask = attention_mask.to(self.accelerator.device)

        kwargs = dict(self.generate_kwargs, **kwargs)

        with torch.no_grad():
            return self.accelerator.unwrap_model(self.model).generate(
                input_ids=input_ids, attention_mask=attention_mask, use_cache=False, **kwargs
            ) # disable cache needed for softprompt compatibility

