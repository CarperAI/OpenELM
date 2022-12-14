import copy
from time import time
from typing import Tuple

import ray
import torch
import torch.nn.functional as F
import wandb
from torch import nn
from torchtyping import TensorType
from trlx.data.configs import TRLConfig
from trlx.model import register_model
from trlx.model.accelerate_ppo_model import AcceleratePPOModel
from trlx.model.nn.ppo_models import CausalLMHydraWithValueHead


class SoftEmbedding(nn.Module):
    def __init__(
        self,
        wte: nn.Embedding,
        n_tokens: int = 10,
        random_range: float = 0.5,
        initialize_from_vocab: bool = True,
    ):
        """
        appends learned embedding as prefix

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
        self.padding_token_id = 50256  # used when input tensors are prefix padded
        self.learned_embedding = (
            nn.parameter.Parameter(  # dim: (n_tokens, embedding_dim)
                self.initialize_embedding(
                    wte, n_tokens, random_range, initialize_from_vocab
                )
            )
        )
        self.init_embedding = copy.deepcopy(self.learned_embedding)

    def initialize_embedding(
        self,
        wte: nn.Embedding,
        n_tokens: int = 10,
        random_range: float = 0.5,
        initialize_from_vocab: bool = True,
    ):
        """
        initializes learned embedding

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
        """
        run forward pass

        Args:
            tokens (torch.long): input tokens before encoding
        Returns:
            seq_embedding (torch.float): encoding of text concatenated with learned task specifc embedding
        """
        prompt_tokens = tokens[
            :, self.n_tokens :
        ]  # dim: (batch_size, seq_length) - without soft prompt padding indices
        if self.padding_token_id in prompt_tokens[:, 0]:  # padding is applied as prefix
            seq_embedding = self.wte(
                tokens
            )  # dim: (batch_size, seq_length, embedding_dim)
            padding_tensor = torch.tensor([self.padding_token_id]).to(
                seq_embedding.device
            )

            # index in each sequence in tokens just after last prefix padding
            # this would be where the (first) soft prompt embedding should be set
            first_prompt_indices = (
                (prompt_tokens == padding_tensor).int().argmin(axis=1)
            )

            # for asserting that the first main sequence token embedding isn't modified by accident
            first_prompt_indices_full_seq = (
                (tokens == padding_tensor).int().argmin(axis=1)
            )
            first_item_idx = first_prompt_indices_full_seq[0]
            main_embedding_before_soft_prompt_assign = seq_embedding[0, first_item_idx]

            # for each batch sequence, replace embeddings at soft prompt indices with correct soft embeddings
            for batch_idx, first_prompt_idx in enumerate(first_prompt_indices.tolist()):
                # indices for assigning soft embeddings
                start = first_prompt_idx
                end = first_prompt_idx + self.n_tokens

                seq_embedding[batch_idx, start:end] = self.learned_embedding

                # debug only
                if batch_idx == 0:
                    first_main_embedding_after_soft_embedding_assign = seq_embedding[
                        0, first_item_idx
                    ]
                    assert torch.equal(
                        main_embedding_before_soft_prompt_assign,
                        first_main_embedding_after_soft_embedding_assign,
                    ), "Error: soft prompt overwrote main prompt embeddings"
        else:
            input_embedding = self.wte(prompt_tokens)
            learned_embedding = self.learned_embedding.repeat(
                input_embedding.size(0), 1, 1
            )
            seq_embedding = torch.cat([learned_embedding, input_embedding], 1)

        assert (
            seq_embedding.shape[1] == prompt_tokens.shape[1] + self.n_tokens
        ), "Number of token embeddings with soft prompts should be number of prompt tokens + number of soft tokens"

        return seq_embedding


@register_model
class AcceleratePPOSoftpromptModel(AcceleratePPOModel):
    def __init__(self, config, train_mode=True):
        # account for extra prefix tokens
        config.method.gen_kwargs["max_length"] += config.method.n_soft_tokens
        config.method.gen_kwargs["min_length"] += config.method.n_soft_tokens

        super().__init__(config)

        assert (
            config.method.n_soft_tokens > 0
        ), "Number of soft prompt tokens should be >=1"

        self.soft_dummy_token_id = 50256  # dummy token for padding soft prompts
        self.measure_soft_embedding_drift = config.method.measure_soft_embedding_drift

    def get_arch(self, config: TRLConfig):
        """
        Load model, and set Soft Prompt module for input embeddings
        """
        model = CausalLMHydraWithValueHead(
            config.model.model_path, config.model.num_layers_unfrozen
        )

        # if all layers are frozen, freeze all params. Softprompt will still be tuned
        if config.model.num_layers_unfrozen == 0:
            model.requires_grad_(False)

            if config.method.tune_v_head:
                model.v_head.requires_grad_(True)  # unfreeze value head

        # here, we setup softprompts by initializing learned softprompt embedding(s)
        # and the model's input embeddings.
        # the model will always concatenate learned softprompt embeddings as prefix to the prompt/query after it's set
        # use config option to initialize embedding from existing vocab, or random
        self.n_soft_tokens = (
            config.method.n_soft_tokens
        )  # number of prefix tokens added to prompt, with learned embeddings

        s_wte = SoftEmbedding(
            model.base_model.get_input_embeddings(),
            n_tokens=self.n_soft_tokens,
            initialize_from_vocab=config.method.initialize_from_vocab,
        )

        model.base_model.set_input_embeddings(s_wte)

        return model

    def generate(
        self,
        input_ids: TensorType["batch_size", "seq_length"],
        attention_mask: TensorType["batch_size", "seq_length"] = None,
        **kwargs,
    ):
        """
        Wraps hf's `generate` adding some specific method's defaults

        Modified to handle indices containing soft prompts
        """
        # pad for soft prompt indices (using same token as for padding)
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
            # extend for soft prompt indices (by extending mask at the end of tensor)
            attention_mask = torch.cat(
                [
                    attention_mask,
                    torch.full((attention_mask.shape[0], self.n_soft_tokens), 1).to(
                        attention_mask.device
                    ),
                ],
                1,
            )
            attention_mask = attention_mask.to(self.accelerator.device)

        kwargs = dict(self.generate_kwargs, **kwargs)

        with torch.no_grad():
            return self.accelerator.unwrap_model(self.model).generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                **kwargs,
            )  # disable cache needed for softprompt compatibility

    def get_model_inputs(
        self,
        query_tensors: TensorType["batch_size", "query_size"],
        response_tensors: TensorType["batch_size", "response_size"],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Used in orchestrator and loss calculation, to compute logprobs and values

        Modified to handle indices containing soft prompts
        """
        tokens = torch.cat((query_tensors, response_tensors), dim=1)
        attention_mask = (
            tokens.not_equal(self.tokenizer.pad_token_id).long().to(tokens.device)
        )
        # to handle extra softprompts, set attention at softprompt indices
        first_non_pad_indices = torch.argmax(attention_mask, dim=1)
        for batch_idx, first_non_pad_idx in enumerate(first_non_pad_indices.tolist()):
            start = first_non_pad_idx - self.n_soft_tokens
            end = first_non_pad_idx
            attention_mask[batch_idx, start:end] = 1.0

        # For a proper positional encoding in case of left padding
        position_ids = attention_mask.cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask.eq(0), 0)
        return tokens, attention_mask, position_ids

    def evaluate(self):
        """
        Samples model on `eval_prompts`, logs stats with `reward_fn` or `metric_fn` if provided

        Modified to support plotting of metrics involving soft prompts
        """
        stats = {}
        all_samples = []
        generate_time = time()
        for prompts in self.eval_dataloader:
            if isinstance(prompts, torch.Tensor):
                samples = self.generate(prompts)
            else:
                samples = self.generate(**prompts)

            if isinstance(samples, tuple):
                samples, *_ = samples

            pad_token = self.tokenizer.eos_token_id if self.tokenizer else 0
            all_samples.append(
                F.pad(
                    samples,
                    (0, self.max_length - samples.shape[1]),
                    value=pad_token,
                )
            )
        stats["generate_time"] = time() - generate_time

        samples = self.accelerator.gather(torch.vstack(all_samples))

        if self.accelerator.is_main_process:
            if self.tokenizer:
                samples = self.tokenizer.batch_decode(samples, skip_special_tokens=True)

            if isinstance(samples[0], str):
                columns_data = [samples]
            else:
                columns_data = [samples.tolist()]
            columns = ["samples"]

            # in online setting, compute the reward for validation
            if self.reward_fn:
                rewards = torch.as_tensor(self.reward_fn(samples), dtype=torch.float)
                mean_reward = rewards.mean()
                columns.append("reward")
                columns_data.append(rewards)
                stats["mean_reward"] = mean_reward
                print(f"{mean_reward=}")

            # log Euclidean distance between init and current Soft Prompt embedding parameters
            if self.measure_soft_embedding_drift:
                softprompt = self.model.base_model.get_input_embeddings()
                stats["softprompt_drift_dist"] = (
                    (softprompt.init_embedding - softprompt.learned_embedding)
                    .pow(2)
                    .sum(1)
                    .sqrt()
                    .mean()
                )

            # additionally log any other metrics
            if self.metric_fn:
                metric_time = time()
                metrics = self.metric_fn(samples)
                stats["metric_time"] = time() - metric_time

                mean_metrics = {
                    f"metrics/{k}": torch.as_tensor(xs).mean(-1)
                    for k, xs in metrics.items()
                }

                stats.update(mean_metrics)

                for metric, values in metrics.items():
                    columns.append(metric)
                    columns_data.append(values)

            rows = list(zip(*columns_data))
            print(rows[0])
            if not ray.is_initialized():
                stats["samples"] = wandb.Table(columns=columns, rows=rows)

        return stats
