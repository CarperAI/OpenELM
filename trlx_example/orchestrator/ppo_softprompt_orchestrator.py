from typing import Callable

import torch
from trlx.data.accelerate_base_datatypes import PromptBatch
from trlx.data.ppo_types import PPORLElement
from trlx.model import BaseRLModel
from trlx.orchestrator import register_orchestrator
from trlx.pipeline import BasePipeline
from trlx.utils import Clock
from trlx.utils.modeling import logprobs_from_logits
from trlx.orchestrator.ppo_orchestrator import PPOOrchestrator

@register_orchestrator
class PPOSoftpromptOrchestrator(PPOOrchestrator):
    def __init__(
        self,
        model: BaseRLModel,
        pipeline: BasePipeline,
        reward_fn: Callable,
        metric_fn: Callable = None,
        chunk_size: int = 512,
    ):
        super().__init__(model, pipeline, reward_fn, metric_fn, chunk_size)
        self.n_soft_tokens = model.model.gpt.get_input_embeddings().n_tokens
    
    def make_experience(self, num_rollouts: int = 1024, iter_count: int = 0):
        """
        Takes `num_rollouts` prompts from `pipeline`, samples model, computes KL againts a reference model appends PPOElements to model's `store`
        """
        ppo_rl_elements = []
        stats = {}
        clock = Clock()
        while len(ppo_rl_elements) < num_rollouts:
            # Get next batch in prompt dataset and refresh if exhausted
            try:
                batch: PromptBatch = next(self.pipeline_iterator)
            except StopIteration:
                self.pipeline_iterator = iter(self.pipeline_loader)
                batch = next(self.pipeline_iterator)

            samples = self.rl_model.generate(**batch)

            query_len = batch.input_ids.shape[1] + self.n_soft_tokens
            query_tensors = samples[:, : query_len]
            response_tensors = samples[:, query_len :] # ignore softprompt padding index tokens
            texts = self.rl_model.tokenizer.batch_decode(
                samples, skip_special_tokens=True
            )
            scores = torch.as_tensor(self.score(texts))

            # Precompute logprobs, values
            with torch.no_grad():
                logits, _, v = self.rl_model.model(samples)
                # TODO(dahoas): When hydra model works need to also support generation on hydra head
                if hasattr(self.rl_model.model, "frozen_head"):
                    ref_logits = self.rl_model.model.forward_hydra(
                        samples, return_dict=False
                    )
                else:
                    ref_logits, _, _ = self.ref_model(samples.cpu())

            ref_logits = ref_logits.to(self.rl_model.accelerator.device)
            logprobs = logprobs_from_logits(logits[:, :-1, :], samples[:, 1:])
            ref_logprobs = logprobs_from_logits(
                ref_logits[:, :-1, :], samples[:, 1:]
            )
            start = query_tensors.size()[1] - 1
            end = query_tensors.size()[1] + response_tensors.size()[1] - 1
            all_values = v[:, start:end]
            all_logprobs = logprobs[:, start:end]
            all_ref_logprobs = ref_logprobs[:, start:end]

            # Compute rewards
            kls = all_logprobs - all_ref_logprobs
            non_score_rewards = -self.rl_model.kl_ctl.value * kls
            all_rewards = non_score_rewards.clone()
            all_rewards[:, -1] += scores.to(self.rl_model.accelerator.device)

            query_tensors = query_tensors.cpu()
            response_tensors = response_tensors.cpu()
            all_logprobs = all_logprobs.cpu()
            all_values = all_values.cpu()
            all_rewards = all_rewards.cpu()

            exp_time = clock.tick()

            new_ppo_rl_elements = [
                PPORLElement(
                    query_tensor=query_tensors[i, :],
                    response_tensor=response_tensors[i, :],
                    logprobs=all_logprobs[i, :],
                    values=all_values[i, :],
                    rewards=all_rewards[i, :],
                )
                for i in range(query_tensors.size()[0])
            ]
            ppo_rl_elements += new_ppo_rl_elements

        stats = {"exp_time": exp_time}
        self.rl_model.accelerator.log(stats, step=iter_count)

        # Push samples and rewards to model's rollout storage
        self.rl_model.push_to_store(ppo_rl_elements)