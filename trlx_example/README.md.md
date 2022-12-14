# Soft Prompt Tuning for Stage 3 Experiments
Here, we have some example scripts and modules to run Soft Prompt tuning using [trlx](https://github.com/CarperAI/trlx).

Stage 3 focuses on training a language model (LM) (pretrained on the Sodaracer dataset in Stage 2) to generate a Sodaracer agent given a specific terrain, using Conditional RL with PPO.

A terrain embedding network (TEN) is used to create a conditional prefix embedding for prompting the language model. One of the TEN methods used is to represent the terrain as a tunable discrete code (i.e. Soft Prompt), which can steer the model to generate a Sodaracer to successfully traverse this terrain.

## Running examples using Soft Prompt Tuning
`ppo_softprompt_sentiment.py` is an example script for tuning a Soft Prompt to steer the LM towards generating positive movie reviews.

In addition to the base code from the example sentiment task script in trlx, the Soft Prompt-based model, and its supporting orchestrator are registered during import. A Soft Prompt config object is also initialized.

To run training, navigate to `trlx_example`, and run one of the example scripts (e.g. `ppo_softprompt_sentiment.py).

### Config Settings:
Besides the PPO-specific settings, we can experiment with some main config options:
- `num_layers_unfrozen` : set to 0 to freeze all the weights in the LM (except for the Soft Prompt embedding parameters)
- `n_soft_tokens` : determines the number of Soft Prompt tokens to initialize (more means higher total trainable parameter count)
- `initialize_from_vocab` : if `True`, use existing embeddings from vocab to initialize Soft Prompts, otherwise, initialize randomly
- `tune_v_head` : if `True`, unfreeze the value head during training

## Notes
- Tested so far using trlx on this [commit](https://github.com/CarperAI/trlx/tree/33deeb1a3534ee46555e40a70c64d12bcabd73db)
- In the config, the `seq_length` is manually edited to be equal to `max_length` + `n_soft_tokens`
- Caching in `generate` function is disabled as it currently breaks the forward pass with Soft Prompts
- Soft prompt embeddings in sequence are set between padding token embeddings, and main prompt embeddings

## Acknowledgements
Thanks to Kip for an [example implementation](https://github.com/kipgparker/soft-prompt-tuning) of a Soft Prompt module used in the experiments here.
