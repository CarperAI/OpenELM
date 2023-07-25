[![DOI](https://zenodo.org/badge/532259603.svg)](https://zenodo.org/badge/latestdoi/532259603)

# LMX generation pseudocode

LMX specific config values:
```
MODEL_USED: model for generation
CLASSIFIER_MODEL: model for AI Feedback
MUTATION_METHOD: use lmx_near or replace to mutate prompts
SOLUTION_INIT_METHOD: method to create an initial population for the map, supports: generated or seed 
```

Necessary config arguments for the generation domain (default values are set to reproduce the short story experiments):
Below we will provide example config arguments for our three experiment domains: short story, movie review and opinion piece
```
    "prompt_template": string used in the prompt to create a repeating few-shot pattern
    "prompt_pool_path": Path to a file with example prompts for the choosen domain 
    "gen_max_len": max token limit for output from API
    "instruction_prompt": used in the diversity AI feedback prompt to evaluate the attribute of generations along an axis
    "quality_feedback_prompt": used in the quality AI feedback prompt to measure how closely generation aligns with desired qualities
```

Pseudocode:

```python
# given
generator = LLM
generator_prompt = "Here is a random example of a review for the movie \"Die Hard\":"
few_shot_prompt = f"{generator_prompt} {gen_movie_review}\n###\n" # repeated 3 times

# init map with first solutions -> generated from scratch
pool_movie_reviews = generate_movie_review_with_zero_shot(generator_prompt)
first_individuals = randomly_sample_few_shot_from_pool_and_generate_with_few_shot(pool_movie_reviews) # store items of few-shot prompt, along with gen_movie_review (genotype), and sentiment measure (phenotype)
evaluate_measure_and_fitness(first_individuals)

# evolve solutions from archive (replace operator)
individual = random_elite_from_map()
few_shot_items_list = individual.genotype # mutate this
few_shot_idx_to_replace = random_idx(few_shot_items_list)
few_shot_items_list[few_shot_idx_to_replace] = random_item(pool_movie_reviews)
new_individual = generate_with_few_shot(few_shot_items_list)
evaluate_measure_and_fitness(new_individual.gen_movie_review)

# more optimal pool
if added_to_archive_successfully(new_individual):
    pool_movie_reviews.append(new_individual.gen_movie_review)
```

Example config arguments for reproducing LMX experiments:
Short Story:
```python
"prompt_template": 'Here is a random example of a fantasy story about a suspicious spy and a rich politician:'
"prompt_pool_path": "src/openelm/environments/lmx_seed_pools/short_story_seed_pool.txt"
"gen_max_len": 100
"instruction_prompt": "You are given an input text of a short story involving multiple characters. Determine if the characters in this story primarily experience a conflict, or have a friendship. Write \"conflict\" if a conflict between characters is present in the story, otherwise answer \"friendship\" if a friendship between characters is present in the story."
"quality_feedback_prompt": "Determine if the input text contains a high-quality short story containing two characters, a suspicious spy, and a rich politician. For example, a high-quality short story would have good flow, interesting plot, and not repeat similar sentences or undesired items such as titles and URLs. Answer \"yes\" if the input contains a high-quality short story about a suspicious spy and a rich politician, otherwise answer \"no\"."
```

Opinion Piece:
```python
"prompt_template": 'Here is a random opinion piece about eating vegetables and plant-based foods:'
"prompt_pool_path": "src/openelm/environments/lmx_seed_pools/opinion_piece_seed_pool.txt"
"gen_max_len": 50
"instruction_prompt": "Determine the sentiment of the given opinion on eating vegetables and plant-based foods (from the input text) by writing \"positive\" or \"negative\" in the output."
"quality_feedback_prompt": "Determine whether or not the input text is closely related to the following topic: \"someone talking about whether or not they like to eat vegetables and plant-based foods as well as an explanation for their preferences\". Answer \"yes\" if it is about the topic, or \"no\" if it is not about the topic."
```

AI feedback for a single dimension is shown as an example in `__post_init__` of `LMXGenerationEnvConfig`:
```python
self.ai_feedback_entries = { # entries to setup ai feedback.
    "sentiment": {
        "answer_space": [
            f"{extra_prefix}positive",
            f"{extra_prefix}negative",
        ],
        "feedback_prompt_template": f"### Instruction:\n{self.instruction_prompt}{extra_suffix}\n\n### Input:{{genotype}}\n\n### Response:"
    },
}
```

If we want to extend the evaluation to multiple AI feedback measures, we can set it like this (either adding extra config options for instruction_prompt, or writing their own prompts directly):
```python
self.ai_feedback_entries = { # entries to setup ai feedback.
    "feedback_1": {
        "answer_space": [
            f"{extra_prefix}answer_1",
            f"{extra_prefix}answer_2",
        ],
        "feedback_prompt_template": f"### Instruction:\n{self.instruction_prompt_1}{extra_suffix}\n\n### Input:{{genotype}}\n\n### Response:"
    },
    "feedback_2": {
        "answer_space": [
            f"{extra_prefix}answer_3",
            f"{extra_prefix}answer_4",
        ],
        "feedback_prompt_template": f"### Instruction:\n{self.instruction_prompt_2}{extra_suffix}\n\n### Input:{{genotype}}\n\n### Response:"
    },
}
```

For access to the AI feedback model used in the experiments `luminous-supreme-qdaif`, please send an email, with [QDAIF] in the subject line, to [support@aleph-alpha.com](mailto:support@aleph-alpha.com)

# OpenELM

This repository is a replication of [Evolution Through Large Models](https://arxiv.org/abs/2206.08896), a recent paper from OpenAI exploring the links between large language models (LLMs) and evolutionary computing, particularly focused on code generation.

LLMs trained on datasets of code, such as OpenAI’s Codex, have shown good results in automated code generation. However, in cases where we are interested in a class of programs which are rarely found in the training distribution,
evolutionary algorithms provide a way to generate code by making mutations to known, or "seed" programs. The ELM approach shows that an LLM trained on code can suggest intelligent mutations for genetic programming (GP) algorithms. Genetic algorithms explore the search space with random perturbations, but typically need to be highly customised with domain knowledge to allow them to make desirable changes — LLMs provide a way of encoding this domain knowledge and guiding the genetic algorithm towards intelligent exploration of the search space.

This project aims to replicate the ELM paper in the original [Sodarace](https://doi.org/10.1162/ARTL_a_00185) environment, before applying the technique to more complex code generation problems.

For more details, see our full research proposal at https://carperai.notion.site/ELM-e8f37b2649944259b1abf9ccaa4edae2. The release blog post: https://carper.ai/openelm-release.

# Architecture
Roughly, ELM consists of a pipeline of different components:
```html
+-------------+                     +-------------+
|  MapElites  | <-----------------> | Environment |
+------+------+                     +------+------+
       |                                   ^
       | collect samples                   |
       v                                   v
+------+---------+     finetune    +-------+--------+    mutate and execute   +----------------+
| Conditional RL | --------------> | Language model | <---------------------> | Sandbox server |
+----------------+                 +----------------+                         +----------------+
```
The basic workflow consists of generate -> evaluate -> finetune. We currently have implemented everything except the conditional RL part.

# Running ELM
Currently, we can run the MAP-Elites algorithm on [a few environments](https://github.com/CarperAI/OpenELM/blob/main/src/openelm/environments/environments.py), apply [prompt mutations](https://github.com/CarperAI/OpenELM/blob/main/src/openelm/diff_model.py), and connect with an optional [sandbox server](https://github.com/CarperAI/OpenELM/blob/main/src/openelm/sandbox).

## Sandbox
To use the code execution sandbox, see the [sandboxing readme](https://github.com/CarperAI/OpenELM/blob/main/src/openelm/sandbox/README.md) for instructions to set it up in a docker container. But for quick testing purpose, one may try the following:
```bash
cd elm/sandbox/server
export FLASK_APP=index.py
flask run
```
## Running MAP-Elites
We have a few toy environments implemented as well as the Sodarace environment in the ELM paper. The `run_elm.py` file gives an example of how to run an ELM loop with MAP-Elites using the Sodarace environment.

## Triton
We also have code to run models in Nvidia's Triton Inference Server. See the [Triton Readme](https://github.com/CarperAI/OpenELM/blob/main/src/openelm/codegen/triton_utils/readme.md) to get started
