[![DOI](https://zenodo.org/badge/532259603.svg)](https://zenodo.org/badge/latestdoi/532259603)
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
