# ELM

This repository is a replication of [Evolution Through Large Models](https://arxiv.org/abs/2206.08896), a recent paper from OpenAI exploring the links between large language models (LLMs) and evolutionary computing, particularly focused on code generation.

LLMs trained on datasets of code, such as OpenAI’s Codex, have shown good results in automated code generation. However, in cases where we are interested in a class of programs which are rarely found in the training distribution,
evolutionary algorithms provide a way to generate code by making mutations to known, or "seed" programs. The ELM approach shows that an LLM trained on code can suggest intelligent mutations for genetic programming (GP) algorithms. Genetic algorithms explore the search space with random perturbations, but typically need to be highly customised with domain knowledge to allow them to make desirable changes — LLMs provide a way of encoding this domain knowledge and guiding the genetic algorithm towards intelligent exploration of the search space.

This project aims to replicate the ELM paper in the original [Sodarace](https://doi.org/10.1162/ARTL_a_00185) environment, before applying the technique to more complex code generation problems.

# Milestones & Progress

Weekly meetings are in the EleutherAI discord at 20:00 UTC on Fridays.

- [ ] Sodarace environment implemented
- [ ] Stage 1: Diff Models & MAP-Elites
  - [ ] Prompt Engineering on CodeGen
  - [ ] Train diff model
  - [ ] MAP-Elites implemented
- [ ] Stage 2: Train LLM on generated data
- [ ] Stage 3: Conditional generation with PPO
