[![DOI](https://zenodo.org/badge/532259603.svg)](https://zenodo.org/badge/latestdoi/532259603)
# OpenELM

OpenELM is an open-source library by CarperAI, designed to enable evolutionary search with language models in both code and natural language.

The OpenELM project has the following goals:
1. Release an open-source version of ELM with its associated diff models.
2. Integrate with both open-source language models (run locally or on Colab) and with closed models via paid APIs, such as the OpenAI API.
We want to support users with many different compute profiles!
3. Provide a simple interface to a range of example environments for evolutionary search, to let users adapt these easily for their domain.
4. Demonstrate the potential of evolution with LLMs.

# Install
`pip install openelm`

To use the sodarace environment, you must first `pip install swig`.

Then:

`pip install openelm[sodaracer]`

See the pyproject.toml for further install options.

# Features

### LLM integration with evolutionary algorithms
OpenELM supports the quality-diversity algorithms MAP-Elites, CVT-MAP-Elites, and Deep Grid MAP-Elites, as well as a simple genetic algorithm baseline.

### Evolutionary operators
OpenELM supports:
1. Prompt-based mutation with instruct models
2. Diff models (specialised for code)
3. Crossover with language models

### LLM support, efficiency, and safety
OpenELM’s language models are instantiated as Langchain classes by default, which means that OpenELM can support practically any existing LLM API, as well as models run on your local GPU via HuggingFace Transformers.

We also provide optional Nvidia Triton Inference Server support, intended for use cases where low latency on 8 or more GPUs is important. Finally, for code generation domains, we provide a sandbox environment, consisting of a container server backed with gVisor (a container runtime that introduces an additional barrier between the host and the container) as well as a heuristic-based safety guard.

### Baseline environments
1. **Sodarace.** A 2D physics-based simulation of robots moving across a variety of terrains. These robots are created by Python programs generated from an LLM.
2. **Image Generation.** OpenELM can evolve over generated images by generating code that returns NumPy arrays containing the images. This serves as a simple test environment for code generation
3. **Programming Puzzles.** OpenELM can be used to generate diverse solutions to programming puzzles. This environment supports co-evolution of both the problem and the solution at the same time.
4. **Prompts.** OpenELM contains a generic environment suitable for evolving prompts for language models, customizable with Langchain templates to the desired domain.
5. We also include a **poetry** environment, demonstrating the use of LLMs to evaluate both the quality and diversity of generated creative writing text, as described in a recent CarperAI blog post on Quality-Diversity with AI Feedback (QDAIF).

## Architecture
Roughly, ELM consists of a pipeline of different components:
1. The `Environment` class. This class defines the mechanics of how to initialize members of the population, mutate them with the desired operator, and how to measure the fitness (and diversity) of individuals.
2. The `MAPElites` class. This class describes how the evolutionary algorithm works, and can be viewed as a wrapper around the environment defining the selection algorithm for generated individuals.
3. The `MutationModel` class, which is responsible for running the LLM to actually generate new individuals. This functions as a wrapper around the LangChain API. The environment is expected to call the `MutationModel` when a new individual is needed.
4. The `ELM` class calls the `MAPElites` algorithm class and runs the search.

All options for these classes are defined in `configs.py`, via dataclasses which are registered as a `hydra` config, and can be overriden via the command line when running one of the example scripts such as `run_elm.py`.

## Running ELM
`python run_elm.py` will start an ELM evolutionary search using the defaults listed in `configs.py`. These can be overriden via the command line. For example, you can use `run_elm.py env=image_evolution` to run the Image Evolution environment.

## Sandbox
To use the code execution sandbox, see the [sandboxing readme](https://github.com/CarperAI/OpenELM/blob/main/src/openelm/sandbox/README.md) for instructions to set it up in a Docker container with the gVisor runtime.

## Triton
We also have code available to run models in Nvidia's Triton Inference Server. See the [Triton Readme](https://github.com/CarperAI/OpenELM/blob/main/src/openelm/codegen/triton_utils/readme.md) to get started

# Contributing
If you'd like to contribute or have questions, go to the #openelm channel on the [CarperAI discord](https://discord.gg/canadagoose)!
