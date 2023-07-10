import json
from typing import Optional

import numpy as np
from langchain.schema import HumanMessage

from openelm.configs import QDEnvConfig
from openelm.environments.base import BaseEnvironment
from openelm.environments.prompt.prompt import PromptGenotype
from openelm.mutation_model import MutationModel, get_model


class PoetryGenotype(PromptGenotype):
    """
    Evolve poems with QDAIF.

    This consists of a base format for all individuals, as well as individual-specific fields which will be evolved.
    Remaining fields will be filled in at evaluation time.

    Args:
        prompt (PromptTemplate): The base template for all individuals.
        fixed_inputs (dict[str, str], optional): Individual-specific fields to fill in. Defaults to None.
    """

    def __init__(
        self, poem: str, genre: Optional[str] = None, tone: Optional[str] = None
    ):
        self.poem = poem
        self.genre = genre
        self.tone = tone
        self.genres_to = {
            num: typ
            for num, typ in enumerate(["haiku", "sonnet", "ballad", "limerick", "hymn"])
        }
        self.tones_to = {
            num: typ
            for num, typ in enumerate(
                ["happy", "dark", "mysterious", "romantic", "reflective"]
            )
        }
        # typology -> label
        self.genres_from = {typ: num for num, typ in self.genres_to.items()}
        self.tones_from = {typ: num for num, typ in self.tones_to.items()}

    def __str__(self) -> str:
        return self.poem

    def evaluate(self, model) -> float:
        quality_str = f"""{self.poem}
Rate the quality of the above poem on a scale from 1 to 10. Answer in JSON with the key 'quality'.
"""
        diversity_str = f"""{self.poem}
What genre is this poem closest to from the following list: ["haiku", "sonnet", "ballad", "limerick", "hymn"]
What tone is this poem closest to from the following list: ["happy", "dark", "mysterious", "romantic", "reflective"]

Respond in JSON with the keys "genre" and "tone".
"""
        quality_result = model([HumanMessage(content=quality_str)]).content
        diversity_result = model([HumanMessage(content=diversity_str)]).content
        try:
            self.quality = json.loads(quality_result)["quality"]
            self.genre = json.loads(diversity_result)["genre"]
            self.tone = json.loads(diversity_result)["tone"]
            return float(self.quality)
        except Exception:
            return -np.inf

    def to_phenotype(self) -> Optional[np.ndarray]:
        if isinstance(self.genre, str) and isinstance(self.tone, str):
            genre_idx = self.genres_from[self.genre]
            tone_idx = self.tones_from[self.tone]
            return np.array([genre_idx, tone_idx])
        return None


class PoetryEvolution(BaseEnvironment[PoetryGenotype]):
    """Evolves a LangChain prompt."""

    def __init__(
        self,
        config: QDEnvConfig,
        mutation_model: MutationModel,
    ):
        self.config: QDEnvConfig = config
        self.batch_size = self.config.batch_size
        self.genotype_space = np.array(self.config.behavior_space).T
        self.genotype_ndim = self.genotype_space.shape[1]
        self.mutation_model = get_model(mutation_model.config)
        self.eval_model = self.mutation_model
        del mutation_model

        self.seed_str = """
Fields of green waves under
The sky grey, rain on the soft
Winds whisper; normality reigns.
"""
        self.seed = PoetryGenotype(self.seed_str, "haiku", "reflective")
        self.genres_to = {
            num: typ
            for num, typ in enumerate(["haiku", "sonnet", "ballad", "limerick", "hymn"])
        }
        self.tones_to = {
            num: typ
            for num, typ in enumerate(
                ["happy", "dark", "mysterious", "romantic", "reflective"]
            )
        }
        # typology -> label
        self.genres_from = {typ: num for num, typ in self.genres_to.items()}
        self.tones_from = {typ: num for num, typ in self.tones_to.items()}
        self.rng = np.random.default_rng(self.config.seed)

    def get_rng_state(self) -> Optional[np.random._generator.Generator]:
        return self.rng

    def set_rng_state(self, rng_state: Optional[np.random._generator.Generator]):
        self.rng = rng_state

    def construct_prompt(self, poem: Optional[PoetryGenotype] = None) -> dict[str, str]:
        target_genre = self.genres_to[self.rng.choice(list(self.genres_to.keys()))]
        target_tone = self.tones_to[self.rng.choice(list(self.tones_to.keys()))]
        if poem is None:
            instruction_str = f"Write a {target_tone} {target_genre} poem of very high, award winning quality.\n"
            prompt_str = f"{self.seed.poem}\n{instruction_str}"
            # prompt_str = "Write a poem of very high, award winning quality.\n"
        else:
            instruction_str = f"Translate this {poem.genre} poem into a {target_tone} {target_genre} poem of very high, award winning quality.\n"
            prompt_str = f"{poem.poem}\n{instruction_str}"
        return {
            "prompt": prompt_str,
            "template": "",
            "target_genre": target_genre,
            "target_tone": target_tone,
        }

    def random(self) -> list[PoetryGenotype]:
        # Mutate seed, and pick random target genre and poem.
        prompt_list = [self.construct_prompt() for _ in range(self.config.batch_size)]
        results = []
        for prompt in prompt_list:
            results.append(
                self.mutation_model(HumanMessage(content=prompt["prompt"]).content)
            )
        return [PoetryGenotype(poem=c) for c in results]

    def mutate(self, genomes: list[PoetryGenotype]) -> list[PoetryGenotype]:
        prompt_list: list[dict[str, str]] = list(map(self.construct_prompt, genomes))
        results = []
        for prompt in prompt_list:
            results.append(
                self.mutation_model(HumanMessage(content=prompt["prompt"]).content)
            )
        return [PoetryGenotype(poem=c) for c in results]

    def fitness(self, x: PoetryGenotype) -> float:
        return x.evaluate(self.eval_model)
