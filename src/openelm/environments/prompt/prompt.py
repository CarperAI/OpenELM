from typing import Optional

import numpy as np
import torch
from langchain import PromptTemplate
from langchain.chains import LLMChain
from transformers import pipeline

from openelm.configs import PromptEnvConfig
from openelm.environments.base import BaseEnvironment, Genotype, Phenotype
from openelm.environments.prompt.utils import (
    AnimalPromptTask,
    AntonymPromptTask,
    COTPromptTask,
    ToyPromptTask,
)
from openelm.mutation_model import MutationModel


def get_positive_score(sentiment, mode="distilbert"):
    """Get the positive score from a sentiment analysis result."""
    if mode == "distilbert":
        return next(
            result["score"] for result in sentiment if result["label"] == "POSITIVE"
        )
    elif mode == "roberta":
        return next(
            result["score"] for result in sentiment if result["label"] == "LABEL_2"
        )
    else:
        raise ValueError(f"Invalid mode: {mode}")


def get_negative_score(sentiment, mode="distilbert"):
    """Get the negative score from a sentiment analysis result."""
    if mode == "distilbert":
        return next(
            result["score"] for result in sentiment if result["label"] == "NEGATIVE"
        )
    elif mode == "roberta":
        return next(
            result["score"] for result in sentiment if result["label"] == "LABEL_0"
        )
    else:
        raise ValueError(f"Invalid mode: {mode}")


def get_sentiment_score(sentiment, mode="distilbert"):
    return get_positive_score(sentiment, mode) - get_negative_score(sentiment, mode)


class PromptGenotype(Genotype):
    """
    Genotype wrapper for a LangChain template.

    This consists of a base format for all individuals, as well as
    individual-specific fields which will be evolved.
    Remaining fields will be filled in at evaluation time.

    Args:
        prompt (PromptTemplate): The base template for all individuals.
        fixed_inputs (dict[str, str], optional): Individual-specific fields to
        fill in. Defaults to None.
    """

    def __init__(
        self,
        prompt: PromptTemplate,
        fixed_inputs: Optional[dict[str, str]] = None,
        behavior_model=None,
    ):
        self.fixed_inputs = fixed_inputs
        if fixed_inputs:
            self.prompt = prompt.partial(**fixed_inputs)
        else:
            self.prompt = prompt
        self.result_obj = None
        if behavior_model:
            # assume sentiment analysis; can expand this later
            sentiment = behavior_model(self.__str__())
            self.behavior = (
                len(self.fixed_inputs["instruction_str"]),
                get_sentiment_score(
                    sentiment[0], mode=behavior_model.model.config.model_type
                ),
            )
        else:
            self.behavior = (len(self.fixed_inputs["instruction_str"]),)

    def __str__(self) -> str:
        return self.fixed_inputs["instruction_str"]
        # return self.prompt.template

    def format(self, **kwargs) -> str:
        return self.prompt.format(**kwargs)

    def evaluate(self, model, inputs):
        chain = LLMChain(llm=model.model, prompt=self.prompt)
        self.result_obj = {
            "prompt": self.format(**inputs),
            "output": chain(inputs),
        }
        return self.result_obj["output"]

    def to_phenotype(self) -> Optional[Phenotype]:
        return self.behavior


class PromptEvolution(BaseEnvironment[PromptGenotype]):
    """Evolves a LangChain prompt."""

    def __init__(
        self,
        config: PromptEnvConfig,
        mutation_model: MutationModel,
        fitness_model=None,
    ):
        self.config: PromptEnvConfig = config
        self.batch_size = self.config.batch_size
        self.mutation_model = mutation_model
        if fitness_model is None:
            self.fitness_model = mutation_model
        self.behavior_model = pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment",
            # model="distilbert-base-uncased-finetuned-sst-2-english",
            top_k=None,
            # return_all_scores=True,
        )

        self.task_name = self.config.task_name
        if self.task_name == "toy":
            self.genotype_ndim = 1
            self.genotype_space = np.array([[0], [250]])
            self.task = ToyPromptTask()
        elif self.task_name == "antonym":
            self.genotype_ndim = 2
            self.genotype_space = np.array([[0, -1], [250, 1]])
            self.task = AntonymPromptTask()
        elif self.task_name == "animal":
            self.genotype_ndim = 2
            self.genotype_space = np.array([[0, -1], [250, 1]])
            self.task = AnimalPromptTask()
        elif self.task_name == "cot":
            self.genotype_ndim = 2
            self.genotype_space = np.array([[0, -1], [250, 1]])
            self.task = COTPromptTask()
        else:
            raise ValueError(f"Unknown task: {self.task_name}")

        self.base_prompt = PromptTemplate(
            template=self.task.base_template, input_variables=self.task.input_variables
        )
        self.rng = np.random.default_rng(self.config.seed)

    def get_rng_state(self) -> Optional[np.random._generator.Generator]:
        return self.rng

    def set_rng_state(self, rng_state: Optional[np.random._generator.Generator]):
        self.rng = rng_state

    def random(self) -> list[PromptGenotype]:
        return [self.random_prompt() for _ in range(self.batch_size)]

    def random_prompt(self):
        if self.task_name == "toy":
            inputs = {
                "n_repetitions": str(self.rng.integers(10)),
                "instruction_str": self.task.instruction_str,
                "few_shot_examples": self.task.create_few_shot_examples(
                    self.task.instruction_str
                ),
            }
        elif (
            self.task_name == "antonym"
            or self.task_name == "animal"
            or self.task_name == "cot"
        ):
            few_shot_examples = self.task.create_few_shot_examples(
                n_examples=10,
            )
            generation_prompt = PromptTemplate(
                input_variables=["few_shot_examples"],
                template=self.task.generation_instruction,
            )
            generation_chain = LLMChain(
                llm=self.fitness_model.model, prompt=generation_prompt
            )
            result = generation_chain({"few_shot_examples": few_shot_examples})
            new_instruction_str = result["text"]

            # take only the first sentence
            new_instruction_str = (
                new_instruction_str.replace('"', "")
                .lstrip("0123456789. \n")
                .split(".")[0]
                + "."
            )

            inputs = {
                "instruction_str": new_instruction_str,
            }

        return PromptGenotype(
            prompt=self.base_prompt,
            fixed_inputs=inputs,
            behavior_model=self.behavior_model,
        )

    def mutate(self, genomes: list[PromptGenotype]) -> list[PromptGenotype]:
        prompts = [self.mutate_prompt(prompt) for prompt in genomes]
        return prompts

    def mutate_prompt(self, prompt):
        if self.task_name == "toy":
            # mutate the instruction string; note that we also need to change the few shot examples to match
            old_instruction_str = prompt.fixed_inputs["instruction_str"]
            result = self.rewrite_string(
                input_str=old_instruction_str,
                rewrite_instruction=np.random.choice(self.task.mutation_instructions),
                variable_name="instruction_str",
            )
            new_instruction_str = (
                result["text"].strip().split()[0]
            )  # take the first word

            inputs = {
                "n_repetitions": str(np.random.randint(10)),
                "instruction_str": new_instruction_str,
                "few_shot_examples": self.task.create_few_shot_examples(
                    new_instruction_str
                ),
            }
        elif (
            self.task_name == "antonym"
            or self.task_name == "animal"
            or self.task_name == "cot"
        ):
            if np.random.random() > 0.3:
                # rewrite the instruction string
                old_instruction_str = prompt.fixed_inputs["instruction_str"]
                result = self.rewrite_string(
                    input_str=old_instruction_str,
                    rewrite_instruction=np.random.choice(
                        self.task.mutation_instructions
                    ),
                    variable_name="instruction_str",
                )
                new_instruction_str = (
                    result["text"]
                    .replace('"', "")
                    .lstrip("0123456789. \n")
                    .split(".")[0]
                    + "."
                )  # take the first sentence
                inputs = {
                    "instruction_str": new_instruction_str,
                }
            else:
                # otherwise, just generate a random prompt
                return self.random_prompt()

        return PromptGenotype(
            prompt=self.base_prompt,
            fixed_inputs=inputs,
            behavior_model=self.behavior_model,
        )

    def rewrite_string(self, input_str, rewrite_instruction, variable_name):
        """
        Prompts an LLM to rewrite a string.

        Args:
            input_str: The string to rewrite.
            rewrite_instruction: String prompt template for the LLM
            variable_name: The name of the variable in the template to replace
            with input_str
        """
        rewrite_prompt = PromptTemplate(
            input_variables=[variable_name],
            template=rewrite_instruction,
        )
        rewrite_chain = LLMChain(llm=self.mutation_model.model, prompt=rewrite_prompt)
        result = rewrite_chain({variable_name: input_str})
        # if self.config.debug:
        #     print(
        #         f"-- Rewrite Instruction --\n{rewrite_instruction}\n-- Input --\n{input_str}\n-- Output --\n{result['text']}\n"
        #     )
        return result

    def fitness(self, x: PromptGenotype) -> float:
        if self.task_name == "toy":
            inputs = {
                "target": self.task.target,
            }
            result = x.evaluate(model=self.fitness_model, inputs=inputs)

            # fitness is number of times it generated the target word in a row
            count = 0
            for word in result["text"].strip().split():
                if word.lower() == self.task.target:
                    count += 1
                else:
                    break

            fitness = count
            if self.config.debug:
                print(
                    f"-- Prompt --\n{x.result_obj['prompt']}\n-- Fitness: {fitness} --\n-- Behavior: {x.to_phenotype()} --\n"
                )
        elif (
            self.task_name == "antonym"
            or self.task_name == "animal"
            or self.task_name == "cot"
        ):
            fitnesses = []
            eval_template = PromptTemplate(
                input_variables=["instruction_str", "input_str", "output_str"],
                template=self.task.evaluation_instruction,
            )
            inputs, outputs = self.task.get_random_data(
                n_examples=self.config.evals_per_prompt
            )
            for input_str, output_str in zip(inputs, outputs):
                fitnesses.append(
                    self.evaluate_template(
                        eval_template,
                        x.fixed_inputs["instruction_str"],
                        input_str,
                        output_str,
                    )
                )
            fitness = np.mean(fitnesses)
            if self.config.debug:
                print(
                    f"-- instruction_str --\n{x.fixed_inputs['instruction_str']}\n-- Fitness: {fitness} --\n-- Behavior: {x.to_phenotype()} --\n"
                )
        elif self.task_name == "imagegen":
            # fitness_prompt = PromptTemplate(
            #     input_variables=["program_str", "instruction_str"],
            #     template=self.task.fitness_template,
            # )
            pass

        return fitness

    def evaluate_template(self, eval_template, instruction_str, input_str, output_str):
        """
        Evaluates a template on the log likelihood of the output_str, given the
        instruction_str and input_str.

        Args:
            eval_template: The template to evaluate.
            instruction_str: The instruction string.
            input_str: The input string.
            output_str: The output string.

        Returns:
            The log likelihood of the tokens in the output string, given the
            instruction and input strings.
        """
        model = self.fitness_model.model.model
        tokenizer = self.fitness_model.model.tokenizer

        partial_template = eval_template.partial(instruction_str=instruction_str)
        filled_prompt = partial_template.format(
            input_str=input_str, output_str=output_str
        )
        # hack; replace the output string to figure out which token numbers
        # correspond to the output (see APE)
        reference_prompt = partial_template.format(input_str=input_str, output_str="~")

        tokens_filled = tokenizer.encode(filled_prompt, return_tensors="pt")
        tokens_reference = tokenizer.encode(reference_prompt, return_tensors="pt")

        # We label only the tokens of interest, and mask otherwise (set to -100)
        # This assumes there's only one section in the middle that we're interested in
        # forward alignment; mask duplicate tokens starting from beginning
        labels = tokens_filled.clone()
        for i, (t1, t2) in enumerate(zip(tokens_filled[0], tokens_reference[0])):
            if t1 == t2:
                labels[0, i] = -100 * torch.ones_like(labels[0, i])
            else:
                break

        # backward alignment
        for i, (t1, t2) in enumerate(
            zip(torch.flip(tokens_filled[0], [0]), torch.flip(tokens_reference[0], [0]))
        ):
            if t1 == t2:
                labels[0, -i - 1] = -100 * torch.ones_like(
                    labels[0, -i - 1]
                )  # adjust index for reversed
            else:
                break

        outputs = model(tokens_filled.cuda(), labels=labels.cuda())

        # self.print_labels(tokens_filled, tokens_reference, labels, tokenizer)
        return -outputs.loss.item()

    def print_labels(self, tokens_filled, tokens_reference, labels, tokenizer):
        from itertools import zip_longest

        print(
            f"{'Label':<10}{'Token Filled':<20}{'Token ID':<10}{'Token Reference':<20}{'Token ID':<10}"
        )

        for tf, tr, label in zip_longest(
            tokens_filled[0], tokens_reference[0], labels[0]
        ):
            decoded_tf, decoded_tr = " ", " "
            if tf is not None:
                decoded_tf = tokenizer.decode(
                    [tf]
                )  # Wrap tf in a list because .decode() expects a list
            if tr is not None:
                decoded_tr = tokenizer.decode([tr])  # Same for tr
            if label is None:
                label = ""
            if tr is None:
                tr = ""
            if tf is None:
                tf = ""
            print(f"{label:<10}{decoded_tf:<20}{tf:<10}{decoded_tr:<20}{tr:<10}")
