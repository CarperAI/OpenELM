from dataclasses import dataclass

import numpy as np


def get_image_target(name: str) -> np.ndarray:
    if name == "circle":
        target = np.zeros((32, 32, 3))
        for y in range(32):
            for x in range(32):
                if (y - 16) ** 2 + (x - 16) ** 2 <= 100:  # a radius-10 circle
                    target[y, x] = np.array([255, 255, 0])
    else:
        raise NotImplementedError(f"Image target {name} not implemented")
    return target


IMAGE_SEED: str = """
def draw():
\tpic = np.zeros((32, 32, 3))
\tfor x in range(2, 30):
\t\tfor y in range(2, 30):
\t\t\tpic[x, y] = np.array([0, 0, 255])
\treturn pic
"""

NULL_SEED: str = ""


@dataclass
class ToyPromptTask:
    base_template = "{few_shot_examples}\n{instruction_str} the word {target} {n_repetitions} times:"
    input_variables = [
        "few_shot_examples",
        "target",
        "instruction_str",
        "n_repetitions",
    ]

    target = "hello"
    instruction_str = "Repeat"

    mutation_instruction = """Q: What is a synonym for happy?
A: Cheerful

Q: What is a synonym for sad?
A: Melancholy

Q: What is a synonym for alter?
A: Adjust

Q: What is a synonym for finish?
A: End

Q: What is a synonym for {instruction_str}?
A:"""

    def create_few_shot_examples(self, instruction_str):
        return f"""{instruction_str} the word {self.target} 2 times: {self.target} {self.target}
{instruction_str} the word {self.target} 3 times: {self.target} {self.target} {self.target}
{instruction_str} the word {self.target} 4 times: {self.target} {self.target} {self.target} {self.target}"""


@dataclass
class AntonymPromptTask:
    base_template = """Instruction: {instruction_str}
Input: {input_str}
Output: {output_str}"""

    input_variables = [
        "instruction_str",
        "input_str",
        "output_str",
    ]

    words = [
        "sane",
        "direct",
        "informally",
        "unpopular",
        "subtractive",
        "nonresidential",
        "inexact",
        "uptown",
        "incomparable",
        "powerful",
        "gaseous",
        "evenly",
        "formality",
        "deliberately",
        "off",
    ]
    antonyms = [
        "insane",
        "indirect",
        "formally",
        "popular",
        "additive",
        "residential",
        "exact",
        "downtown",
        "comparable",
        "powerless",
        "solid",
        "unevenly",
        "informality",
        "accidentally",
        "on",
    ]

    generation_instruction = """I gave a friend an instruction. Based on the instruction they produced the following input-output pairs:\n{few_shot_examples}\nThe instruction was to """

    def create_few_shot_examples(self, input_strings, output_strings):
        few_shot_examples = ""
        for input_str, output_str in zip(input_strings, output_strings):
            few_shot_examples += f"Input: {input_str}\nOutput: {output_str}\n\n"

        return few_shot_examples
