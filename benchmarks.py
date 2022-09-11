import itertools
import re

import torch
from transformers import CodeGenForCausalLM, GPT2TokenizerFast


def create_tokenizer():
    t = GPT2TokenizerFast.from_pretrained('gpt2')
    t.max_model_input_sizes['gpt2'] = 1e20
    return t


def create_model(ckpt, fp16=True):
    if fp16:
        return CodeGenForCausalLM.from_pretrained(ckpt, revision='float16',
                                                  torch_dtype=torch.float16,
                                                  low_cpu_mem_usage=True)
    else:
        return CodeGenForCausalLM.from_pretrained(ckpt)


def include_whitespace(t, n_min=2, n_max=20, as_special_tokens=False):
    t.add_tokens([' ' * n for n in reversed(range(n_min, n_max))],
                 special_tokens=as_special_tokens)
    return t


def include_tabs(t, n_min=2, n_max=20, as_special_tokens=False):
    t.add_tokens(['\t' * n for n in reversed(range(n_min, n_max))],
                 special_tokens=as_special_tokens)
    return t


def create_custom_gpt2_tokenizer():
    t = create_tokenizer()
    t = include_whitespace(t=t, n_min=2, n_max=32, as_special_tokens=False)
    t = include_tabs(t=t, n_min=2, n_max=10, as_special_tokens=False)
    return t


def truncate(completion):

    def find_re(string, pattern, start_pos):
        m = pattern.search(string, start_pos)
        return m.start() if m else -1

    terminals = [
        re.compile(r, re.MULTILINE)
        for r in
        [
            '^#',
            re.escape('<|endoftext|>'),
            "^'''",
            '^"""',
            '\n\n\n'
        ]
    ]

    prints = list(re.finditer('^print', completion, re.MULTILINE))
    if len(prints) > 1:
        completion = completion[:prints[1].start()]

    defs = list(re.finditer('^def', completion, re.MULTILINE))
    if len(defs) > 1:
        completion = completion[:defs[1].start()]

    start_pos = 0

    terminals_pos = [pos for pos in [find_re(completion, terminal, start_pos)
                                     for terminal in terminals] if pos != -1]
    if len(terminals_pos) > 0:
        return completion[:min(terminals_pos)]
    else:
        return completion


def four_parity_reference(b1, b2, b3, b4):
    bit_sum = sum([b1, b2, b3, b4])
    return bit_sum % 2


def eval_model(task="parity"):
    if task == "parity":
        inputs = [i for i in itertools.product(range(2), repeat=4)]
        ground_truth = {i: four_parity_reference(*i) for i in inputs}

        code = """def fixed_bugs(b1,b2,b3,b4):\n    bit_sum = sum([b1,b2,b3,b4])\n    return bit_sum % 2\n"""
        print("About to exec")
        exec(code)
        for i in inputs:
            print(fixed_bugs(*inputs[i]), ground_truth[i])

    else:
        raise ValueError("Unknown task: {}".format(task))


if __name__ == "__main__":
    ckpt = './checkpoints/codegen-350M-mono'

    eval_model()
