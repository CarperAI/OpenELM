"""This is the benchmark test of diff models."""
import functools
import itertools
import multiprocessing as mp
import re
from typing import Union

import hydra
import torch
from benchmarks import eval_code_string, parity_reference
from omegaconf import OmegaConf
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from openelm.codegen.codegen_utilities import set_seed
from openelm.constants import SRC_PATH
from openelm.utils.diff_eval import apply_diff, split_diff

parity_test_data = {
    i: parity_reference(*i) for i in itertools.product(range(2), repeat=4)
}


def eval_completions(
    codes: Union[str, list[str]], task: str = "parity", timeout: int = 5
) -> Union[int, list[int]]:
    """
    Evaluate (a batch of) the modified codes on a task.
    Args:
        codes: either a string or a batch of strings. The code(s) to be evaluated.
        task: (Optional) the task to be performed.
        timeout: (Optional) the timeout (in seconds).
    Returns:
        either the status code of the string or a list of status codes of the batch
        of strings (depending on whether `codes` is batched).
    """
    if task == "parity":
        _codes = codes if isinstance(codes, list) else [codes]
        results = []
        for code in _codes:
            results.append(eval_code_string(code, parity_test_data, timeout))
        if isinstance(codes, list):
            # Batch evaluation returns the batch
            return results
        else:
            # Single evaluation returns a single result
            return results[0]
    else:
        raise ValueError(f"Unknown task: {task}")


def mutate_code(n_bugs: int = 5, task: str = "parity") -> tuple:
    """
    Mutate code to create n bugs. Output the prompt in diff format.
    Args:
        n_bugs: number of bugs to introduce (from 1 to 5).
        task: (Optional) the task to be performed.
    Returns:
        mutated_code, function_string
    """
    mutation_template = [
        f"<NME> {task}.py\n<BEF> ",
        "",  # placeholder for the context, e.g., the buggy code
        "\n<MSG> Fixed bugs",
    ]
    if task == "parity":
        variables = ["b", "b", "b", "b", 2]
        for i in range(n_bugs):
            variables[i] = "c" if i < 4 else 3
        func_str = (
            'def parity(b1,b2,b3,b4):\n    """Return binary parity of a sequence of input bits.'
            ' Return 0 for even parity, 1 for odd parity."""\n    bit_sum = sum(['
            "{}1,{}2,{}3,{}4])\n    return bit_sum % {}".format(*variables)
        )
        mutation_template[1] = func_str
        return "".join(mutation_template), func_str
    else:
        raise ValueError(f"Unknown task: {task}")


@hydra.main(
    config_path=str(SRC_PATH / "config"),
    config_name="benchmark_diff_cfg",
    version_base="1.2",
)
def main(cfg):
    print("----------------- Config ---------------")
    print(OmegaConf.to_yaml(cfg))
    print("-----------------  End -----------------")

    device = torch.device("cuda" if cfg.cuda else "cpu")
    config = AutoConfig.from_pretrained(cfg.model)
    # Sometimes our model just fresh came out of training. Force use_cache to be true.
    config.use_cache = True
    tokenizer = AutoTokenizer.from_pretrained(cfg.model)
    if cfg.gpus > 1:
        model = torch.nn.DataParallel(
            AutoModelForCausalLM.from_pretrained(cfg.model, config=config),
            device_ids=list(range(cfg.gpus)),
        ).to(device)
        model.generate = model.module.generate
    else:
        model = AutoModelForCausalLM.from_pretrained(cfg.model, config=config).to(
            device
        )

    # Prepare the mutated codes with cfg.n_bugs bugs.
    mutated_str, function_str = mutate_code(n_bugs=cfg.n_bugs, task=cfg.tasks[0])
    mutated_encoding = tokenizer(
        [mutated_str],
        return_tensors="pt",
    ).to(device)
    token_len = mutated_encoding["input_ids"].shape[1]

    # Generate codes
    num_batches = cfg.n_trials // cfg.batch_size
    end_of_diff = re.compile("\n[^ +-@]+")
    codes = []
    for _ in tqdm(range(num_batches), desc=f"Running benchmark with {cfg.n_bugs} bugs"):
        set_seed(torch.random.seed())
        tokens = model.generate(
            **mutated_encoding,
            do_sample=True,
            num_return_sequences=cfg.batch_size,
            temperature=cfg.temp,
            max_length=token_len + cfg.gen_max_len,
            top_p=cfg.top_p,
            pad_token_id=cfg.pad_token,
        )
        texts = tokenizer.batch_decode(tokens)
        for text in texts:
            # split the diff text according to <NME>, <BEF>, <MSG>, <DFF>.
            parsed = split_diff(text)
            # truncate the diff hunk at the first line not starting with " ", "+", "-", or "@".
            if parsed and all(
                [s in parsed for s in ["name", "file", "message", "diff"]]
            ):
                diff_hunk = end_of_diff.split(parsed["diff"])[0]
                codes.append(apply_diff(function_str, diff_hunk))
            else:
                # Invalid format. No patching.
                codes.append(function_str)

    # Evaluate the codes in threads (most importantly, separate reliability_guard in separate
    # threads as it would otherwise disable things in this current thread).
    # We apply diff patch loosely:
    #   1. it ignores the line numbers;
    #   2. it ignores invalid lines (not starting with " ", "+" or "-" and not being "@@ ... @@").
    with mp.Pool(processes=cfg.num_process) as pool:
        eval_fn = functools.partial(
            eval_completions, task=cfg.tasks[0], timeout=cfg.timeout
        )
        results = list(pool.map(eval_fn, codes))

    corr_cnt = sum([r == 0 for r in results])
    print(f"Number of bugs: {cfg.n_bugs}\n")
    print(f"Mutated code to be fixed:\n{function_str}\n")
    print(
        f"Result: {corr_cnt} successful completions in {cfg.n_trials} trials, {(corr_cnt / cfg.n_trials) * 100}%"
    )


if __name__ == "__main__":
    main()
