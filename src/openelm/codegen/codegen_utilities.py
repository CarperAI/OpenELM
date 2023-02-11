import random
import re

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def set_seed(seed=None, deterministic=True) -> int:
    if seed is None:
        seed = torch.Generator().seed()
    random.seed(seed)
    # TODO: test this
    # os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = not deterministic
        # torch.use_deterministic_algorithms(deterministic)
    return seed


def create_model(path=None, fp16=True):
    if fp16:
        return AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=torch.float16, low_cpu_mem_usage=True
        )
    else:
        return AutoModelForCausalLM.from_pretrained(path)


def truncate(completion, def_num=1, print_num=1, only_local_scope=False):
    def find_re(string, pattern, start_pos):
        m = pattern.search(string, start_pos)
        return m.start() if m else -1

    terminals = [
        re.compile(r, re.MULTILINE)
        for r in ["^#", re.escape("<|endoftext|>"), "^'''", '^"""', "\n\n\n"]
    ]
    prints = list(re.finditer("^print", completion, re.MULTILINE))
    if len(prints) > print_num:
        completion = completion[: prints[print_num].start()]

    if only_local_scope:
        global_lines = list(re.finditer("^[a-zA-Z]", completion, re.MULTILINE))
        if global_lines:
            completion = completion[: global_lines[0].start()]
    else:
        defs = list(re.finditer("^def", completion, re.MULTILINE))
        if len(defs) > def_num:
            completion = completion[: defs[def_num].start()]

    start_pos = 0

    terminals_pos = [
        pos
        for pos in [find_re(completion, terminal, start_pos) for terminal in terminals]
        if pos != -1
    ]
    if len(terminals_pos) > 0:
        return completion[: min(terminals_pos)]
    else:
        return completion


def model_setup(cfg, device=None):
    set_seed(cfg.seed, deterministic=True)
    if device is None:
        device = torch.device("cuda" if cfg.cuda else "cpu")
    use_fp16 = True
    if not cfg.fp16 or device.type == "cpu":
        use_fp16 = False

    if cfg.model.startswith("codegen-16B"):
        use_fp16 = True

    tokenizer = AutoTokenizer.from_pretrained(cfg.model)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = cfg.pad_token

    ckpt_path = cfg.model
    if cfg.gpus > 1:
        model = torch.nn.DataParallel(
            create_model(ckpt_path, fp16=use_fp16), device_ids=list(range(cfg.gpus))
        ).to(device)
    else:
        model = create_model(ckpt_path, fp16=use_fp16).to(device)
    return model, tokenizer, device


def sample(
    cfg, model, tokenizer, batch, batch_size=None, starting_idx=None
) -> list[str]:
    # TODO: add dict param for extra hparams
    """Run a model on a batch of contexts for a particular task."""
    if batch_size is None:
        batch_size = cfg.batch_size
    device = torch.device("cuda" if cfg.cuda else "cpu")
    input_ids_len = batch["input_ids"].shape[1]
    if starting_idx is None:
        starting_idx = input_ids_len
    with torch.inference_mode():
        batch = batch.to(device)
        # TODO: num_gpus > 1
        if cfg.gpus > 1:
            tokens = model.module.generate(
                **batch,
                do_sample=True,
                num_return_sequences=batch_size,
                temperature=cfg.temp,
                max_new_tokens=cfg.gen_max_len,
                top_p=cfg.top_p,
                pad_token_id=cfg.pad_token,
                use_cache=True,
            )
        else:
            tokens = model.generate(
                **batch,
                do_sample=True,
                num_return_sequences=batch_size,
                temperature=cfg.temp,
                max_new_tokens=cfg.gen_max_len,
                top_p=cfg.top_p,
                pad_token_id=cfg.pad_token,
                use_cache=True,
            )
        text = tokenizer.batch_decode(tokens[:, starting_idx:, ...])
    return text
