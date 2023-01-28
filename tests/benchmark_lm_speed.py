from time import time

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from openelm.environments import SQUARE_SEED


def main():
    device = torch.device("cuda")
    model_str = "Salesforce/codegen-2B-mono"
    config = AutoConfig.from_pretrained(model_str)
    # Sometimes our model just fresh came out of training. Force use_cache to be true.
    config.use_cache = True
    tokenizer = AutoTokenizer.from_pretrained(model_str)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = 50256
    fp16 = True
    if fp16:
        model = AutoModelForCausalLM.from_pretrained(
            model_str, torch_dtype=torch.float16, low_cpu_mem_usage=True
        ).to(device)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_str, config=config).to(
            device
        )

    mutated_encoding = tokenizer(
        [SQUARE_SEED["program_str"]],
        return_tensors="pt",
    ).to(device)
    token_len = mutated_encoding["input_ids"].shape[1]
    print(mutated_encoding["input_ids"].shape)
    start = time()
    print("Starting inference")
    for i in range(1):
        with torch.inference_mode():
            tokens = model.generate(
                **mutated_encoding,
                do_sample=True,
                num_return_sequences=1,
                temperature=0.85,
                max_length=min(token_len + 1024, 2048),
                top_p=0.95,
                pad_token_id=tokenizer.pad_token_id,
            )
            _ = tokenizer.batch_decode(tokens)

    print(f"Time: {time() - start} seconds")


if __name__ == "__main__":
    main()
