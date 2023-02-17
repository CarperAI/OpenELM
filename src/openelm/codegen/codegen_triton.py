import random
import string

import numpy as np
import tritonclient.grpc as client_util
from tritonclient.utils import InferenceServerException, np_to_triton_dtype

from .codegen_utilities import create_custom_gpt2_tokenizer

np.finfo(np.dtype("float32"))
np.finfo(np.dtype("float64"))


class CodeGenProxy:
    def __init__(self, host: str = "0.0.0.0", port: int = 8001, verbose: bool = False):
        self.client = client_util.InferenceServerClient(
            url=f"{host}:{port}", verbose=verbose
        )
        self.PAD_CHAR = 50256
        # Max number of tokens the model can handle
        self.MAX_MODEL_LEN = 2048

    class TokensExceedsMaximum(Exception):
        pass

    @staticmethod
    def prepare_tensor(name: str, tensor_input):
        t = client_util.InferInput(
            name, tensor_input.shape, np_to_triton_dtype(tensor_input.dtype)
        )
        t.set_data_from_numpy(tensor_input)
        return t

    @staticmethod
    def trim_with_stopwords(output: str, stopwords: list) -> str:
        for w in sorted(stopwords, key=len, reverse=True):
            if output.endswith(w):
                output = output[: -len(w)]
                break
        return output

    @staticmethod
    def to_word_list_format(word_dict, tokenizer):
        flat_ids = []
        offsets = []
        for word_dict_item in word_dict:
            item_flat_ids = []
            item_offsets = []

            for word in word_dict_item:
                ids = tokenizer.encode(word).ids

                if len(ids) == 0:
                    continue

                item_flat_ids += ids
                item_offsets.append(len(ids))

                # Hack, can we do this better?
                if word == "\n\n":
                    item_flat_ids += [198, 198]
                    item_offsets.append(2)

            flat_ids.append(np.array(item_flat_ids))
            offsets.append(np.cumsum(np.array(item_offsets)))

        pad_to = max(1, max(len(ids) for ids in flat_ids))

        for i, (ids, offs) in enumerate(zip(flat_ids, offsets)):
            flat_ids[i] = np.pad(ids, (0, pad_to - len(ids)), constant_values=0)
            offsets[i] = np.pad(offs, (0, pad_to - len(offs)), constant_values=-1)

        return np.array([flat_ids, offsets], dtype="int32").transpose((1, 0, 2))

    def generate(self, input_ids, batch_size, temperature, top_p, gen_max_length):

        model_name = "fastertransformer"
        # ugly hack to set the data type correctly. Huggingface models want
        # int32, but fastertransformer needs uint32
        # i could've done the conversion from uint32 to int32 in the model but
        # that'd be inefficient.
        np_type = np.uint32
        input_start_ids = input_ids.astype(np_type)
        input_start_ids = np.repeat(input_start_ids, batch_size, axis=0).astype(np_type)
        prompt_len = input_start_ids.shape[1]
        input_len = prompt_len * np.ones([input_start_ids.shape[0], 1]).astype(np_type)
        max_tokens = gen_max_length
        prompt_tokens: int = input_len[0][0]
        requested_tokens = max_tokens + prompt_tokens

        if requested_tokens > self.MAX_MODEL_LEN:
            print(1)
            raise self.TokensExceedsMaximum(
                f"This model's maximum context length is {self.MAX_MODEL_LEN}, however you requested "
                f"{requested_tokens} tokens ({prompt_tokens} in your prompt; {max_tokens} for the completion). "
                f"Please reduce your prompt; or completion length."
            )
        output_len = np.ones_like(input_len).astype(np_type) * max_tokens
        runtime_top_p = top_p * np.ones([input_start_ids.shape[0], 1]).astype(
            np.float32
        )
        temperature = temperature * np.ones([input_start_ids.shape[0], 1]).astype(
            np.float32
        )
        random_seed = np.random.randint(
            0, 1e9, [input_start_ids.shape[0], 1], dtype=np.int32
        )
        # beam_width = np.ones([input_start_ids.shape[0], 1]).astype(np_type)
        # beam_search_diversity_rate = 0.5*np.ones([input_start_ids.shape[0], 1]).astype(np.float32)
        # frequency_penalty = 1.0*np.ones([input_start_ids.shape[0], 1]).astype(np.float32)

        inputs = [
            self.prepare_tensor("input_ids", input_start_ids),
            self.prepare_tensor("input_lengths", input_len),
            self.prepare_tensor("request_output_len", output_len),
            self.prepare_tensor("runtime_top_p", runtime_top_p),
            self.prepare_tensor("temperature", temperature),
            self.prepare_tensor("random_seed", random_seed),
        ]
        result = self.client.infer(model_name, inputs)
        output_data = result.as_numpy("output_ids")

        if output_data is None:
            raise RuntimeError("No output data")

        output_data = output_data.squeeze(1)
        return output_data
        # All of these squeeze(1)s are to remove the beam width dimension.

    @staticmethod
    def random_completion_id():
        return "cmpl-" + "".join(
            random.choice(string.ascii_letters + string.digits) for _ in range(29)
        )

    def __call__(self, input_ids, batch_size, temperature, top_p, gen_max_length):
        try:
            tokens = self.generate(
                input_ids,
                batch_size=batch_size,
                temperature=temperature,
                top_p=top_p,
                gen_max_length=128,
            )
        except InferenceServerException as E:
            print(E)
        return tokens


def setup_triton(cfg):
    cg_triton = CodeGenProxy(cfg.triton_host, cfg.triton_port)
    tokenizer = create_custom_gpt2_tokenizer()
    tokenizer.padding_side = "left"
    tokenizer.pad_token = cfg.pad_token
    return cg_triton, tokenizer


def sample_triton(batch, cfg, cg_triton, tokenizer, add_def=False):
    input_ids = batch["input_ids"]
    input_ids_len = input_ids.shape[1]
    tokens = cg_triton.generate(
        input_ids,
        batch_size=cfg.batch_size,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        gen_max_length=cfg.gen_max_length,
    )
    if add_def:
        input_ids_len -= 1
    text = tokenizer.batch_decode(tokens[:, input_ids_len:, ...])
    return text
