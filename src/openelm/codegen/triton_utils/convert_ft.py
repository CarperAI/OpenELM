import argparse
import os
from string import Template

import torch
from codegen_gptj_converter import cg2gptj
from gptj_ftconverter import split_and_convert_main
from transformers import AutoTokenizer, GPTJConfig


def round_up(x, multiple):
    remainder = x % multiple
    return x if remainder == 0 else x + multiple - remainder


SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
CONFIG_TEMPLATE_PATH = os.path.join(SCRIPT_DIR, "config_template.pbtxt")

# Generate a config file for a CodeGen model for use with Triton

parser = argparse.ArgumentParser("Convert SalesForce CodeGen model to GPT-J FT")
parser.add_argument(
    "--model_name",
    default="Salesforce/codegen-350M-mono",
    help="which SalesForce model to convert",
)
parser.add_argument(
    "--template", default=CONFIG_TEMPLATE_PATH, help="Path to the config template"
)
parser.add_argument(
    "--tokenizer",
    default="Salesforce/codegen-16B-multi",
    help="Name or path to the tokenizer",
)
parser.add_argument(
    "--output_dir", required=True, help="Where to store the converted model"
)
parser.add_argument(
    "--n_gpus",
    "--num_gpus",
    help="Number of GPUs to use for inference",
    type=int,
    default=1,
)
parser.add_argument(
    "--t_gpus",
    "--train_gpus",
    help="Number of GPUs used for training",
    type=int,
    default=1,
)
parser.add_argument(
    "--processes",
    "--p",
    type=int,
    help="How many processes to spawn for conversion (default: 4)",
    default=1,
)
parser.add_argument(
    "--weight_data_type",
    type=str,
    default="fp32",
    choices=["fp32", "fp16"],
    help="output weight data type",
)
args = parser.parse_args()

# Vars we need to fill in:
# name
# tensor_para_size
# max_seq_len
# is_half
# head_num
# size_per_head
# inter_size
# vocab_size
# start_id
# end_id
# decoder_layers
# name
# rotary_embedding
# checkpoint_path

# Global options
gptj_model = cg2gptj(args.model_name)
config = gptj_model.config
tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
max_seq_len = config.n_positions
is_half = "1" if config.torch_dtype == torch.float16 else "0"

# Read in the template config file
with open(args.template, "r") as f:
    template = Template(f.read())

# from code_model
model_name = args.model_name.split("/")[-1]

version = "1"
params = {}
params["tensor_para_size"] = args.n_gpus
params["name"] = model_name
params["max_seq_len"] = max_seq_len
params["is_half"] = is_half
params["head_num"] = config.n_head
params["size_per_head"] = config.n_embd // config.n_head
params["inter_size"] = 4 * config.n_embd
# The original script rounded up vocab size to a multiple of 1024. But for diff models we do not.
params["vocab_size"] = gptj_model.config.vocab_size
params["start_id"] = tokenizer.eos_token_id
params["end_id"] = tokenizer.eos_token_id
params["decoder_layers"] = config.n_layer
params["rotary_embedding"] = config.rotary_dim
# NOTE: this assumes that the model dir follows the format used by the other conversion scripts
model_dir = os.path.join(args.output_dir, f"{model_name}-{args.n_gpus}gpu")
os.makedirs(model_dir, exist_ok=True)
weights_path = os.path.join(
    model_dir, "fastertransformer", f"{version}", f"{args.n_gpus}-gpu"
)
params["checkpoint_path"] = weights_path
triton_config = template.substitute(params)
assert "${" not in triton_config

# Make directory structure
os.makedirs(weights_path, exist_ok=True)

# Write config file
config_path = os.path.join(model_dir, "fastertransformer", "config.pbtxt")
with open(config_path, "w") as f:
    f.write(triton_config)

print(f"Created config file for {model_name}")
print(f"  Config:  {config_path}")

# Convert the model weights
# args [gptj model] [weight path] [n-gpu] [t-gpu] [fp16]

split_and_convert_main(
    gptj_model,
    weights_path,
    args.n_gpus,
    args.t_gpus,
    args.weight_data_type,
    args.processes,
)
print("==========================================================")
print(f"Converted weights for {model_name}")
print(f"  Config:  {config_path}")
print(f"  Weights: {weights_path}")
print("==========================================================")
