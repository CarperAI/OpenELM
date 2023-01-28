import itertools
import os
import re
import shutil
from typing import Iterator

import hydra
import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
import torch

from openelm.codegen.codegen_utilities import model_setup, sample, truncate
from openelm.codegen.codex_execute import (
    TimeoutException,
    create_tempdir,
    reliability_guard,
    swallow_io,
    time_limit,
)
from openelm.constants import SRC_PATH

from openelm.codegen.codegen_utilities import truncate
from openelm.environments.sodaracer.walker import Walker
from openelm.sandbox.server.utils import sandbox_unsafe_execute

CIRCLE = """
def make_circle(wc, cx, cy, radius, num_points):
    \"\"\"Approximate a circle with center (cx,cy) square with num_points points.\"\"\"
    joints = []

    tot_ang = 3.14 * 2.0

    for idx in range(num_points):
        ang = (tot_ang / num_points) * idx
        x = math.cos(ang) * radius + cx
        y = math.sin(ang) * radius + cy
        joints.append(wc.add_joint(x, y))

    return joints

"""

RADIAL = """
def make_radial_walker():
    \"\"\"Create a radial walker.\"\"\"
    wc = walker_creator()

    num_points = 8
    rad = 5.0
    cx, cy = (5, 5)
    # the main body is a square
    points = make_circle(wc, cx, cy, rad, num_points)
    center = wc.add_joint(cx, cy)

    for k in range(num_points):
        wc.add_muscle(points[k], points[(k + 1) % num_points])
        wc.add_muscle(points[k], center, float(k) / num_points, float(k) / num_points)

    return wc.get_walker()

"""

WHEEL = """
def make_wheel_walker():
    \"\"\"Create a wheel walker.\"\"\"
    wc = walker_creator()
    num_points = 8
    rad = 3.0
    cx, cy = (11, 5)
    points = make_circle(wc, 0.6, -0.5, rad / 2, num_points)
    center = wc.add_joint(cx + 1, cy + 1)
    for j in range(num_points):
        for i in range(num_points - 5):
            wc.add_muscle(points[j], points[(i + j) % num_points],
                          0.0, 1.0, (j + 1) / num_points)
        wc.add_muscle(points[j], center, 3, (j + 1) / num_points)
    return wc.get_walker()

"""

SQUARE = """
def make_square(wc, x0, y0, x1, y1):
    \"\"\"Make a square with top left x0,y0 and top right x1,y1.\"\"\"
    j0 = wc.add_joint(x0, y0)
    j1 = wc.add_joint(x0, y1)
    j2 = wc.add_joint(x1, y1)
    j3 = wc.add_joint(x1, y0)
    return j0, j1, j2, j3


def make_square_walker():
    \"\"\"Create a square walker.\"\"\"
    wc = walker_creator()

    # the main body is a square
    sides = make_square(wc, 0, 0, 10, 10)
    center = wc.add_joint(5, 5)

    # connect the square with distance muscles
    for k in range(len(sides) - 1):
        wc.add_muscle(sides[k], sides[k + 1])
    wc.add_muscle(sides[3], sides[0])

    # one prong of the square is a distance muscle
    wc.add_muscle(sides[3], center)

    # the other prongs from the center of the square are active
    wc.add_muscle(sides[0], center, 5.0, 0.0)
    wc.add_muscle(sides[1], center, 10.0, 0.0)
    wc.add_muscle(sides[2], center, 2.0, 0.0)

    return wc.get_walker()

"""

IMPORTS = """
from openelm.environments.sodaracer.walker.walk_creator import walker_creator
import math

"""

INSTRUCTION = """
#Combine the radial, wheel, and square seed programs above to make a new walker.

def make_walker():

"""

def benchmark_crossover(cfg, model, tokenizer, device):
    func_start = "\ndef make_walker():\n"
    temperature = 0.88

    encoding = tokenizer(
        [PROMPT + func_start],
        truncation=True,
        padding=True,
        return_tensors="pt",
        max_length=2048
    ).to(device)
    token_len = encoding.input_ids.shape[1]
    results = []
    for i in range(cfg.batch_size):
        with torch.inference_mode():
                tokens = model.generate(
                    **encoding,
                    do_sample=True,
                    num_return_sequences=1,
                    temperature=temperature,
                    max_length=2048,
                    top_p=cfg.top_p,
                    pad_token_id=cfg.pad_token,
                    use_cache=True,
                )
                text = tokenizer.batch_decode(tokens[:, token_len - 1:, ...])
        truncations = map(truncate, text)
        for truncation in truncations:
            try:
                execution_result = sandbox_unsafe_execute(PROMPT + func_start + truncation, "make_walker")
                if isinstance(execution_result, Walker):
                    if execution_result.validate():
                        results.append(1)
                else:
                    print("Failed execution, type:", execution_result)
                    results.append(execution_result)
            except Exception as e:
                print(e, "Exception:")
                results.append(6)

    print((results.count(1) / len(results)) * 100, "%")

# Load hydra config from yaml files and command line arguments.
@hydra.main(
    config_path=str(SRC_PATH / "config"),
    config_name="benchmark_cfg",
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
    tokenizer.padding_side = "left"
    tokenizer.pad_token = 50256

    if cfg.fp16:
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model, torch_dtype=torch.float16, low_cpu_mem_usage=True
        ).to(device)
    else:
        model = AutoModelForCausalLM.from_pretrained(cfg.model, config=config).to(
            device
        )
    #model, tokenizer = model_setup(cfg)
    benchmark_crossover(cfg, model, tokenizer, device)

if __name__ == "__main__":
    main()
