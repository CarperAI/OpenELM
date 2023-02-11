import functools
import json
import os
import time
from itertools import permutations
from pathlib import Path
from typing import Union

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import trange

from openelm.codegen.codegen_utilities import model_setup, sample, truncate
from openelm.constants import SRC_PATH
from openelm.environments.environments import Sodaracer
from openelm.environments.sodaracer.walker import Walker
from openelm.map_elites import Map
from openelm.utils.code_eval import pool_exec_processes

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
def make_walker():
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
def make_walker():
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

SQUARE_PREREQ = """
def make_square(wc, x0, y0, x1, y1):
    \"\"\"Make a square with top left x0,y0 and top right x1,y1.\"\"\"
    j0 = wc.add_joint(x0, y0)
    j1 = wc.add_joint(x0, y1)
    j2 = wc.add_joint(x1, y1)
    j3 = wc.add_joint(x1, y0)
    return j0, j1, j2, j3

"""

SQUARE = """
def make_walker():
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

GALLOPER_PREREQ = """
def make_sensor(wc, x0, y0, x1, y1, d):
    return (
        wc.add_joint(x0, y0),
        wc.add_joint(x1, y1),
        wc.add_joint(x1, y0),
        wc.add_joint(x0, y1),
        wc.add_joint(d, 0.5),
        wc.add_joint(x1, 0.5),
    )

"""

GALLOPER = """
def make_walker(
    dx=0.0,
    dy=0.0,
    ddr=0,
    ddc=1.6,
):
    wc = walker_creator()
    ends = [
        make_sensor(wc, 5 + dx, -1 + dy, ddr, ddc, 4.5),
        make_sensor(wc, 0, -0.1, sid, 9.5, 0.03),
        make_sensor(wc, 5.5, -0.001, 5.0, 4.86 + 0.8, 0.07),
        make_sensor(wc, 5.5, -3.0, 6.0, 4.86 + 0.8, 0.07),
        make_sensor(wc, 0, dx, ddr, ddc, 1.0),
    ]

    sides = ends[0] + ends[1] + ends[2] + ends[-1] + ends[-2] + ends[-3]

    center = wc.add_joint(dx, dy)

    # connect the square with distance muscles
    for k in range(len(sides) - 6):
        wc.add_muscle(sides[k], sides[k + 1], 30, 0.5)
    wc.add_muscle(sides[2], sides[4], 4.0, 0.8)
    for k in range(len(sides) - 2):
        wc.add_muscle(sides[k], sides[k + 2], 18.0, 60.0 / 5.5)

    for k in reversed(range(len(sides) - 6)):
        wc.add_muscle(sides[k], sides[k + 5], 4.0, 20.0 / 9.0)

    wc.add_muscle(center, sides[7], 2, 0, 90.0 / 9.0)
    return wc.get_walker()

"""

QUERY_CPPN = """
def query_cppn(wc, xgrid, ygrid, scale, connect_func, amp_func, phase_func):
    \"\"\"Create a grid of points and functionally connect them.\"\"\"
    joints = {}
    for x in range(xgrid):
        for y in range(ygrid):
            joints[(x, y)] = wc.add_joint(x * scale, y * scale)
    for x1 in range(xgrid):
        for y1 in range(ygrid):
            for x2 in range(x1, xgrid):
                for y2 in range(y1, ygrid):
                    if x1 == y1 and x2 == y2:
                        continue
                    if connect_func(x1, y1, x2, y2):
                        amp = amp_func(x1, y1, x2, y2)
                        phase = phase_func(x1, y1, x2, y2)
                        wc.add_muscle(joints[(x1, y1)], joints[(x2, y2)], amp, phase)
    return joints

"""

CPPN_FIXED = """
def make_walker():
    wc = walker_creator()

    def connect(x1, y1, x2, y2):
        if ((x1 - x2) ** 2 + (y1 - y2) ** 2) > 4.5:
            return False
        return True

    def amp(x1, y1, x2, y2):
        return max(abs(x1 - x2), abs(y1 - y2))

    def phase(x1, y1, x2, y2):
        return np.sign(x1)

    _ = query_cppn(wc, 8, 3, 1.5, connect, amp, phase)

    return wc.get_walker()

"""

CPPN_MUTABLE = """
def make_walker():
    wc = walker_creator()

    def connect(x1, y1, x2, y2):
        if ((x1 - x2) ** 2 + (y1 - y2) ** 2) > 4.5:
            return False
        return True

    def amp(x1, y1, x2, y2):
        return max(abs(x1 - x2), abs(y1 - y2))

    def phase(x1, y1, x2, y2):
        return x1 if x1 % 2 == 1 else -x1

    _ = query_cppn(wc, 8, 3, 1.5, connect, amp, phase)

    return wc.get_walker()

"""

RUNNER = """
def make_walker(p_scale=1):  # acrylic of current (m)
    wc = walker_creator()

    def connect(x1, y1, x2, y2):
        if -2 * x1 + x2 * 2 > 2:
            return True
        return x1 <= abs(y1 - y2)

    def amp(x, y, x2, y2):
        return abs(x - x2) + abs(y - y2)

    def phase(x1, y1, x2, y2):
        return -x1 / 2 - math.cos(math.pi / 9)

    joints = query_cppn(wc, 5, 7 + p_scale, 2, connect, amp, phase)
    return wc.get_walker()

"""

IMPORTS = """
from openelm.environments.sodaracer.walker import walker_creator
import math

"""

INSTRUCTIONS = {
    0: "",
    1: "def make_walker():\n",
    2: "#Create a new walker by modifying the starting function above.\ndef make_walker():\n",
    3: "#Combine the ,starting programs above to make a new program.\ndef make_walker():\n",
}

SEEDS_DICT = {
    "wheel": WHEEL,
    "radial": RADIAL,
    "square": SQUARE,
    "cppn_fixed": CPPN_FIXED,
    "cppn_mutable": CPPN_MUTABLE,
    "galloper": GALLOPER,
    "runner": RUNNER,
}


class CrossoverBenchmark:
    def __init__(self, cfg):
        self.cfg = cfg
        self.reverse_seeds: dict[str, str] = {v: k for k, v in SEEDS_DICT.items()}

        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        self.device = torch.device("cuda" if cfg.cuda else "cpu")
        self.model, self.tokenizer, self.device = model_setup(cfg, self.device)

    def construct_prompt(self, seeds):
        prompt_str: str = IMPORTS
        seeds = [SEEDS_DICT[seed] for seed in seeds]
        if SQUARE in seeds:
            prompt_str += SQUARE_PREREQ
        if GALLOPER in seeds:
            prompt_str += GALLOPER_PREREQ
        if RADIAL in seeds or WHEEL in seeds:
            prompt_str += CIRCLE
        if CPPN_FIXED in seeds or CPPN_MUTABLE in seeds or RUNNER in seeds:
            prompt_str += QUERY_CPPN
        import_str: str = prompt_str
        if self.cfg.instruction == 3:
            instruction_str: str = INSTRUCTIONS[self.cfg.instruction].split(",")[0]
        for seed in seeds:
            prompt_str += seed
            if self.cfg.instruction == 3:
                instruction_str += self.reverse_seeds[seed] + ", "
        if self.cfg.instruction == 3:
            instruction_str += INSTRUCTIONS[self.cfg.instruction].split(",")[1]
        else:
            instruction_str = INSTRUCTIONS[self.cfg.instruction]
        if self.cfg.instruction != 0:
            import_str += instruction_str
        prompt_str += instruction_str
        return prompt_str, import_str

    def to_mapindex(self, b, bins):
        """Converts a phenotype (position in behaviour space) to a map index."""
        return (
            None
            if b is None
            else tuple(np.digitize(x, bins) for x, bins in zip(b, bins))
        )

    def benchmark_seeds(self, seeds):
        prompt, imports = self.construct_prompt(seeds)
        encoding = self.tokenizer(
            [prompt],
            truncation=True,
            padding=True,
            return_tensors="pt",
            max_length=2048,
        ).to(self.device)

        # Map setup
        n_bins: int = 12
        genotype_space = np.array([[0, 1000], [0, 1000], [0, 2000]]).T
        bins = np.linspace(*genotype_space, n_bins + 1)[1:-1].T  # type: ignore
        fitness_map: Map = Map(
            dims=(n_bins,) * genotype_space.shape[1],
            fill_value=-np.inf,
            dtype=float,
        )

        results: list[int] = []
        valid_fitnesses: list[float] = []
        local_scope_exec: bool = self.cfg.instruction != 0
        token_len: int = encoding.input_ids.shape[1]
        print("Benchmarking seeds: ", ", ".join(seeds))
        print(f"Prompt length: {token_len} tokens.")

        for _ in trange(self.cfg.n_trials // self.cfg.batch_size):
            completions: list[str] = sample(
                self.cfg,
                self.model,
                self.tokenizer,
                encoding,
                starting_idx=token_len,
            )
            trunc = functools.partial(truncate, only_local_scope=local_scope_exec)
            truncations: list[str] = list(
                imports + trunc for trunc in map(trunc, completions)
            )
            execution_results = pool_exec_processes(
                truncations,
                func_name="make_walker",
                processes=self.cfg.processes,
                debug=self.cfg.debug,
            )
            for i, result in enumerate(execution_results):
                try:
                    if isinstance(result, Walker) and result.validate():
                        sodaracer = Sodaracer(
                            program_str=truncations[i],
                            result_obj=result.to_dict(),
                            error_code=0,
                        )
                        if sodaracer.valid:
                            fitness: float = sodaracer.evaluate(1000)
                            if fitness is not None:
                                valid_fitnesses.append(fitness)
                                map_idx = self.to_mapindex(
                                    sodaracer.to_phenotype(), bins=bins
                                )
                                results.append(1)
                                if fitness > fitness_map[map_idx]:
                                    fitness_map[map_idx] = fitness
                    else:
                        if self.cfg.debug:
                            print("Failed execution, type:", result)
                            print(truncations[i])
                        results.append(result)
                except Exception as e:
                    if self.cfg.debug:
                        print(type(e), e)
                    results.append(6)

        # TODO: Instantiate MAP-Elites here to evolve the map

        valid_rate: float = (results.count(1) / len(results)) * 100
        qd_score: float = fitness_map.qd_score
        niches_filled: int = fitness_map.niches_filled
        result_dict: dict[str, Union[list, float, int]] = {
            "valid_rate": valid_rate,
            "qd_score": qd_score,
            "niches_filled": niches_filled,
            "valid_fitnesses": valid_fitnesses,
        }

        print(f"Valid rate for {seeds}: {valid_rate}%")
        print(f"QD score: {qd_score}")
        print(f"Niches filled: {niches_filled}")
        print(f"Average fitness: {np.nanmean(valid_fitnesses)}")
        return result_dict

    def run_benchmark(self):
        perm: list[tuple] = list(permutations(self.cfg.seeds))
        valid_rates: list[float] = []
        qd_scores: list[float] = []
        niches: list[int] = []
        all_fitnesses: dict[str, list[float]] = {}
        print("Permutations: ", perm)

        for seeds in perm:
            perm_results: dict = self.benchmark_seeds(seeds)
            valid_rates.append(perm_results["valid_rate"])
            qd_scores.append(perm_results["qd_score"])
            niches.append(perm_results["niches_filled"])
            all_fitnesses[", ".join(seeds)] = perm_results["valid_fitnesses"]

        valid_stats = (np.nanmean(valid_rates), np.nanstd(valid_rates))
        qd_stats = (np.nanmean(qd_scores), np.nanstd(qd_scores))
        niche_stats = (np.nanmean(niches), np.nanstd(niches))

        print(f"Validity stats: {valid_stats[0]:.2f}, {valid_stats[1]:.2f}")
        print(f"QD stats: {qd_stats[0]:.2f}, {qd_stats[1]:.2f}")
        print(f"Niche stats: {niche_stats[0]:.2f}, {niche_stats[1]:.2f}")
        results_dct = {
            "rates": valid_rates,
            "fitnesses": all_fitnesses,
            "qd_scores": qd_scores,
            "niches": niches,
            "valid_stats": valid_stats,
            "qd_stats": qd_stats,
            "niche_stats": niche_stats,
            "config": OmegaConf.to_container(self.cfg),
            "permutations": perm,
        }

        Path(self.cfg.save_path, f"{time.strftime('%Y%m%d-%H%M%S')}.json").write_text(
            json.dumps(results_dct)
        )


# Load hydra config from yaml files and command line arguments.
@hydra.main(
    config_path=str(SRC_PATH / "config"),
    config_name="benchmark_crossover_cfg",
    version_base="1.2",
)
def main(cfg):
    print("----------------- Config ---------------")
    print(OmegaConf.to_yaml(cfg))
    print("-----------------  End -----------------")

    crossover = CrossoverBenchmark(cfg)
    crossover.run_benchmark()


if __name__ == "__main__":
    main()
