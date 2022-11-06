import hydra
from omegaconf import OmegaConf

from elm.constants import SRC_PATH
from elm.diff_model import DiffModel
from elm.environments import Sodarace
from elm.map_elites import MAPElites

TEST_SEED = {
    "program_str": """from elm.environments.sodaracer.walker import walker_creator


def make_square(wc, x0, y0, x1, y1):
    \"\"\"Make a square with top left x0,y0 and top right x1,y1.\"\"\"
    j0 = wc.add_joint(x0, y0)
    j1 = wc.add_joint(x0, y1)
    j2 = wc.add_joint(x1, y1)
    j3 = wc.add_joint(x1, y0)
    return j0, j1, j2, j3


def make_walker():
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
    wc.add_muscle(sides[0], center, False, 5.0, 0.0)
    wc.add_muscle(sides[1], center, False, 10.0, 0.0)
    wc.add_muscle(sides[2], center, False, 2.0, 0.0)

    return wc.get_walker()
""",
    "result_dict": {
        "useLEO": True,
        "nodes": [
            {"x": 0, "y": 0},
            {"x": 0, "y": 10},
            {"x": 10, "y": 10},
            {"x": 10, "y": 0},
            {"x": 5, "y": 5},
        ],
        "connections": [
            {"sourceID": "0", "targetID": "1", "cppnOutputs": [0, 0, 0, -10.0]},
            {"sourceID": "1", "targetID": "2", "cppnOutputs": [0, 0, 0, -10.0]},
            {"sourceID": "2", "targetID": "3", "cppnOutputs": [0, 0, 0, -10.0]},
            {"sourceID": "3", "targetID": "0", "cppnOutputs": [0, 0, 0, -10.0]},
            {"sourceID": "3", "targetID": "4", "cppnOutputs": [0, 0, 0, -10.0]},
            {"sourceID": "0", "targetID": "4", "cppnOutputs": [0, 0, 0.0, 5.0]},
            {"sourceID": "1", "targetID": "4", "cppnOutputs": [0, 0, 0.0, 10.0]},
            {"sourceID": "2", "targetID": "4", "cppnOutputs": [0, 0, 0.0, 2.0]},
        ],
    },
}


class ELM:
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        # TODO: hierarchical hydra config for different environments, config type validation.
        self.diff_model = DiffModel(self.cfg)
        self.seed = TEST_SEED
        self.environment = Sodarace(
            seed=TEST_SEED,
            diff_model=self.diff_model,
            eval_steps=self.cfg.evaluation_steps,
        )
        self.map_elites = MAPElites(
            self.environment,
            n_bins=self.cfg.behavior_n_bins,
            history_length=self.cfg.evo_history_length,
        )

    def run(self) -> str:
        return self.map_elites.search(
            initsteps=self.cfg.evo_init_steps, totalsteps=self.cfg.evo_n_steps
        )


# Load hydra config from yaml files and command line arguments.
@hydra.main(
    config_path=str(SRC_PATH / "config"), config_name="elm_cfg", version_base="1.2"
)
def main(cfg):
    print("----------------- Config ---------------")
    print(OmegaConf.to_yaml(cfg))
    print("-----------------  End -----------------")
    elm = ELM(cfg)
    print("Best Sodaracer: ", elm.run())


if __name__ == "__main__":
    main()
