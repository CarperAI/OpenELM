"""
This module gives an example of how to run the main ELM class.

It uses the hydra library to load the config from the
config/elm_sodarace_cfg.yaml file.

This config file demonstrates an example of running ELM with the Sodarace
environment, a 2D physics-based environment in which robots specified by
Python dictionaries are evolved over.

"""
import hydra
from omegaconf import OmegaConf

from openelm import ELM
from openelm.constants import SRC_PATH


# Load hydra config from yaml files and command line arguments.
@hydra.main(
    config_path=str(SRC_PATH / "config"), config_name="elm_sodarace_cfg", version_base="1.2"
)
def main(cfg):
    print("----------------- Config ---------------")
    print(OmegaConf.to_yaml(cfg))
    print("-----------------  End -----------------")
    elm = ELM(cfg)
    print("Best Sodaracer: ", elm.run())


if __name__ == "__main__":
    main()
