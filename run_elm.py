"""
This module gives an example of how to run the main ELM class.

It uses the hydra library to load the config from the config dataclasses in
configs.py.

This config file demonstrates an example of running ELM with the Sodarace
environment, a 2D physics-based environment in which robots specified by
Python dictionaries are evolved over.

"""
import hydra
from omegaconf import OmegaConf

from openelm import ELM


# Load hydra config from yaml files and command line arguments.
@hydra.main(
    config_name="elmconfig",
    version_base="1.2",
)
def main(cfg):
    print("----------------- Config ---------------")
    print(OmegaConf.to_yaml(cfg))
    print("-----------------  End -----------------")
    cfg = OmegaConf.to_object(cfg)
    print(cfg)
    elm = ELM(cfg)
    print("Best Individual: ", elm.run(init_steps=cfg.qd.init_steps,
                                       total_steps=cfg.qd.total_steps))


if __name__ == "__main__":
    main()
