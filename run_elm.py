"""
This module gives an example of how to run the main ELM class.

It uses the hydra library to load the config from the
config/elm_sodarace_cfg.yaml file.

This config file demonstrates an example of running ELM with the Sodarace
environment, a 2D physics-based environment in which robots specified by
Python dictionaries are evolved over.

"""
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

from openelm import ELM
from openelm.configs import SodaraceELMConfig

cs = ConfigStore.instance()
cs.store(name="config", node=SodaraceELMConfig)


# Load hydra config from yaml files and command line arguments.
@hydra.main(
    config_name="config",
    version_base="1.2",
)
def main(cfg):
    print("----------------- Config ---------------")
    print(OmegaConf.to_yaml(cfg))
    print("-----------------  End -----------------")
    elm = ELM(cfg)
    print("Best Individual: ", elm.run())


if __name__ == "__main__":
    main()
