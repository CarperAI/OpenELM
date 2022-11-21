import hydra
from omegaconf import OmegaConf

from elm import ELM
from elm.constants import SRC_PATH


# Load hydra config from yaml files and command line arguments.
@hydra.main(
    config_path=str(SRC_PATH / "config"), config_name="elm_cfg", version_base="1.2"
)
def main(cfg):
    print("----------------- Config ---------------")
    print(OmegaConf.to_yaml(cfg))
    print("-----------------  End -----------------")
    elm = ELM(cfg, )
    print("Best Sodaracer: ", elm.run())


if __name__ == "__main__":
    main()
