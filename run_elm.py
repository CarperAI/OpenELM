import hydra
from omegaconf import OmegaConf

from elm import ELM
from elm.constants import SRC_PATH
from elm.diff_model import PromptMutationForSodarace


# Load hydra config from yaml files and command line arguments.
@hydra.main(
    config_path=str(SRC_PATH / "config"), config_name="elm_sodarace_cfg", version_base="1.2"
)
def main(cfg):
    print("----------------- Config ---------------")
    print(OmegaConf.to_yaml(cfg))
    print("-----------------  End -----------------")
    elm = ELM(cfg, PromptMutationForSodarace)
    print("Best Sodaracer: ", elm.run())


if __name__ == "__main__":
    main()
