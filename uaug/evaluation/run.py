import logging
import os
from datetime import datetime

import hydra
from dotenv import load_dotenv
from omegaconf import OmegaConf

from uaug.config import Config
from uaug.utils.misc import print_config

# Load environment variables from .env
load_dotenv()

# Initialize Hydra config store
config_store = hydra.core.config_store.ConfigStore.instance()
config_store.store(name="base_config", node=Config)


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: Config):
    # Load and print configuration
    OmegaConf.resolve(cfg)
    cfg_yaml = OmegaConf.to_yaml(cfg)
    print_config(cfg)

    # Set up logging
    current_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join(cfg.logging.log_dir, "evaluation")
    log_file = os.path.join(log_dir, current_date + ".log")
    weights = cfg.model.checkpoint.split("/")[-1] if cfg.model.checkpoint else "base"
    model_dir = cfg.model.name + ("-thinking" if cfg.model.thinking else "")
    agent_dir = cfg.agent.name + ("" if cfg.user.name == "patient" else f"-{cfg.user.name}")
    results_dir = os.path.join(
        cfg.logging.results_dir, cfg.task.name, cfg.dataset.name, "evaluation", model_dir, agent_dir,
        f"k_{cfg.general.k}", weights, current_date
    )
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    logging_handlers = [logging.FileHandler(log_file, encoding="utf-8")]
    if cfg.logging.console:
        logging_handlers.append(logging.StreamHandler())

    logging.basicConfig(
        handlers=logging_handlers,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        force=True,
    )
    logging.getLogger("LiteLLM").setLevel(logging.WARNING)
    print(f"Logging to: {log_file}")
    logging.info("Logging to: %s", log_file)

    # Save config near outputs before the run starts
    with open(os.path.join(results_dir, "config.yaml"), "w", encoding="utf-8") as outfile:
        outfile.write(cfg_yaml)

    # Run evaluation pipeline
    pipeline = hydra.utils.instantiate(cfg.evaluation)
    pipeline.run(results_dir=results_dir, max_workers=cfg.general.max_workers)

    logging.info("Evaluation results saved to: %s", results_dir)
    print(f"Evaluation results saved to: {results_dir}")


if __name__ == "__main__":
    main()
