import numpy as np

from ocpmodels import models
from ocpmodels.common import logger
from ocpmodels.common.flags import flags
from ocpmodels.common.utils import build_config, setup_logging
from ocpmodels.trainers import ForcesTrainer


def main(config):
    trainer = ForcesTrainer(
        task=config["task"],
        dataset=config["dataset"],
        optimizer=config["optim"],
        model=config["model"],
        identifier=config["identifier"],
        timestamp_id=config.get("timestamp_id", None),
        run_dir=config.get("run_dir", "./"),
        is_debug=config.get("is_debug", False),
        print_every=config.get("print_every", 10),
        seed=config.get("seed", 0),
        logger=config.get("logger", "tensorboard"),
        local_rank=config["local_rank"],
        amp=config.get("amp", False),
        cpu=config.get("cpu", False),
        slurm=config.get("slurm", {}),
        noddp=config.get("noddp", False),
    )

    trainer.load_checkpoint(checkpoint_path=config["checkpoint"])
    trainer.run_relaxations()


if __name__ == "__main__":
    setup_logging()

    parser = flags.get_parser()
    args, override_args = parser.parse_known_args()
    config = build_config(args, override_args)
    print(config)

    main(config)
