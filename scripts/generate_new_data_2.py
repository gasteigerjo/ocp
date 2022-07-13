import logging
import os

import numpy as np

from ocpmodels import models
from ocpmodels.common import logger
from ocpmodels.common.flags import flags
from ocpmodels.common.utils import build_config, load_config, setup_logging
from ocpmodels.datasets import TrajectoryLmdbDataset
from ocpmodels.trainers import ForcesTrainer
from scripts.download_data import get_data

setup_logging()


if __name__ == "__main__":

    #################################
    # -1. download 2M dataset
    # bash: python scripts/download_data.py --task s2ef --split "200k" --get-edges --num-workers 10 --ref-energy

    dataset = [{"src": "data/s2ef/200k/train", "normalize_labels": False}]

    #################################
    # 0. get pretrained model and its configs
    # bash: wget -q https://dl.fbaipublicfiles.com/opencatalystproject/models/2021_08/s2ef/gemnet_t_direct_h512_all.pt

    teacher_config, _, _ = load_config("configs/s2ef/2M/gemnet/gemnet-oc.yml")
    print(teacher_config, dataset)

    teacher_pretrained_trainer = ForcesTrainer(
        task=teacher_config["task"],
        model=teacher_config["model"],
        dataset=dataset,
        optimizer=teacher_config["optim"],
        identifier="S2EF-gemnet-t",
    )

    teacher_pretrained_trainer.load_checkpoint(
        checkpoint_path="./gemnet_oc_base_s2ef_2M.pt"
    )

    # generate relaxation trajectories starting at rattled states
    num_relaxations = 1
    for i in range(num_relaxations):

        #################################
        # 1. load one of the given initial structures from the 2M dataset
        # Here, we first need to understand how the data is structured. For
        # that we probably need to see how the data gets loader by default,
        # i.e. backtrack the settings from "teacher_config"

        #################################
        # 2. convert to ASE format to be able to select which atoms to perturb

        #################################
        # 3. converting ASE Atoms objects back to PyTorch Geometric Data"
        #   - see preprocessing tutorial

        #################################
        # 4. simulate relaxation with new atom configuration
        teacher_pretrained_trainer.run_relaxations()

        #################################
        # 5. save the trajectory unless we want to generate samples on the fly
