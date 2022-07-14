import logging
import os

import numpy as np

from ocpmodels import models
from ocpmodels.common.data_parallel import ParallelCollater
from ocpmodels.common.flags import flags
from ocpmodels.common.registry import registry
from ocpmodels.common.transforms import RandomJitter
from ocpmodels.common.utils import build_config, load_config, setup_logging
from ocpmodels.datasets import TrajectoryLmdbDataset
from ocpmodels.modules.normalizer import Normalizer
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

    def OurForceTrainer(ForceTrainer):
        def __init__(
            self,
            task,
            model,
            dataset,
            optimizer,
            identifier,
            transform,
            normalizer=None,
            timestamp_id=None,
            run_dir=None,
            is_debug=False,
            is_hpo=False,
            print_every=100,
            seed=None,
            logger="tensorboard",
            local_rank=0,
            amp=False,
            cpu=False,
            slurm={},
            noddp=False,
        ):
            super().__init__(
                task=task,
                model=model,
                dataset=dataset,
                optimizer=optimizer,
                identifier=identifier,
                normalizer=normalizer,
                timestamp_id=timestamp_id,
                run_dir=run_dir,
                is_debug=is_debug,
                is_hpo=is_hpo,
                print_every=print_every,
                seed=seed,
                logger=logger,
                local_rank=local_rank,
                amp=amp,
                cpu=cpu,
                name="s2ef",
                slurm=slurm,
                noddp=noddp,
            )
            self.transform = transform

        def load_datasets(self):
            self.parallel_collater = ParallelCollater(
                0 if self.cpu else 1,
                self.config["model_attributes"].get("otf_graph", False),
            )

            self.train_loader = self.val_loader = self.test_loader = None

            if self.config.get("dataset", None):
                self.train_dataset = registry.get_dataset_class(
                    self.config["task"]["dataset"]
                )(self.config["dataset"], self.transform)
                self.train_sampler = self.get_sampler(
                    self.train_dataset,
                    self.config["optim"]["batch_size"],
                    shuffle=True,
                )
                self.train_loader = self.get_dataloader(
                    self.train_dataset,
                    self.train_sampler,
                )

                if self.config.get("val_dataset", None):
                    self.val_dataset = registry.get_dataset_class(
                        self.config["task"]["dataset"]
                    )(self.config["val_dataset"], self.transform)
                    self.val_sampler = self.get_sampler(
                        self.val_dataset,
                        self.config["optim"].get(
                            "eval_batch_size",
                            self.config["optim"]["batch_size"],
                        ),
                        shuffle=False,
                    )
                    self.val_loader = self.get_dataloader(
                        self.val_dataset,
                        self.val_sampler,
                    )

                if self.config.get("test_dataset", None):
                    self.test_dataset = registry.get_dataset_class(
                        self.config["task"]["dataset"]
                    )(self.config["test_dataset"], self.transform)
                    self.test_sampler = self.get_sampler(
                        self.test_dataset,
                        self.config["optim"].get(
                            "eval_batch_size",
                            self.config["optim"]["batch_size"],
                        ),
                        shuffle=False,
                    )
                    self.test_loader = self.get_dataloader(
                        self.test_dataset,
                        self.test_sampler,
                    )

            # Normalizer for the dataset.
            # Compute mean, std of training set labels.
            self.normalizers = {}
            if self.normalizer.get("normalize_labels", False):
                if "target_mean" in self.normalizer:
                    self.normalizers["target"] = Normalizer(
                        mean=self.normalizer["target_mean"],
                        std=self.normalizer["target_std"],
                        device=self.device,
                    )
                else:
                    self.normalizers["target"] = Normalizer(
                        tensor=self.train_loader.dataset.data.y[
                            self.train_loader.dataset.__indices__
                        ],
                        device=self.device,
                    )

    teacher_pretrained_trainer = OurForceTrainer(
        task=teacher_config["task"],
        model=teacher_config["model"],
        dataset=dataset,
        transform=RandomJitter,
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
