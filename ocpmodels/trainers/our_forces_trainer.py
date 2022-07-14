"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import logging
import os
import pathlib
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch_geometric
from tqdm import tqdm

from ocpmodels.common import distutils
from ocpmodels.common.data_parallel import ParallelCollater
from ocpmodels.common.registry import registry
from ocpmodels.common.relaxation.ml_relaxation import ml_relax
from ocpmodels.common.transforms import RandomJitter
from ocpmodels.common.utils import check_traj_files
from ocpmodels.modules.evaluator import Evaluator
from ocpmodels.modules.normalizer import Normalizer
from ocpmodels.trainers import ForcesTrainer


@registry.register_trainer("our_forces")
class OurForceTrainer(ForcesTrainer):
    def load_datasets(self):
        self.parallel_collater = ParallelCollater(
            0 if self.cpu else 1,
            self.config["model_attributes"].get("otf_graph", False),
        )

        self.train_loader = self.val_loader = self.test_loader = None

        if self.config.get("dataset", None):
            transform = RandomJitter(
                {"max_translation": 0.1, "translation_probability": 0.5}
            )
            self.train_dataset = registry.get_dataset_class(
                self.config["task"]["dataset"]
            )(self.config["dataset"], transform)
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
                )(self.config["val_dataset"])
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
                )(self.config["test_dataset"])
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
