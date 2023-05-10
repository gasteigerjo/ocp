import inspect

import torch.optim.lr_scheduler as lr_scheduler
from torch.optim.lr_scheduler import LambdaLR

from ocpmodels.common.utils import warmup_lr_lambda


class LRScheduler:
    """
    Learning rate scheduler class for torch.optim learning rate schedulers

    Notes:
        If no learning rate scheduler is specified in the config the default
        scheduler is warmup_lr_lambda (ocpmodels.common.utils) not no scheduler,
        this is for backward-compatibility reasons. To run without a lr scheduler
        specify scheduler: "Null" in the optim section of the config.

    Args:
        config (dict): Optim dict from the input config
        optimizer (obj): torch optim object
    """

    def __init__(self, optimizer, config):
        self.optimizer = optimizer
        self.config = config.copy()
        if "scheduler" in self.config:
            self.scheduler_type = self.config["scheduler"]
        else:
            self.scheduler_type = "LambdaLR"
            scheduler_lambda_fn = lambda x: warmup_lr_lambda(x, self.config)
            self.config["lr_lambda"] = scheduler_lambda_fn

        if self.scheduler_type != "Null":
            if self.scheduler_type == "LinearWarmupExponentialDecay":
                self.scheduler = LinearWarmupExponentialDecay
            else:
                self.scheduler = getattr(lr_scheduler, self.scheduler_type)
            scheduler_args = self.filter_kwargs(config)
            self.scheduler = self.scheduler(optimizer, **scheduler_args)

    def step(self, metrics=None, epoch=None):
        if self.scheduler_type == "Null":
            return
        if self.scheduler_type == "ReduceLROnPlateau":
            if metrics is None:
                raise Exception(
                    "Validation set required for ReduceLROnPlateau."
                )
            self.scheduler.step(metrics)
        else:
            self.scheduler.step()

    def filter_kwargs(self, config):
        # adapted from https://stackoverflow.com/questions/26515595/
        sig = inspect.signature(self.scheduler)
        filter_keys = [
            param.name
            for param in sig.parameters.values()
            if param.kind == param.POSITIONAL_OR_KEYWORD
        ]
        filter_keys.remove("optimizer")
        scheduler_args = {
            arg: self.config[arg] for arg in self.config if arg in filter_keys
        }
        return scheduler_args

    def get_lr(self):
        for group in self.optimizer.param_groups:
            return group["lr"]


class LinearWarmupExponentialDecay(LambdaLR):
    """This schedule combines a linear warmup with an exponential decay.

    Parameters
    ----------
        optimizer: Optimizer
            Optimizer instance.
        decay_steps: float
            Number of steps until learning rate reaches learning_rate*decay_rate
        decay_rate: float
            Decay rate.
        warmup_steps: int
            Total number of warmup steps of the learning rate schedule.
        staircase: bool
            If True use staircase decay and not (continous) exponential decay.
        last_step: int
            Only needed when resuming training to resume learning rate schedule at this step.
    """

    def __init__(
        self,
        optimizer,
        warmup_steps,
        decay_steps,
        decay_rate,
        staircase=False,
        last_step=-1,
        verbose=False,
    ):
        assert decay_rate <= 1

        if warmup_steps == 0:
            warmup_steps = 1

        def lr_lambda(step):
            # step starts at 0
            warmup = min(1 / warmup_steps + 1 / warmup_steps * step, 1)
            exponent = step / decay_steps
            if staircase:
                exponent = int(exponent)
            decay = decay_rate**exponent
            return warmup * decay

        super().__init__(
            optimizer, lr_lambda, last_epoch=last_step, verbose=verbose
        )
