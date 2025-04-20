import copy
import json
import os
import pathlib
import sys

import torch
from termcolor import colored

from lift3d.helpers.common import Logger


def get_optimizer_groups(model, default_wd):
    param_group_names, param_group_vars = dict(), dict()
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "token" in n:
            name_apx = "t"
            wd_val = 0.0
        elif "pos_embed" in n:
            name_apx = "p"
            wd_val = 0.0
        elif "bn" in n or "ln" in n or "norm" in n:
            name_apx = "n"
            wd_val = 0.0
        elif "bias" in n:
            name_apx = "b"
            wd_val = 0.0
        else:
            name_apx = "w"
            wd_val = default_wd

        param_group = f"wd:{name_apx}"
        if param_group not in param_group_names:
            item = {"params": [], "weight_decay": wd_val}
            param_group_names[param_group] = copy.deepcopy(item)
            param_group_vars[param_group] = copy.deepcopy(item)
        param_group_names[param_group]["params"].append(n)
        param_group_vars[param_group]["params"].append(p)

    param_list = list(param_group_vars.values())

    param_group_str = colored(
        json.dumps(param_group_names, sort_keys=True, indent=2), "blue"
    )
    print("Parameter groups:\n" + param_group_str)

    return param_list


class Optimizers(object):
    @staticmethod
    def get_constant_scheduler(optimizer: torch.optim.Optimizer, lr: float):
        lambda_func = lambda _: 1.0
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda_func)
        return scheduler

    @staticmethod
    def get_warmup_cosine_annealing_scheduler(
        optimizer: torch.optim.Optimizer,
        num_warmup_epochs: int,
        num_epochs: int,
        warmup_factor: float = 0.1,
    ):
        scheduler_warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer=optimizer,
            start_factor=warmup_factor,
            total_iters=num_warmup_epochs,
        )
        scheduler_train = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer,
            T_max=(num_epochs - num_warmup_epochs),
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer=optimizer,
            schedulers=[scheduler_warmup, scheduler_train],
            milestones=[num_warmup_epochs],
        )
        return scheduler


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n

    @property
    def avg(self):
        return self.sum / self.count


def log_params_to_file(model, filename, requires_grad):
    save_dir = pathlib.Path(filename).parent
    save_dir.mkdir(parents=True, exist_ok=True)

    trainable_params = []
    for name, p in model.named_parameters():
        if p.requires_grad == requires_grad:
            trainable_params.append(name)

    with open(filename, "w") as f:
        for param in trainable_params:
            f.write(f"{param}\n")

    Logger.log_info(
        f"{'Trainable' if requires_grad else 'Freezed'} parameters saved to {filename}"
    )
