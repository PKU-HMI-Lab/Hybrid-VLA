import functools
import json
import os
import pathlib
import sys

import hydra
import torch
import wandb
from hydra.utils import call, instantiate
from omegaconf import OmegaConf
from termcolor import colored

from lift3d.envs import Evaluator
from lift3d.helpers.common import Logger, WandBLogger, set_seed
from lift3d.helpers.pytorch import AverageMeter, log_params_to_file


@hydra.main(version_base=None, config_path="../config", config_name="train")
def main(config):
    #############################
    # log important information #
    #############################
    Logger.log_info(
        f'Running {colored(pathlib.Path(__file__).absolute(), "red")} with following config:'
    )
    Logger.log_info(f'Task: {colored(config.task_name, "green")}')
    Logger.log_info(f'Dataset directory: {colored(config.dataset_dir, "green")}')
    Logger.log_info(f'Image size: {colored(config.image_size, "green")}')
    Logger.log_info(
        f'WandB: Project {colored(config.wandb.project, "green")}; '
        f'Group {colored(config.wandb.group, "green")}; '
        f'Name {colored(config.wandb.name, "green")}; '
        f'Notes {colored(config.wandb.notes, "green")}; '
        f'Mode {colored(config.wandb.mode, "green")}'
    )
    Logger.log_info(
        f'Agent: {colored(config.agent.name, color="green")}\n{json.dumps(OmegaConf.to_container(config.agent, resolve=True), indent=4)}'
    )
    Logger.log_info(
        f'Benchmark: {colored(config.benchmark.name, color="green")}\n{json.dumps(OmegaConf.to_container(config.benchmark, resolve=True), indent=4)}'
    )
    Logger.print_seperator()

    ############
    # set seed #
    ############
    set_seed(config.seed)

    ################
    # wandb logger #
    ################
    wandb_logger = WandBLogger(
        config=config.wandb,
        hyperparameters=OmegaConf.to_container(config, resolve=True),
    )
    wandb_logger.run.define_metric("train_interation/*", step_metric="iteration_step")
    wandb_logger.run.define_metric("train_epoch/*", step_metric="epoch_step")
    wandb_logger.run.define_metric("validation/*", step_metric="epoch_step")

    ##########################
    # datasets and evaluator #
    ##########################
    train_dataset = instantiate(
        config=config.benchmark.dataset_instantiate_config,
        data_dir=config.dataset_dir,
        split="train",
    )
    valid_dataset = instantiate(
        config=config.benchmark.dataset_instantiate_config,
        data_dir=config.dataset_dir,
        split="validation",
    )
    evaluator: Evaluator = instantiate(
        config=config.benchmark.evaluator_instantiate_config,
        task_name=config.task_name,
    )

    ###############
    # dataloaders #
    ###############
    DataLoaderConstuctor = functools.partial(
        torch.utils.data.DataLoader,
        batch_size=config.dataloader.batch_size,
        num_workers=config.dataloader.num_workers,
        shuffle=config.dataloader.shuffle,
        pin_memory=config.dataloader.pin_memory,
    )
    train_loader = DataLoaderConstuctor(train_dataset)
    valid_loader = DataLoaderConstuctor(valid_dataset)
    _, _, sample_robot_state, _, sample_action, _ = next(iter(train_loader))
    robot_state_dim = sample_robot_state.size(-1)
    action_dim = sample_action.size(-1)
    Logger.log_info(f'Robot state dim: {colored(robot_state_dim, "red")}')
    Logger.log_info(f'Action dim: {colored(action_dim, "red")}')

    #########
    # Model #
    #########
    model = instantiate(
        config=config.agent.instantiate_config,
        robot_state_dim=robot_state_dim,
        action_dim=action_dim,
    )
    model = model.to(config.device)

    #############
    # Optimizer #
    #############
    optimizer: torch.optim.Optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=config.train.learning_rate,
    )
    local_run_output_dir = pathlib.Path(
        hydra.core.hydra_config.HydraConfig.get()["runtime"]["output_dir"]
    )
    log_params_to_file(model, local_run_output_dir / "train_model_params.txt", True)
    log_params_to_file(
        model, local_run_output_dir / "train_model_params_freeze.txt", False
    )

    ###########################
    # Learning rate scheduler #
    ###########################
    scheduler: torch.optim.lr_scheduler.LRScheduler = instantiate(
        config=config.train.scheduler_instantiate_config,
        optimizer=optimizer,
    )

    ############
    # Training #
    ############
    max_success, max_rewards, best_success = 0.0, 0.0, -1.0
    loss_train = AverageMeter()
    for cur_epoch in range(config.train.num_epochs):
        epoch_logging_info = {"epoch_step": cur_epoch + 1}
        model.train()
        for cur_iter, (
            images,
            point_clouds,
            robot_states,
            raw_states,
            actions,
            texts,
        ) in enumerate(train_loader):
            iteration_info = {}

            # training iteration
            images = images.to(config.device)
            point_clouds = point_clouds.to(config.device)
            robot_states = robot_states.to(config.device)
            actions = actions.to(config.device, non_blocking=True)
            preds = model(images, point_clouds, robot_states, texts)
            loss_result = call(config.benchmark.loss_func, preds, actions)

            # loss verbose
            if isinstance(loss_result, tuple):
                loss = loss_result[0]
                loss_dict = loss_result[1]
                for key, value in loss_dict.items():
                    iteration_info[f"train_interation/{key}"] = value
            else:
                loss = loss_result

            # step
            optimizer.zero_grad()
            loss.backward()

            # clip gradient
            if config.train.clip_grad_value > 0.0:
                torch.nn.utils.clip_grad_value_(
                    model.parameters(), config.train.clip_grad_value
                )

            # update model
            optimizer.step()
            loss_train.update(loss.item())

            # training iteration log
            iteration_info.update(
                {
                    "iteration_step": cur_epoch * len(train_loader) + cur_iter + 1,
                    "train_interation/epoch": cur_epoch,
                    "train_interation/loss": loss.item(),
                    "train_interation/learning_rate": scheduler.get_last_lr()[0],
                }
            )
            wandb_logger.log(iteration_info)

        # update lr
        scheduler.step()

        # training epoch log
        epoch_logging_info.update({"train_epoch/epoch_loss": loss_train.avg})
        Logger.log_info(f"[train] epoch={cur_epoch}, loss={loss_train.avg}")

        # Validation
        periodic_validation = (cur_epoch + 1 > config.evaluation.num_skip_epochs) and (
            (cur_epoch + 1) % config.evaluation.validation_frequency_epochs == 0
        )
        last_epoch = (cur_epoch + 1) == config.train.num_epochs
        if periodic_validation or last_epoch:
            model.eval()

            # validation loss
            loss_val = AverageMeter()
            for cur_iter, (
                images,
                point_clouds,
                robot_states,
                raw_states,
                actions,
                texts,
            ) in enumerate(valid_loader):
                images = images.to(config.device)
                point_clouds = point_clouds.to(config.device)
                robot_states = robot_states.to(config.device)
                actions = actions.to(config.device, non_blocking=True)
                with torch.no_grad():
                    preds = model(images, point_clouds, robot_states, texts)
                loss_result = call(config.benchmark.loss_func, preds, actions)

                if isinstance(loss_result, tuple):
                    loss_val.update(loss_result[0].item(), actions.shape[0])
                    loss_dict = loss_result[1]
                    for key, value in loss_dict.items():
                        epoch_logging_info[f"validation/{key}"] = value
                else:
                    loss_val.update(loss_result.item(), actions.shape[0])

            # validation success and rewards
            avg_success, avg_rewards = evaluator.evaluate(
                config.evaluation.validation_trajs_num, model
            )
            max_success, max_rewards = max(max_success, avg_success), max(
                max_rewards, avg_rewards
            )
            epoch_logging_info.update(
                {
                    "validation/epoch": cur_epoch,
                    "validation/loss": loss_val.avg,
                    "validation/success": avg_success,
                    "validation/rewards": avg_rewards,
                    "validation/max_success": max_success,
                    "validation/max_rewards": max_rewards,
                    "validation/video_steps": wandb.Video(
                        evaluator.env.get_frames().transpose(0, 3, 1, 2), fps=30
                    ),
                }
            )
            evaluator.callback(epoch_logging_info)
            Logger.log_info(
                f"[validation] epoch={cur_epoch}, "
                f"validation_loss={loss_val.avg}, "
                f"avg_success={avg_success}, "
                f"avg_rewards={avg_rewards}, "
                f"max_success={max_success}, "
                f"max_rewards={max_rewards}"
            )

            # save best model
            if config.evaluation.save_best_model and avg_success > best_success:
                best_success = avg_success
                model_path = os.path.join(
                    hydra.core.hydra_config.HydraConfig.get()["runtime"]["output_dir"],
                    "best_model.pth",
                )
                torch.save(model.state_dict(), model_path)
                Logger.log_info(f'Save best model to {colored(model_path, "red")}')
                with open(
                    os.path.join(
                        hydra.core.hydra_config.HydraConfig.get()["runtime"][
                            "output_dir"
                        ],
                        "best_model.json",
                    ),
                    "w",
                ) as f:
                    model_info = {
                        "epoch": cur_epoch,
                        "loss": loss_val.avg,
                        "avg_success": avg_success,
                        "avg_rewards": avg_rewards,
                    }
                    json.dump(model_info, f, indent=4)

        # log epoch info
        wandb_logger.log(epoch_logging_info)

    Logger.log_ok("Training Finished!")


if __name__ == "__main__":
    main()
