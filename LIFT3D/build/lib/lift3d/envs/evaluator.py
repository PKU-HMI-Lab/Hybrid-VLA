from abc import ABC, abstractmethod


class Evaluator(ABC):
    @abstractmethod
    def evaluate(self, num_episodes, policy):
        pass

    def callback(self, logging_info: dict):
        pass

    def callback_verbose(self, wandb_logger):
        pass
