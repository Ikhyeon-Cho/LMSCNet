"""
Author: Ikhyeon Cho
Link: https://github.com/Ikhyeon-Cho
File: optimizer.py
Date: 2024/11/13 18:50
"""

import torch.optim as optim
from models.LMSCNet import LMSCNet


class Optimizer:

    def __init__(self, model: LMSCNet, config: dict):
        self._model = model
        self._config = config

        # Public attributes
        self.optimizer = self._init_optimizer(config)
        self.scheduler = self._init_scheduler(config)

    def _init_optimizer(self, config: dict):
        """Sets up the optimizer based on configuration."""

        LEARNING_RATE = config['learning_rate']
        OPTIMIZER_TYPE = config['type']

        ADAM_BETA1 = config['Adam_beta1']
        ADAM_BETA2 = config['Adam_beta2']

        SGD_MOMENTUM = config['SGD_momentum']
        SGD_WEIGHT_DECAY = config['SGD_weight_decay']

        if OPTIMIZER_TYPE == 'Adam':
            optimizer = optim.Adam(self._model.parameters(),
                                   lr=LEARNING_RATE,
                                   betas=(ADAM_BETA1, ADAM_BETA2))
        elif OPTIMIZER_TYPE == 'SGD':
            optimizer = optim.SGD(self._model.parameters(),
                                  lr=LEARNING_RATE,
                                  momentum=SGD_MOMENTUM,
                                  weight_decay=SGD_WEIGHT_DECAY)
        else:
            raise ValueError(
                f"Unsupported optimizer type: {OPTIMIZER_TYPE}. Current options: Adam, SGD")

        return optimizer

    def _init_scheduler(self, config: dict):
        """Sets up the scheduler based on configuration."""

        SCHEDULER_TYPE = config['scheduler']

        if SCHEDULER_TYPE == 'None':
            return None

        if SCHEDULER_TYPE == 'constant':
            def lambda1(epoch): return 1
            return optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda1)

        elif SCHEDULER_TYPE == 'power_iteration':
            def lambda1(epoch): return (0.98) ** (epoch)
            return optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda1)
        else:
            raise ValueError(
                f"Unsupported scheduler type: {SCHEDULER_TYPE}. Current options: constant, power_iteration")
