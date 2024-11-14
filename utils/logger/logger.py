"""
Author: Ikhyeon Cho
Link: https://github.com/Ikhyeon-Cho
File: tensorboard.py
Date: 2024/11/12 22:50
"""

from torch.utils.tensorboard import SummaryWriter
from utils.logger.time import now
import logging
import os
import numpy as np
import torch


def Tensorboard(log_dir: str):
    """Get TensorBoard logger instance"""
    return SummaryWriter(log_dir=log_dir)


class TensorboardLogger:
    def __init__(self, log_dir: str, ns: str = None):
        self.tb_writer = SummaryWriter(log_dir=log_dir)
        self.namespace = ns

    def log_batch_loss(self, loss: dict, steps: list = None):
        """Log batch losses to tensorboard.

        Args:
            batch_loss: Dictionary of batch losses {loss_name: loss_values}
            steps: List of steps corresponding to batch losses
        """
        if not isinstance(loss, dict):
            raise TypeError(
                'TensorboardLogger: Batch loss must be a dictionary')

        if steps is None:
            for loss_name, loss_vals in loss.items():
                steps = list(range(len(loss_vals)))
                break

        for loss_name, loss_vals in loss.items():
            loss_vals = torch.stack(loss_vals).cpu().numpy()
            steps = np.array(steps)

            # Handle tensorboard namespace
            tag = f'Batch Loss/{loss_name}'
            if self.namespace:
                tag = f'{self.namespace} {tag}'

            loss_tracked = zip(loss_vals, steps)
            for val, step in loss_tracked:
                self.tb_writer.add_scalar(tag, val, step)

    def log_epoch_loss(self, loss: dict, epoch: int):
        """Write mean epoch losses to tensorboard.

        Args:
            loss: Dictionary of epoch losses {loss_name: avg_loss}
            epoch: Current epoch number
        """
        for loss_name, avg_loss in loss.items():
            tag = f'Epoch Loss/{loss_name}'
            if self.namespace:
                tag = f'{self.namespace} {tag}'
            self.tb_writer.add_scalar(tag, avg_loss, epoch)

    def log_lr(self, lr: float, global_step: int):
        """Log learning rate to tensorboard.

        Args:
            lr: Learning rate
            global_step: Global step number
        """
        self.tb_writer.add_scalar('Learning Rate', lr, global_step)


class LossTracker:
    def __init__(self, log_dir: str, ns: str = None):
        # tracked_losses: {loss_name: [loss_values]}
        # steps: [steps_corresponding_to_loss_values]
        self.losses_tracked: dict[str, list] = {}
        self.steps: list[int] = []
        self.tb_writer = SummaryWriter(log_dir=log_dir)
        self.namespace = ns

    def append(self, loss_to_add: dict, step: int = None):
        """Add losses for batch logging.

        Args:
            loss: Dictionary of losses {loss_name: loss_value}
            step: Global step number
        """
        if not isinstance(loss_to_add, dict):
            raise ValueError('LossTrackerTB: Loss must be a dictionary')

        # Append losses
        for loss_name, loss_val in loss_to_add.items():
            if loss_name not in self.losses_tracked:
                self.losses_tracked[loss_name] = []
            # Store tensor values as list
            self.losses_tracked[loss_name].append(loss_val.detach())

        # Append steps
        if step is not None:
            self.steps.append(step)

    def batch_loss(self) -> dict[str, list]:
        """Get all stored losses for batch logging."""
        return self.losses_tracked

    def get_steps(self) -> list[int]:
        """Get all stored steps."""
        return self.steps

    def epoch_loss(self) -> dict[str, float]:
        """Get all stored losses for epoch logging."""
        epoch_losses = {}
        for loss_name, loss_vals in self.losses_tracked.items():
            # Calculate mean in one operation on GPU
            mean_loss = torch.stack(loss_vals).mean().cpu().item()
            epoch_losses[loss_name] = mean_loss

        return epoch_losses

    def reset(self):
        """Restart tracked losses and steps."""
        self.losses_tracked = {}
        self.steps = []

    def __del__(self):
        self.tb_writer.close()


def Console(log_dir: str, filename: str = 'logs.log'):
    """Get logger instance"""

    # Create logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Set output format
    formatter = logging.Formatter('%(asctime)s -- %(message)s')

    # Add console handler to print logs to console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Ensure log directory exists
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Add file handler to save logs to file
    file_handler = logging.FileHandler(os.path.join(log_dir, filename))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


if __name__ == "__main__":

    # Example code
    tb_logger = Tensorboard(log_dir='test')
    tb_logger.add_scalar(tag='Loss/batch_total',
                         scalar_value=1.0, global_step=0)

    logger = Console(log_dir='test', filename='train.log')
    logger.info('Hello, World!')
