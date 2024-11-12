"""
Author: Ikhyeon Cho
Link: https://github.com/Ikhyeon-Cho
File: tensorboard.py
Date: 2024/11/12 22:50
"""

from torch.utils.tensorboard import SummaryWriter
from utils.logger.time import get_current_time
import logging
import os


def Tensorboard(log_dir: str, timezone: str = None):
    """Get TensorBoard logger instance"""
    return SummaryWriter(
        log_dir=os.path.join(log_dir, f'log_{get_current_time(timezone=timezone)}'))


def Console(log_dir: str, filename: str = 'logs', timezone: str = None):
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
    file_path = os.path.join(
        log_dir, f'log_{get_current_time(timezone=timezone)}')
    if not os.path.exists(file_path):
        os.makedirs(file_path)

    # Add file handler to save logs to file
    file_handler = logging.FileHandler(
        os.path.join(file_path, filename))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


if __name__ == "__main__":

    # Example code
    # tb_logger = Tensorboard(log_dir='experiments')
    # tb_logger.add_scalar(tag='Loss/batch_total',
    #                      scalar_value=1.0, global_step=0)

    logger = Console(log_dir='experiments', filename='train.log')
    logger.info('Hello, World!')
