"""
Author: Ikhyeon Cho
Link: https://github.com/Ikhyeon-Cho
File: console.py
Date: 2024/11/12 22:50
"""

import logging
import os


def Logger(log_dir: str, filename: str = 'logs.log'):
    """Get logger instance with setup"""

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
    logger = Logger(log_dir='test', filename='train.log')
    logger.info('Hello, World!')
