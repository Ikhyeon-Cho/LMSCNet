"""
Author: Ikhyeon Cho
Link: https://github.com/Ikhyeon-Cho
File: config_tools.py
Date: 2024/11/2 18:50
"""

import os
import yaml


def load_yaml(yaml_path):
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"The file {yaml_path} does not exist.")
    return yaml.safe_load(open(yaml_path, 'r'))


if __name__ == "__main__":

    # Example usage
    train_cfg = load_yaml("configs/LMSCNet.yaml")
    print("Testing config loader:")
    print("-"*10)
    print(train_cfg)
