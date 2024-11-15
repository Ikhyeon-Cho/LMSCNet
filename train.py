"""
Author: Ikhyeon Cho
Link: https://github.com/Ikhyeon-Cho
File: train.py
Date: 2024/11/2 18:50

Unofficial re-implementation of LMSCNet.
Reference: https://github.com/astra-vision/LMSCNet
"""

import argparse
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from data.dataloader import LMSCNetDataset, VOXEL_DIMS
from models.LMSCNet import LMSCNet
from train.trainer import LMSCNetTrainer
from train.optimizer import Optimizer
from utils.system import time
from utils.pytorch import device_tools, config_tools, seed
import os


def log_dirname(logger_yaml: dict):
    log_time = time.now(timezone=logger_yaml['timezone'])
    log_dirname = os.path.join(
        logger_yaml['log_dir'], f'LMSCNet_SemanticKITTI_{log_time}')
    return log_dirname


def main(args):
    config_path = args.config_file
    print(f'============ Training routine: "{config_path}" ============')

    # 0. Set seed, device
    print('=> Setting seed and device...')
    seed.seed_all(0)
    device = device_tools.get_device()

    # 1. Load configs
    print('=> Loading configs...')
    LMSCNet_yaml = config_tools.load_yaml(config_path)
    MODEL_CFG = LMSCNet_yaml['MODEL']
    TRAIN_CFG = LMSCNet_yaml['TRAIN']
    LOSS_CFG = LMSCNet_yaml['LOSS']
    LOGGER_CFG = LMSCNet_yaml['LOGGER']

    # 2. Load dataset
    print('=> Loading dataset...')
    DATASET_ROOT = args.dataset_root
    if DATASET_ROOT is None:
        DATASET_ROOT = LMSCNet_yaml['DATASET']['root_dir']

    train_dataset = LMSCNetDataset(DATASET_ROOT, phase='train')
    val_dataset = LMSCNetDataset(DATASET_ROOT, phase='valid')
    dataloader_dict = {
        'train': DataLoader(train_dataset,
                            batch_size=TRAIN_CFG['batch_size'],
                            shuffle=True,
                            num_workers=TRAIN_CFG['num_workers']),
        'valid': DataLoader(val_dataset,
                            batch_size=TRAIN_CFG['batch_size'],
                            shuffle=False,
                            num_workers=TRAIN_CFG['num_workers']),
    }

    # 3. Load model
    print('=> Loading network architecture...')
    model = LMSCNet(input_dims=VOXEL_DIMS,
                    num_classes=MODEL_CFG['num_classes'])
    criterion = CrossEntropyLoss()

    # 4. Train model
    print('=> Loading optimizer...')
    optimizer = Optimizer(model=model,
                          config=TRAIN_CFG)

    print('=> No checkpoint. Initializing model from scratch')
    trainer = LMSCNetTrainer(model=model,
                             data=dataloader_dict,
                             crit=criterion,
                             optim=optimizer,
                             cfg=TRAIN_CFG,
                             device=device,
                             log_dir=log_dirname(LOGGER_CFG))

    print('=> Training model...')
    trainer.train()
    print('=> Training routine completed...')

    # 5. Get best record
    # print("Best record: ", trainer.get_best_record())


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='LMSCNet training')
    parser.add_argument(
        '--cfg',
        dest='config_file',
        default='configs/LMSCNet.yaml',
        metavar='FILE',
        help='path to config file',
        type=str,
    )
    parser.add_argument(
        '--dset_root',
        dest='dataset_root',
        default=None,
        metavar='DATASET',
        help='path to dataset root folder',
    )
    args = parser.parse_args()

    main(args)
