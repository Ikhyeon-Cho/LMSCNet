"""
Author: Ikhyeon Cho
Link: https://github.com/Ikhyeon-Cho
File: train.py
Date: 2024/11/2 18:50

Re-implementation of LMSCNet.
Reference: https://github.com/astra-vision/LMSCNet
"""

import argparse
from torch.utils.data import DataLoader
from data.dataloader import LMSCNetDataset, VOXEL_DIMS
from models.LMSCNet import LMSCNet, LMSCNetLoss
from train.trainer import LMSCNetTrainer
from train.optimizer import LMSCNetOptimizer
from utils.pytorch import gpu_tools, seed
from utils.yaml import config_tools


def main(args):

    # 0. Set seed, device
    seed.seed_all(0)
    device = gpu_tools.get_device()

    # 1. Load config
    config_path = args.config_file
    LMSCNet_yaml = config_tools.load_yaml(config_path)
    OPTIMIZER_CFG = LMSCNet_yaml['OPTIMIZER']
    TRAIN_CFG = LMSCNet_yaml['TRAIN']

    # 2. Load dataset
    DATASET_ROOT = args.dataset_root
    if DATASET_ROOT is None:
        DATASET_ROOT = LMSCNet_yaml['DATASET']['root_dir']

    # 3. Load dataloaders
    train_dataset = LMSCNetDataset(DATASET_ROOT, phase='train')
    val_dataset = LMSCNetDataset(DATASET_ROOT, phase='valid')

    dataloader_dict = {
        'train': DataLoader(train_dataset,
                            batch_size=TRAIN_CFG['batch_size'],
                            shuffle=True,
                            num_workers=2),
        'valid': DataLoader(val_dataset,
                            batch_size=TRAIN_CFG['batch_size'],
                            shuffle=False,
                            num_workers=2),
    }

    # 4. Load model
    model = LMSCNet(input_dims=VOXEL_DIMS,
                    num_classes=20)
    # model.apply(model.weights_initializer)

    # 5. Train model
    optimizer = LMSCNetOptimizer(model=model,
                                 config=OPTIMIZER_CFG)

    trainer = LMSCNetTrainer(model=model,
                             dataloader=dataloader_dict,
                             optimizer=optimizer,
                             train_cfg=TRAIN_CFG,
                             device=device)

    trainer.train()

    # 6. Get best record
    print("Best record: ", trainer.get_best_record())


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
