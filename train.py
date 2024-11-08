import argparse
import torch
from torch.utils.data import DataLoader
from models.LMSCNet import LMSCNet
from data.dataset import LMSCNetDataLoader
from train.trainer import LMSCNetTrainer
from utils.data import seed
from utils import config_tools
from semantic_kitti_pytorch.data.datasets import KITTI_METADATA
from semantic_kitti_pytorch.data.datasets import SemanticKITTI_Completion, VOXEL_DIMS


def main(args):

    # 0. Set seed
    seed.seed_all(0)

    # 1. Load config
    config_path = args.config_file
    LMSCNet_yaml = config_tools.load_yaml(config_path)
    TRAIN_CFG = LMSCNet_yaml['TRAIN']

    # 2. Load dataset
    DATASET_ROOT = args.dataset_root
    if DATASET_ROOT is None:
        DATASET_ROOT = LMSCNet_yaml['DATASET']['root_dir']

    # 3. Load dataloaders
    # dataloaders_dict = {
        # 'train': LMSCNetDataLoader(train_dataset, batch_size=TRAIN_CFG['batch_size'], shuffle=True),
        # 'valid': None,
    # }

    train_dataset = SemanticKITTI_Completion(DATASET_ROOT, phase='train')
    val_dataset = SemanticKITTI_Completion(DATASET_ROOT, phase='valid')
    dataloader_dict = {
        'train': DataLoader(train_dataset, batch_size=TRAIN_CFG['batch_size'], shuffle=True),
        'valid': DataLoader(val_dataset, batch_size=TRAIN_CFG['batch_size'], shuffle=False),
    }

    # 4. Initialize model
    network_kitti = LMSCNet(n_classes=20, voxel_dim=VOXEL_DIMS)
    # model.apply(model.weights_initializer)
    print(network_kitti)

    # 5. Train model
    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')
    device = 'cpu'
    trainer = LMSCNetTrainer(model=network_kitti,
                             dataloader_dict=dataloader_dict,
                             TRAIN_CFG=TRAIN_CFG,
                             device=device)
    trainer.train()

    # 6. Get best record
    print("Best record: ", trainer.get_best_record())

    ##### End of main #####


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
