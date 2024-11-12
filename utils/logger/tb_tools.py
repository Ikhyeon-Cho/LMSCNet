from datetime import datetime

from torch.utils.tensorboard import SummaryWriter


class TensorBoardLogger:
    def __init__(self, log_dir: str):
        self.writer = SummaryWriter(log_dir=log_dir)
