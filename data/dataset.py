from torch.utils.data import DataLoader


class LMSCNetDataLoader:
    def __init__(self, dataset, batch_size, shuffle):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

    def _collate_fn(self, batch):
        return batch
