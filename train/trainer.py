import torch
import torch.optim as optim
from torch.nn import CrossEntropyLoss
from models.LMSCNet import LMSCNet
from utils.data.data_processing import to_device


class LMSCNetTrainer:
    def __init__(self, model: LMSCNet, dataloader: dict, train_cfg: dict, device: str):
        self.model = model.to(device)
        self.dataloader = dataloader
        self.train_cfg = train_cfg
        self.device = device

        self.loss_fn = CrossEntropyLoss().to(device)
        # optimizer does not need to be moved to device
        self.optimizer = self._set_optimizer()
        self.scheduler = self._set_scheduler()

    def train(self):
        torch.backends.cudnn.benchmark = True
        self.model.train()
        NUM_EPOCHS = self.train_cfg['epochs']

        for epoch in range(NUM_EPOCHS):
            print(f"Epoch {epoch+1}/{NUM_EPOCHS}")

            for batch_dict in self.dataloader['train']:
                # Move batch to device
                batch_dict = to_device(batch_dict, self.device)
                # Forward pass
                preds = self.model(batch_dict['occupancy'])
                # Compute loss
                loss = self.model.compute_loss(preds, batch_dict['label'])
                # Backward pass
                self.optimizer.zero_grad()
                loss['total'].backward()
                # Update model parameters
                self.optimizer.step()

    def validate(self):
        pass

    def get_best_record(self):
        return None

    def _set_optimizer(self):
        """Sets up the optimizer based on configuration."""

        lr = self.train_cfg['learning_rate']

        OPTIMIZER_TYPE = self.train_cfg['optimizer']
        if OPTIMIZER_TYPE == 'Adam':
            optimizer = optim.Adam(params=self.model.parameters(),
                                   lr=lr,
                                   betas=(self.train_cfg['Adam_beta1'], self.train_cfg['Adam_beta2']))
        elif OPTIMIZER_TYPE == 'SGD':
            optimizer = optim.SGD(params=self.model.parameters(),
                                  lr=lr,
                                  momentum=self.train_cfg['SGD_momentum'],
                                  weight_decay=self.train_cfg['SGD_weight_decay'])
        else:
            raise ValueError(f"Unsupported optimizer type: {OPTIMIZER_TYPE}")

        return optimizer

    def _set_scheduler(self):
        """Sets up the scheduler based on configuration."""

        SCHEDULER_TYPE = self.train_cfg['scheduler']
        if SCHEDULER_TYPE is None:
            return None

        if SCHEDULER_TYPE == 'constant':
            def lambda1(epoch): return 1
            return optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda1)

        elif SCHEDULER_TYPE == 'power_iteration':
            def lambda1(epoch): return (0.98) ** (epoch)
            return optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda1)

        else:
            raise ValueError(
                f"Unsupported scheduler type: {SCHEDULER_TYPE}")


if __name__ == "__main__":

    from utils import config_tools

    LMSCNet_yaml = config_tools.load_yaml("configs/LMSCNet.yaml")
    TRAIN_CFG = LMSCNet_yaml['TRAIN']
    net = LMSCNet(20, (256, 256, 32))

    trainer = LMSCNetTrainer(net, None, TRAIN_CFG, 'cuda')
