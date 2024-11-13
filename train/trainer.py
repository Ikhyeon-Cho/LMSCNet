"""
Author: Ikhyeon Cho
Link: https://github.com/Ikhyeon-Cho
File: trainer.py
Date: 2024/11/2 18:50

Re-implementation of LMSCNet. 
Reference: https://github.com/astra-vision/LMSCNet
"""

import os
import torch
from models.LMSCNet import LMSCNet, LMSCNetLoss
from train.optimizer import Optimizer
from utils.pytorch.gpu_tools import to_device
from utils.logger import logger, time


class LMSCNetTrainer:
    def __init__(self, model: LMSCNet, dataloader: dict, optimizer: Optimizer, train_cfg: dict, device: str):
        # Model, Data, Loss
        self.model = model.to(device)
        self.dataloader = dataloader
        self.loss_fn = LMSCNetLoss(config=train_cfg)

        # Optimizer and training configs
        self.optimizer = optimizer.get_optimizer()
        self.scheduler = optimizer.get_scheduler()
        self.CFG = train_cfg
        self.device = device

        # Logging directory to monitor training process
        self.global_step = 0
        self.time = time.get_current_time(timezone="Asia/Seoul")
        self.log_dir = os.path.join(self.CFG['log_dir'], f'log_{self.time}')

        # Store best loss for model saving
        self.best_loss = float('inf')

    def train(self):

        self.model.train()

        # Prepare loggers
        CONSOLE = logger.Console(self.log_dir, filename='train.log')
        TB_LOGGER = logger.Tensorboard(self.log_dir)

        # Parameters
        NUM_EPOCHS = self.CFG['epochs']
        SUMMARY_PERIOD = self.CFG['summary_period']

        # Training loop
        for epoch in range(1, NUM_EPOCHS+1):

            # To calculate epoch loss
            sum_batch_loss = 0.0
            num_batches = 0

            CONSOLE.info(
                f'=> =============== Epoch [{epoch}/{NUM_EPOCHS}] ===============')
            CONSOLE.info(
                f'=> Learning rate: {self.optimizer.param_groups[0]["lr"]}')

            for batch in self.dataloader['train']:
                batch = to_device(batch, self.device)

                # Compute training losses
                pred = self.model(batch['occupancy'])
                loss_1_1 = self.loss_fn.CE_Loss_1_1(pred, batch['label'])
                loss_1_2 = self.loss_fn.CE_Loss_1_2(pred, batch['label_1_2'])
                loss_1_4 = self.loss_fn.CE_Loss_1_4(pred, batch['label_1_4'])
                loss_1_8 = self.loss_fn.CE_Loss_1_8(pred, batch['label_1_8'])
                batch_loss = (loss_1_1 + loss_1_2 + loss_1_4 + loss_1_8) / 4

                # Update epoch loss
                sum_batch_loss += batch_loss
                num_batches += 1

                # Print losses per summary period
                if (num_batches == 1) or (num_batches % SUMMARY_PERIOD == 0):
                    loss_print = (
                        f"=> Epoch [{epoch}/{NUM_EPOCHS}] | "
                        f"Step [{num_batches}/{len(self.dataloader['train'])}] | "
                        f"lr: {self.optimizer.param_groups[0]['lr']:.6f} | "
                        f"Avg loss: {batch_loss:.4f} ("
                        f"1_1: {loss_1_1:.4f}, "
                        f"1_2: {loss_1_2:.4f}, "
                        f"1_4: {loss_1_4:.4f}, "
                        f"1_8: {loss_1_8:.4f})"
                    )
                    CONSOLE.info(loss_print)

                self.validate()

                # Update network parameters
                self.optimizer.zero_grad()
                batch_loss.backward()
                self.optimizer.step()

                # Update learning rate
                if self.scheduler is not None:
                    self.scheduler.step()
                TB_LOGGER.add_scalar('Learning Rate',
                                     self.optimizer.param_groups[0]['lr'],
                                     self.global_step)

                # Record batch loss
                TB_LOGGER.add_scalar(
                    'Loss/batch_total', batch_loss, self.global_step)
                TB_LOGGER.add_scalar(
                    'Loss/batch_1_1', loss_1_1.item(), self.global_step)
                TB_LOGGER.add_scalar(
                    'Loss/batch_1_2', loss_1_2.item(), self.global_step)
                TB_LOGGER.add_scalar(
                    'Loss/batch_1_4', loss_1_4.item(), self.global_step)
                TB_LOGGER.add_scalar(
                    'Loss/batch_1_8', loss_1_8.item(), self.global_step)
                # Update global step
                self.global_step += 1

            # Record epoch loss
            epoch_loss = sum_batch_loss / num_batches
            TB_LOGGER.add_scalar('Loss/epoch', epoch_loss, epoch)

            # Update best loss and save model if improved
            if epoch_loss < self.best_loss:
                self.best_loss = epoch_loss
                self._save_checkpoint(epoch, epoch_loss)

            CONSOLE.info(
                f"=> Epoch {epoch} Average Loss: {epoch_loss:.4f}")

        # End of training loop
        TB_LOGGER.close()

    def validate(self):
        return 1

    def get_best_record(self):
        return {
            'best_loss': self.best_loss,
            'checkpoint_dir': self.log_dir
        }

    def _save_checkpoint(self, epoch, loss):
        """Save model checkpoint when best loss is achieved"""

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'loss': loss,
            'config': self.CFG
        }

        checkpoint_path = os.path.join(
            self.log_dir, f'best_model_{self.time}.pth')
        torch.save(checkpoint, checkpoint_path)


if __name__ == "__main__":

    from utils.yaml import config_tools

    LMSCNet_yaml = config_tools.load_yaml("configs/LMSCNet.yaml")
    TRAIN_CFG = LMSCNet_yaml['TRAIN']
    net = LMSCNet(20, (256, 256, 32))

    trainer = LMSCNetTrainer(net, None, TRAIN_CFG, 'cuda')
