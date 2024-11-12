"""
Author: Ikhyeon Cho
Link: https://github.com/Ikhyeon-Cho
File: trainer.py
Date: 2024/11/2 18:50

Re-implementation of LMSCNet. 
Reference: https://github.com/astra-vision/LMSCNet
"""

import torch
import torch.optim as optim
from models.LMSCNet import LMSCNet, LMSCNetLoss
from train.optimizer import LMSCNetOptimizer
from utils.pytorch.gpu_tools import to_device
import os
from utils.logger import logger
from utils.logger.time import get_current_time


class LMSCNetTrainer:
    def __init__(self, model: LMSCNet, dataloader: dict, optimizer: LMSCNetOptimizer, train_cfg: dict, device: str):
        # Model, Data, Loss
        self.model = model.to(device)
        self.dataloader = dataloader
        self.loss_fn = LMSCNetLoss(config=train_cfg)

        # Optimizer and training configs
        self.optimizer = optimizer.get_optimizer()
        self.scheduler = optimizer.get_scheduler()
        self.CFG = train_cfg
        self.device = device

        # Logger to monitor training process
        self.log_dir = self.CFG['log_dir']
        self.global_step = 0

        # Store best loss for model saving
        self.best_loss = float('inf')

    def train(self):

        self.model.train()

        # Prepare loggers
        CONSOLE = logger.Console(self.log_dir, filename='train.log')
        TB_LOGGER = logger.Tensorboard(self.log_dir)

        # Parameters
        NUM_EPOCHS = self.CFG['epochs']

        # Training loop
        for epoch in range(1, NUM_EPOCHS+1):

            epoch_loss = 0.0
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

                # Record losses
                loss_print = (
                    f"=> Epoch [{epoch}/{NUM_EPOCHS}], Iteration [{num_batches+1}/{len(self.dataloader['train'])}], "
                    f"Learn Rate: {self.optimizer.param_groups[0]['lr']}, Train Losses: {loss_1_1:.4f}, {loss_1_2:.4f}, {loss_1_4:.4f}, {loss_1_8:.4f}"
                )
                CONSOLE.info(loss_print)

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

                # Update epoch loss
                epoch_loss += batch_loss
                num_batches += 1

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

                if num_batches % 10 == 0:  # Print every 10 batches
                    CONSOLE.info(
                        f"  Batch {num_batches}: Loss = {batch_loss:.4f}")

                # Update global step
                self.global_step += 1

            # Record epoch loss
            avg_epoch_loss = epoch_loss / num_batches
            TB_LOGGER.add_scalar('Loss/epoch', avg_epoch_loss, epoch)

            # Update best loss and save model if improved
            if avg_epoch_loss < self.best_loss:
                self.best_loss = avg_epoch_loss
                self._save_checkpoint(epoch, avg_epoch_loss)

            CONSOLE.info(
                f"  Epoch {epoch} Average Loss: {avg_epoch_loss:.4f}")

        # End of training loop
        TB_LOGGER.close()

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
            self.CFG['checkpoint_dir'], f'best_model_{get_current_time(timezone="Asia/Seoul")}.pth')
        torch.save(checkpoint, checkpoint_path)

    def validate(self):
        pass

    def get_best_record(self):
        return {
            'best_loss': self.best_loss,
            'checkpoint_dir': self.log_dir
        }

    def __del__(self):
        pass


if __name__ == "__main__":

    from utils.yaml import config_tools

    LMSCNet_yaml = config_tools.load_yaml("configs/LMSCNet.yaml")
    TRAIN_CFG = LMSCNet_yaml['TRAIN']
    net = LMSCNet(20, (256, 256, 32))

    trainer = LMSCNetTrainer(net, None, TRAIN_CFG, 'cuda')
