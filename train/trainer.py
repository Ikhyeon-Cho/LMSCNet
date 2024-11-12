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
from utils.logger.tb_tools import TensorBoardLogger


class LMSCNetTrainer:
    def __init__(self, model: LMSCNet, dataloader: dict, optimizer: LMSCNetOptimizer, train_cfg: dict, device: str):
        # Model, Data, Loss
        self.model = model.to(device)
        self.dataloader = dataloader
        self.loss_fn = LMSCNetLoss(config=train_cfg)

        # Optimizer and training configs
        self.optimizer = optimizer.get_optimizer()
        self.scheduler = optimizer.get_scheduler()
        self.CFG_TRAIN = train_cfg
        self.device = device

        # Initialize TensorBoard writer
        current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_dir = os.path.join('runs', f'LMSCNet_{current_time}')

        self.writer = TensorBoardLogger(log_dir=self.log_dir)

        # Store best loss for model saving
        self.best_loss = float('inf')

    def train(self):
        NUM_EPOCHS = self.CFG_TRAIN['epochs']
        self.model.train()
        global_step = 0  # For tracking total training steps

        for epoch in range(1, NUM_EPOCHS+1):
            print(f"Epoch {epoch}/{NUM_EPOCHS}")
            epoch_loss = 0.0
            num_batches = 0

            for batch in self.dataloader['train']:
                batch = to_device(batch, self.device)
                b_loss = torch.tensor(0.0, device=self.device)

                # Compute individual losses
                pred = self.model(batch['occupancy'])
                loss_1_1 = self.loss_fn.CE_Loss_1_1(pred, batch['label'])
                loss_1_2 = self.loss_fn.CE_Loss_1_2(pred, batch['label_1_2'])
                loss_1_4 = self.loss_fn.CE_Loss_1_4(pred, batch['label_1_4'])
                loss_1_8 = self.loss_fn.CE_Loss_1_8(pred, batch['label_1_8'])
                b_loss = (loss_1_1 + loss_1_2 + loss_1_4 + loss_1_8) / 4

                # Update model params
                self.optimizer.zero_grad()
                b_loss.backward()
                self.optimizer.step()

                if self.scheduler is not None:
                    self.scheduler.step()
                    # Log learning rate
                    self.writer.add_scalar('Learning Rate',
                                           self.scheduler.get_last_lr()[0],
                                           global_step)

                # Log losses
                batch_loss = b_loss.item()
                epoch_loss += batch_loss
                num_batches += 1

                # Log detailed metrics to TensorBoard
                self.writer.add_scalar(
                    'Loss/batch_total', batch_loss, global_step)
                self.writer.add_scalar(
                    'Loss/batch_1_1', loss_1_1.item(), global_step)
                self.writer.add_scalar(
                    'Loss/batch_1_2', loss_1_2.item(), global_step)
                self.writer.add_scalar(
                    'Loss/batch_1_4', loss_1_4.item(), global_step)
                self.writer.add_scalar(
                    'Loss/batch_1_8', loss_1_8.item(), global_step)

                if num_batches % 10 == 0:  # Print every 10 batches
                    print(f"  Batch {num_batches}: Loss = {batch_loss:.4f}")

                global_step += 1

            # Log epoch metrics
            avg_epoch_loss = epoch_loss / num_batches
            self.writer.add_scalar('Loss/epoch', avg_epoch_loss, epoch)

            # Update best loss and save model if improved
            if avg_epoch_loss < self.best_loss:
                self.best_loss = avg_epoch_loss
                self._save_checkpoint(epoch, avg_epoch_loss)

            print(f"  Epoch {epoch} Average Loss: {avg_epoch_loss:.4f}")

    def _save_checkpoint(self, epoch, loss):
        """Save model checkpoint when best loss is achieved"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
        }
        checkpoint_path = os.path.join(self.log_dir, 'best_model.pth')
        torch.save(checkpoint, checkpoint_path)

    def validate(self):
        pass

    def get_best_record(self):
        return {
            'best_loss': self.best_loss,
            'checkpoint_dir': self.log_dir
        }

    def __del__(self):
        """Cleanup: Close the TensorBoard writer when the trainer is destroyed"""
        self.writer.close()


if __name__ == "__main__":

    from utils.yaml import config_tools

    LMSCNet_yaml = config_tools.load_yaml("configs/LMSCNet.yaml")
    TRAIN_CFG = LMSCNet_yaml['TRAIN']
    net = LMSCNet(20, (256, 256, 32))

    trainer = LMSCNetTrainer(net, None, TRAIN_CFG, 'cuda')
