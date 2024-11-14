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
from models.LMSCNet import LMSCNet, LMSCNetLoss, LMSCNetMetrics
from train.optimizer import Optimizer
from utils.pytorch.gpu_tools import to_device
from utils.logger import logger, time


class LMSCNetTrainer:
    def __init__(self, model: LMSCNet, dataloader: dict, optimizer: Optimizer, train_cfg: dict, device: str):
        # Model, Data, Loss, Optimizer, Scheduler
        self.model = model.to(device)
        self.dataloader = dataloader
        self.loss_fn = LMSCNetLoss(config=train_cfg)
        self.optimizer = optimizer.get_optimizer()
        self.scheduler = optimizer.get_scheduler()

        # Training configs
        self.CFG = train_cfg
        self.device = device

        # Logging directory to monitor training process
        self.global_step = 0
        self.time = time.now(timezone="Asia/Seoul")
        self.log_dir = os.path.join(self.CFG['log_dir'], f'log_{self.time}')
        self.CONSOLE = logger.Console(self.log_dir, filename='train.log')
        self.TRAIN_LOGGER = logger.TensorboardLogger(self.log_dir, ns='Train')
        self.metrics = LMSCNetMetrics(num_classes=20)

        # Store best loss for model saving
        self.best_loss = float('inf')

    def train(self):

        # Parameters
        NUM_EPOCHS = self.CFG['epochs']
        SUMMARY_PERIOD = self.CFG['summary_period']
        SCHEDULER_PERIOD = self.CFG['scheduler_frequency']

        # Training loop
        for epoch in range(1, NUM_EPOCHS+1):
            LOSS_TRACKER = logger.LossTracker(self.log_dir, ns='Train')

            # Print learning rate of current epoch
            learning_rate = self.optimizer.param_groups[0]['lr']
            self.CONSOLE.info(f'=> ====== Epoch [{epoch}/{NUM_EPOCHS}] ======')
            self.CONSOLE.info(f'=> Learning rate: {learning_rate}')
            self.TRAIN_LOGGER.log_lr(learning_rate, self.global_step)

            # Batch loop
            self.model.train()
            num_batches = 0
            for batch in self.dataloader['train']:
                batch = to_device(batch, self.device)

                # Compute training losses
                pred = self.model(batch['occupancy'])
                loss_1_1 = self.loss_fn.CE_Loss_1_1(pred, batch['label'])
                loss_1_2 = self.loss_fn.CE_Loss_1_2(pred, batch['label_1_2'])
                loss_1_4 = self.loss_fn.CE_Loss_1_4(pred, batch['label_1_4'])
                loss_1_8 = self.loss_fn.CE_Loss_1_8(pred, batch['label_1_8'])
                batch_loss = (loss_1_1 + loss_1_2 + loss_1_4 + loss_1_8) / 4

                # Compute performance metrics (delayed update)
                self.metrics.add_pred_1_1(pred, batch['label'])
                self.metrics.add_pred_1_2(pred, batch['label_1_2'])
                self.metrics.add_pred_1_4(pred, batch['label_1_4'])
                self.metrics.add_pred_1_8(pred, batch['label_1_8'])

                # Update network parameters
                self.optimizer.zero_grad()
                batch_loss.backward()
                self.optimizer.step()

                # Update batch counter
                num_batches += 1

                # Print losses per period
                if (num_batches == 1) or (num_batches % SUMMARY_PERIOD == 0):
                    loss_print = (
                        f"=> Epoch [{epoch}/{NUM_EPOCHS}] | "
                        f"Step [{num_batches}/{len(self.dataloader['train'])}] | "
                        f"lr: {learning_rate:.6f} | "
                        f"Avg loss: {batch_loss:.4f} ("
                        f"1_1: {loss_1_1:.4f}, "
                        f"1_2: {loss_1_2:.4f}, "
                        f"1_4: {loss_1_4:.4f}, "
                        f"1_8: {loss_1_8:.4f})"
                    )
                    self.CONSOLE.info(loss_print)

                # Save batch losses for tensorboard logging
                batch_loss_dict = {
                    '1_1': loss_1_1, '1_2': loss_1_2,
                    '1_4': loss_1_4, '1_8': loss_1_8, 'total': batch_loss
                }
                LOSS_TRACKER.append(batch_loss_dict, self.global_step)

                # Update learning rate
                if (self.scheduler is not None) and (SCHEDULER_PERIOD == 'batch'):
                    self.scheduler.step()
                    self.TRAIN_LOGGER.log_lr(learning_rate, self.global_step)

                # Update global step
                self.global_step += 1

            # End of batch loop

            # Logging batch losses to tensorboard
            # self.loss_tracker.write_batch_losses()
            self.TRAIN_LOGGER.log_batch_loss(
                LOSS_TRACKER.batch_loss(), LOSS_TRACKER.get_steps())

            # Validation
            self.validate(epoch)

            # Update learning rate
            if (self.scheduler is not None) and (SCHEDULER_PERIOD == 'epoch'):
                self.scheduler.step()
                self.TRAIN_LOGGER.log_lr(learning_rate, self.global_step)

            # Print & record epoch loss
            # self.loss_tracker.write_epoch_losses(epoch)
            self.TRAIN_LOGGER.log_epoch_loss(
                LOSS_TRACKER.epoch_loss(), epoch)

            epoch_loss = sum_batch_loss / num_batches
            self.CONSOLE.info(
                f"=> Epoch {epoch} Average Loss: {epoch_loss:.4f}")

            # Update best loss and save model if improved
            if epoch_loss < self.best_loss:
                self.best_loss = epoch_loss
                self._save_checkpoint(epoch, epoch_loss)

        # End of training loop
        self.TRAIN_LOGGER.close()

    def validate(self, epoch: int):

        valid_loss_tracker = logger.LossTracker(self.log_dir, ns='Val')

        self.model.eval()
        with torch.no_grad():
            for batch in self.dataloader['valid']:
                batch = to_device(batch, self.device)

                # Compute validation losses
                pred = self.model(batch['occupancy'])
                loss_1_1 = self.loss_fn.CE_Loss_1_1(pred, batch['label'])
                loss_1_2 = self.loss_fn.CE_Loss_1_2(pred, batch['label_1_2'])
                loss_1_4 = self.loss_fn.CE_Loss_1_4(pred, batch['label_1_4'])
                loss_1_8 = self.loss_fn.CE_Loss_1_8(pred, batch['label_1_8'])
                batch_loss = (loss_1_1 + loss_1_2 + loss_1_4 + loss_1_8) / 4

                # TODO: to be fixed: separate metrics for train and valid? or clean at the end of train
                self.metrics.add_pred_1_1(pred, batch['label'])
                self.metrics.add_pred_1_2(pred, batch['label_1_2'])
                self.metrics.add_pred_1_4(pred, batch['label_1_4'])
                self.metrics.add_pred_1_8(pred, batch['label_1_8'])

                loss_dict = {
                    '1_1': loss_1_1,
                    '1_2': loss_1_2,
                    '1_4': loss_1_4,
                    '1_8': loss_1_8,
                    'total': batch_loss
                }
                valid_loss_tracker.append(loss_dict)
                # valid_loss_tracker.write_batch_losses()

    def record_training_process(self, epoch: int, batch_loss: dict, num_batches: int):
        pass

    def _log_batch_loss(self, losses: dict, step: int):
        """Log multiple loss values to tensorboard.

        Args:
            losses: Dictionary of losses with format {'loss_name': loss_value}
            step: Global step for tensorboard logging
        """
        for loss_name, loss_value in losses.items():
            self.TRAIN_LOGGER.add_scalar(
                f'Batch Loss/{loss_name}',
                loss_value.item() if torch.is_tensor(loss_value) else loss_value,
                step
            )

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
