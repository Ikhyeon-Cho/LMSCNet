"""
Author: Ikhyeon Cho
Link: https://github.com/Ikhyeon-Cho
File: trainer.py
Date: 2024/11/2 18:50

Unofficial re-implementation of LMSCNet.
Reference: https://github.com/astra-vision/LMSCNet
"""

import torch
from models.LMSCNet import LMSCNet, LMSCNetMetrics
from torch.nn import CrossEntropyLoss
from train.optimizer import Optimizer
import train.eval_tools as eval_tools
from utils.pytorch.device_tools import to_device
from utils.logger import console, tboard
import os

class LMSCNetTrainer:
    def __init__(self, model: LMSCNet, data: dict, crit: CrossEntropyLoss, optim: Optimizer, cfg: dict, device: str, log_dir: str):
        # Model, Data, Loss, Optimizer, Scheduler
        self.model = model.to(device)
        self.dataloader = data
        self.criterion = crit
        self.optimizer = optim.optimizer
        self.scheduler = optim.scheduler

        # Training configs
        self.CFG = cfg
        self.device = device

        # Logger settings
        self.log_dir = log_dir
        self.logger = console.Logger(log_dir, filename='train.log')
        self.tensorboard = tboard.Logger(log_dir, ns='Train')

        self.global_batch_step = 0
        self.eval = LMSCNetMetrics(num_classes=model.num_classes)

        # Store best loss for model saving
        self.best_loss = float('inf')

    def train(self):
        NUM_EPOCHS = self.CFG['epochs']
        SUMMARY_PERIOD = self.CFG['summary_period']
        CHECKPOINT_PERIOD = self.CFG['checkpoint_period']
        SCHEDULER_PERIOD = self.CFG['scheduler_frequency']

        LOSS_TRACKER = eval_tools.LossTracker()

        # Training loop
        for epoch in range(1, NUM_EPOCHS+1):

            lr = self.optimizer.param_groups[0]['lr']
            self.logger.info(f'=> ====== Epoch [{epoch}/{NUM_EPOCHS}] ======')
            self.logger.info(f'=> Learning rate: {lr}')

            self.model.train()
            LOSS_TRACKER.reset()

            # Batch loop
            for num_batches, batch in enumerate(self.dataloader['train']):
                batch = to_device(batch, self.device)

                # Compute training losses
                pred = self.model(batch['occupancy'])
                loss_1_1 = self.criterion(pred['pred'], batch['label'])
                loss_1_2 = self.criterion(pred['pred_1_2'], batch['label_1_2'])
                loss_1_4 = self.criterion(pred['pred_1_4'], batch['label_1_4'])
                loss_1_8 = self.criterion(pred['pred_1_8'], batch['label_1_8'])
                batch_loss = (loss_1_1 + loss_1_2 + loss_1_4 + loss_1_8) / 4

                # Compute performance metrics (updated after batch loop)
                self.eval.add_preds_1_1(pred, batch['label'])
                self.eval.add_preds_1_2(pred, batch['label_1_2'])
                self.eval.add_preds_1_4(pred, batch['label_1_4'])
                self.eval.add_preds_1_8(pred, batch['label_1_8'])

                # Update network parameters
                self.optimizer.zero_grad()
                batch_loss.backward()
                self.optimizer.step()

                # Print losses per period
                if (num_batches == 0) or ((num_batches+1) % SUMMARY_PERIOD == 0):
                    loss_print = (
                        f"=> Epoch [{epoch}/{NUM_EPOCHS}] | "
                        f"Step [{num_batches+1}/{len(self.dataloader['train'])}] | "
                        f"lr: {lr:.6f} | "
                        f"Avg loss: {batch_loss:.4f} ("
                        f"1_1: {loss_1_1:.4f}, "
                        f"1_2: {loss_1_2:.4f}, "
                        f"1_4: {loss_1_4:.4f}, "
                        f"1_8: {loss_1_8:.4f})"
                    )
                    self.logger.info(loss_print)

                # Save batch losses for tensorboard logging
                batch_loss_dict = {
                    '1_1': loss_1_1, '1_2': loss_1_2,
                    '1_4': loss_1_4, '1_8': loss_1_8, 'total': batch_loss
                }
                LOSS_TRACKER.append(batch_loss_dict, self.global_batch_step)

                # Update learning rate
                if (self.scheduler is not None) and (SCHEDULER_PERIOD == 'batch'):
                    self.tensorboard.log_lr(lr, self.global_batch_step)
                    self.scheduler.step()

                self.global_batch_step += 1

            # End of batch loop

            # Print epoch losses
            epoch_loss = LOSS_TRACKER.epoch_loss()
            self.logger.info("=> -------- Summary ---------")
            loss_print = (
                f"=>   Training Loss (Epoch {epoch}) | "
                f"Avg loss: {epoch_loss['total']:.4f} | "
                f"1_1: {epoch_loss['1_1']:.4f} | "
                f"1_2: {epoch_loss['1_2']:.4f} | "
                f"1_4: {epoch_loss['1_4']:.4f} | "
                f"1_8: {epoch_loss['1_8']:.4f}"
            )
            self.logger.info(loss_print)

            # Log losses, lr to tensorboard
            self.tensorboard.log_batch_loss(
                LOSS_TRACKER.batch_loss(), LOSS_TRACKER.batch_steps())
            self.tensorboard.log_epoch_loss(epoch_loss, epoch)
            self.tensorboard.log_lr(lr, epoch)

            # Update learning rate
            if (self.scheduler is not None) and (SCHEDULER_PERIOD == 'epoch'):
                self.scheduler.step()

            # Validation
            self.validate(epoch)

            # Save checkpoint per period
            if epoch % CHECKPOINT_PERIOD == 0:
                self._save_checkpoint(
                    epoch, epoch_loss, f'LMSCNet_epoch_{epoch}')

            # Save best model if improved
            if epoch_loss["total"] < self.best_loss:
                self.best_loss = epoch_loss["total"]
                self._save_checkpoint(
                    epoch, epoch_loss, f'LMSCNet_best')

        # End of training loop

    def validate(self, epoch: int):

        VALIDATION_SUMMARY_PERIOD = self.CFG['validation_summary_period']

        VAL_LOSS_TRACKER = eval_tools.LossTracker()
        VAL_EVAL = LMSCNetMetrics(num_classes=self.model.num_classes)
        VAL_TENSORBOARD = tboard.Logger(self.log_dir, ns='Val')

        self.model.eval()
        with torch.no_grad():
            # Batch loop
            for batch in self.dataloader['valid']:
                batch = to_device(batch, self.device)

                # Compute validation losses
                pred = self.model(batch['occupancy'])
                loss_1_1 = self.criterion(pred['pred'], batch['label'])
                loss_1_2 = self.criterion(pred['pred_1_2'], batch['label_1_2'])
                loss_1_4 = self.criterion(pred['pred_1_4'], batch['label_1_4'])
                loss_1_8 = self.criterion(pred['pred_1_8'], batch['label_1_8'])
                batch_loss = (loss_1_1 + loss_1_2 + loss_1_4 + loss_1_8) / 4

                # TODO: to be fixed: separate metrics for train and valid? or clean at the end of train
                VAL_EVAL.add_preds_1_1(pred, batch['label'])
                VAL_EVAL.add_preds_1_2(pred, batch['label_1_2'])
                VAL_EVAL.add_preds_1_4(pred, batch['label_1_4'])
                VAL_EVAL.add_preds_1_8(pred, batch['label_1_8'])

                loss_dict = {
                    '1_1': loss_1_1,
                    '1_2': loss_1_2,
                    '1_4': loss_1_4,
                    '1_8': loss_1_8,
                    'total': batch_loss
                }
                VAL_LOSS_TRACKER.append(loss_dict)
            # End of batch loop

            # Log losses to tensorboard
            VAL_TENSORBOARD.log_batch_loss(
                VAL_LOSS_TRACKER.batch_loss(), VAL_LOSS_TRACKER.batch_steps())
            VAL_TENSORBOARD.log_epoch_loss(
                VAL_LOSS_TRACKER.epoch_loss(), epoch)

            # Print validation summary
            if (epoch == 0) or (epoch % VALIDATION_SUMMARY_PERIOD == 0):
                loss_print = (
                    f"=> Validation Loss (Epoch {epoch}) | "
                    f"Avg loss: {VAL_LOSS_TRACKER.epoch_loss()['total']:.4f} | "
                    f"1_1: {VAL_LOSS_TRACKER.epoch_loss()['1_1']:.4f} | "
                    f"1_2: {VAL_LOSS_TRACKER.epoch_loss()['1_2']:.4f} | "
                    f"1_4: {VAL_LOSS_TRACKER.epoch_loss()['1_4']:.4f} | "
                    f"1_8: {VAL_LOSS_TRACKER.epoch_loss()['1_8']:.4f}"
                )
                self.logger.info(loss_print)
                self.logger.info("=> --------------------------")

    def _save_checkpoint(self, epoch, loss, ckpt_name: str):
        """Save model checkpoint when best loss is achieved"""

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'loss': loss,
            'config': self.CFG
        }

        checkpoint_path = os.path.join(self.log_dir, f'{ckpt_name}.pth')
        torch.save(checkpoint, checkpoint_path)

    def get_best_record(self):
        return {
            'best_loss': self.best_loss,
            'checkpoint_dir': self.log_dir
        }
