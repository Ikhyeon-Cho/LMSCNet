import LMSCNet.common.checkpoint as checkpoint
from LMSCNet.common.metrics import Metrics
from LMSCNet.common.io_tools import dict_to
from LMSCNet.common.optimizer import build_optimizer, build_scheduler
from LMSCNet.common.logger import get_logger
from LMSCNet.common.model import get_model
from LMSCNet.common.dataset import get_dataset
from LMSCNet.common.config import CFG
from LMSCNet.common.seed import seed_all
import os
import argparse
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import sys
from glob import glob


# Append root directory to system path for imports
repo_path, _ = os.path.split(os.path.realpath(__file__))
repo_path, _ = os.path.split(repo_path)
sys.path.append(repo_path)


def parse_args():
    parser = argparse.ArgumentParser(description='LMSCNet training')
    parser.add_argument(
        '--cfg',
        dest='config_file',
        default='SSC_configs/examples/LMSCNet.yaml',
        metavar='FILE',
        help='path to config file',
        type=str,
    )
    parser.add_argument(
        '--dset_root',
        dest='dataset_root',
        default='/data/semanticKITTI',
        metavar='DATASET',
        help='path to dataset root folder',
        type=str,
    )
    args = parser.parse_args()
    return args


def train(model, optimizer, scheduler, dataset, _cfg, start_epoch, logger, tbwriter):
    """
    Train a model using the PyTorch Module API.
    Inputs:
    - model: A PyTorch Module giving the model to train.
    - optimizer: An Optimizer object we will use to train the model
    - scheduler: Scheduler for learning rate decay if used
    - dataset: The dataset to load files
    - _cfg: The configuration dictionary read from config file
    - start_epoch: The epoch at which start the training (checkpoint)
    - logger: The logger to save info
    - tbwriter: The tensorboard writer to save plots
    Returns: Nothing, but prints model accuracies during training.
    """
    # 1. Set device
    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')
    dtype = torch.float32  # Tensor type to be used

    # 2. Move model and optimizer to used device
    model = model.to(device)
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)

    # 3. Initialize metrics
    dset = dataset['train']
    n_iterations = len(dset)  # number of iteration depends on batch size
    metrics = Metrics(dset.dataset.nbr_classes,
                      n_iterations, model.get_scales())
    metrics.reset_evaluator()
    metrics.losses_track.set_validation_losses(
        model.get_validation_loss_keys())
    metrics.losses_track.set_train_losses(model.get_train_loss_keys())

    # 4. Train
    model.train()
    n_epochs = _cfg.dict_['TRAIN']['EPOCHS']

    for epoch in range(start_epoch, n_epochs+1):

        logger.info(
            '=> =============== Epoch [{}/{}] ==============='.format(epoch, n_epochs))
        logger.info(
            '=> Reminder - Output of routine on {}'.format(_cfg.dict_['OUTPUT']['OUTPUT_PATH']))
        logger.info('=> Learning rate: {}'.format(scheduler.get_lr()[0]))

        for t, (data, indices) in enumerate(dset):

            # Load data to device
            data = dict_to(data, device, dtype)

            scores = model(data)

            loss = model.compute_loss(scores, data)

            # Zero out the gradients.
            optimizer.zero_grad()
            # Backward pass: gradient of loss wr. each model parameter.
            loss['total'].backward()
            # update parameters of model by gradients.
            optimizer.step()

            if _cfg.dict_['SCHEDULER']['FREQUENCY'] == 'iteration':
                scheduler.step()

            for l_key in loss:
                tbwriter.add_scalar('train_loss_batch/{}'.format(l_key),
                                    loss[l_key].item(), len(dset) * (epoch-1) + t)
            # Updating batch losses to then get mean for epoch loss
            metrics.losses_track.update_train_losses(loss)

            if (t + 1) % _cfg.dict_['TRAIN']['SUMMARY_PERIOD'] == 0:
                loss_print = '=> Epoch [{}/{}], Iteration [{}/{}], Learn Rate: {}, Train Losses: '\
                    .format(epoch, n_epochs, t+1, len(dset), scheduler.get_lr()[0])
                for key in loss.keys():
                    loss_print += '{} = {:.6f},  '.format(key, loss[key])
                logger.info(loss_print[:-3])

            metrics.add_batch(prediction=scores, target=model.get_target(data))

        for l_key in metrics.losses_track.train_losses:
            tbwriter.add_scalar('train_loss_epoch/{}'.format(l_key),
                                metrics.losses_track.train_losses[l_key].item(
            )/metrics.losses_track.train_iteration_counts,
                epoch - 1)
        tbwriter.add_scalar('lr/lr', scheduler.get_lr()[0], epoch - 1)

        epoch_loss = metrics.losses_track.train_losses['total'] / \
            metrics.losses_track.train_iteration_counts

        for scale in metrics.evaluator.keys():
            tbwriter.add_scalar('train_performance/{}/mIoU'.format(scale),
                                metrics.get_semantics_mIoU(scale).item(), epoch-1)
            tbwriter.add_scalar('train_performance/{}/IoU'.format(scale),
                                metrics.get_occupancy_IoU(scale).item(), epoch-1)
            # tbwriter.add_scalar('train_performance/{}/Precision'.format(scale), metrics.get_occupancy_Precision(scale).item(), epoch-1)
            # tbwriter.add_scalar('train_performance/{}/Recall'.format(scale), metrics.get_occupancy_Recall(scale).item(), epoch-1)
            # tbwriter.add_scalar('train_performance/{}/F1'.format(scale), metrics.get_occupancy_F1(scale).item(), epoch-1)

        logger.info(
            '=> [Epoch {} - Total Train Loss = {}]'.format(epoch, epoch_loss))
        for scale in metrics.evaluator.keys():
            loss_scale = metrics.losses_track.train_losses['semantic_{}'.format(
                scale)].item()/metrics.losses_track.train_iteration_counts
            logger.info('=> [Epoch {} - Scale {}: Loss = {:.6f} - mIoU = {:.6f} - IoU = {:.6f} '
                        '- P = {:.6f} - R = {:.6f} - F1 = {:.6f}]'
                        .format(epoch, scale, loss_scale,
                                metrics.get_semantics_mIoU(scale).item(),
                                metrics.get_occupancy_IoU(scale).item(),
                                metrics.get_occupancy_Precision(scale).item(),
                                metrics.get_occupancy_Recall(scale).item(),
                                metrics.get_occupancy_F1(scale).item()))

        logger.info('=> Epoch {} - Training set class-wise IoU:'.format(epoch))
        for i in range(1, metrics.nbr_classes):
            class_name = dset.dataset.dataset_config['labels'][dset.dataset.dataset_config['learning_map_inv'][i]]
            class_score = metrics.evaluator['1_1'].getIoU()[1][i]
            logger.info('    => IoU {}: {:.6f}'.format(
                class_name, class_score))

        # Reset evaluator for validation...
        metrics.reset_evaluator()

        checkpoint_info = validate(
            model, dataset['val'], _cfg, epoch, logger, tbwriter, metrics)

        # Reset evaluator and losses for next epoch...
        metrics.reset_evaluator()
        metrics.losses_track.restart_train_losses()
        metrics.losses_track.restart_validation_losses()

        if _cfg.dict_['SCHEDULER']['FREQUENCY'] == 'epoch':
            scheduler.step()

        # Save checkpoints
        for k in checkpoint_info.keys():
            checkpoint_path = os.path.join(
                _cfg.dict_['OUTPUT']['OUTPUT_PATH'], 'chkpt', k)
            _cfg.dict_['STATUS'][checkpoint_info[k]] = checkpoint_path
            checkpoint.save(checkpoint_path, model, optimizer,
                            scheduler, epoch, _cfg.dict_)

        # Save checkpoint if current epoch matches checkpoint period
        if epoch % _cfg.dict_['TRAIN']['CHECKPOINT_PERIOD'] == 0:
            checkpoint_path = os.path.join(
                _cfg.dict_['OUTPUT']['OUTPUT_PATH'], 'chkpt', str(epoch).zfill(2))
            checkpoint.save(checkpoint_path, model, optimizer,
                            scheduler, epoch, _cfg.dict_)

        # Update config file
        _cfg.update_config(resume=True)

    return metrics.best_metric_record


def validate(model, dset, _cfg, epoch, logger, tbwriter, metrics):

    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')
    dtype = torch.float32  # Tensor type to be used

    nbr_epochs = _cfg.dict_['TRAIN']['EPOCHS']

    logger.info('=> Passing the network on the validation set...')

    model.eval()

    with torch.no_grad():

        for t, (data, indices) in enumerate(dset):

            data = dict_to(data, device, dtype)

            scores = model(data)

            loss = model.compute_loss(scores, data)

            for l_key in loss:
                tbwriter.add_scalar('validation_loss_batch/{}'.format(l_key),
                                    loss[l_key].item(), len(dset) * (epoch-1) + t)
            # Updating batch losses to then get mean for epoch loss
            metrics.losses_track.update_validaiton_losses(loss)

            if (t + 1) % _cfg.dict_['VAL']['SUMMARY_PERIOD'] == 0:
                loss_print = '=> Epoch [{}/{}], Iteration [{}/{}], Train Losses: '.format(
                    epoch, nbr_epochs, t+1, len(dset))
                for key in loss.keys():
                    loss_print += '{} = {:.6f},  '.format(key, loss[key])
                logger.info(loss_print[:-3])

            metrics.add_batch(prediction=scores, target=model.get_target(data))

        for l_key in metrics.losses_track.validation_losses:
            tbwriter.add_scalar('validation_loss_epoch/{}'.format(l_key),
                                metrics.losses_track.validation_losses[l_key].item(
            )/metrics.losses_track.validation_iteration_counts,
                epoch - 1)

        epoch_loss = metrics.losses_track.validation_losses['total'] / \
            metrics.losses_track.validation_iteration_counts

        for scale in metrics.evaluator.keys():
            tbwriter.add_scalar('validation_performance/{}/mIoU'.format(scale),
                                metrics.get_semantics_mIoU(scale).item(), epoch-1)
            tbwriter.add_scalar('validation_performance/{}/IoU'.format(scale),
                                metrics.get_occupancy_IoU(scale).item(), epoch-1)
            # tbwriter.add_scalar('validation_performance/{}/Precision'.format(scale), metrics.get_occupancy_Precision(scale).item(), epoch-1)
            # tbwriter.add_scalar('validation_performance/{}/Recall'.format(scale), metrics.get_occupancy_Recall(scale).item(), epoch-1)
            # tbwriter.add_scalar('validation_performance/{}/F1'.format(scale), metrics.get_occupancy_F1(scale).item(), epoch-1)

        logger.info(
            '=> [Epoch {} - Total Validation Loss = {}]'.format(epoch, epoch_loss))
        for scale in metrics.evaluator.keys():
            loss_scale = metrics.losses_track.validation_losses['semantic_{}'.format(
                scale)].item()/metrics.losses_track.train_iteration_counts
            logger.info('=> [Epoch {} - Scale {}: Loss = {:.6f} - mIoU = {:.6f} - IoU = {:.6f} '
                        '- P = {:.6f} - R = {:.6f} - F1 = {:.6f}]'
                        .format(epoch, scale, loss_scale,
                                metrics.get_semantics_mIoU(scale).item(),
                                metrics.get_occupancy_IoU(scale).item(),
                                metrics.get_occupancy_Precision(scale).item(),
                                metrics.get_occupancy_Recall(scale).item(),
                                metrics.get_occupancy_F1(scale).item()))

        logger.info(
            '=> Epoch {} - Validation set class-wise IoU:'.format(epoch))
        for i in range(1, metrics.nbr_classes):
            class_name = dset.dataset.dataset_config['labels'][dset.dataset.dataset_config['learning_map_inv'][i]]
            class_score = metrics.evaluator['1_1'].getIoU()[1][i]
            logger.info('    => {}: {:.6f}'.format(class_name, class_score))

        checkpoint_info = {}

        if epoch_loss < _cfg.dict_['OUTPUT']['BEST_LOSS']:
            logger.info('=> Best loss on validation set encountered: ({} < {})'.
                        format(epoch_loss, _cfg.dict_['OUTPUT']['BEST_LOSS']))
            _cfg.dict_['OUTPUT']['BEST_LOSS'] = epoch_loss.item()
            checkpoint_info['best-loss'] = 'BEST_LOSS'

        mIoU_1_1 = metrics.get_semantics_mIoU('1_1')
        IoU_1_1 = metrics.get_occupancy_IoU('1_1')
        if mIoU_1_1 > _cfg.dict_['OUTPUT']['BEST_METRIC']:
            logger.info('=> Best metric on validation set encountered: ({} > {})'.
                        format(mIoU_1_1, _cfg.dict_['OUTPUT']['BEST_METRIC']))
            _cfg.dict_['OUTPUT']['BEST_METRIC'] = mIoU_1_1.item()
            checkpoint_info['best-metric'] = 'BEST_METRIC'
            metrics.update_best_metric_record(
                mIoU_1_1, IoU_1_1, epoch_loss.item(), epoch)

        checkpoint_info['last'] = 'LAST'

    model.train()  # revert model to training mode
    return checkpoint_info


def main():

    # https://github.com/pytorch/pytorch/issues/27588
    torch.backends.cudnn.enabled = False

    seed_all(0)

    # Parse arguments
    args = parse_args()
    train_cfg = args.config_file
    dataset_root = args.dataset_root

    # Read train cfg file
    _cfg = CFG()
    _cfg.from_config_yaml(train_cfg)

    # Replace dataset path in config file by the one passed by argument
    if dataset_root is not None:
        _cfg.dict_['DATASET']['ROOT_DIR'] = dataset_root

    # Create writer for Tensorboard
    tbwriter = SummaryWriter(log_dir=os.path.join(
        _cfg.dict_['OUTPUT']['OUTPUT_PATH'], 'metrics'))

    # Set the logger to print statements and also save them into logs file
    logger = get_logger(_cfg.dict_['OUTPUT']['OUTPUT_PATH'], 'logs_train.log')
    logger.info('============ Training routine: "%s" ============\n' % train_cfg)
    dataloader_dict = get_dataset(_cfg)

    # Load model
    logger.info('=> Loading network architecture...')
    model = get_model(_cfg.dict_['MODEL']['TYPE'],
                      dataloader_dict['train'].dataset)
    # Multi-GPU settings
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        model = model.module

    # Load optimizer
    logger.info('=> Loading optimizer...')
    optimizer = build_optimizer(_cfg, model)
    scheduler = build_scheduler(_cfg, optimizer)

    # Load checkpoint
    resume = _cfg.dict_['STATUS']['RESUME']
    if resume:
        logger.info('=> Continuing training routine. Checkpoint loaded')
    else:
        logger.info('=> No checkpoint. Initializing model from scratch')
    model, optimizer, scheduler, epoch = checkpoint.load(
        model,
        optimizer,
        scheduler,
        resume,
        _cfg.dict_['STATUS']['LAST'])

    print(model)

    # Train
    best_record = train(model,
                        optimizer,
                        scheduler,
                        dataloader_dict,
                        _cfg,
                        epoch,
                        logger,
                        tbwriter)

    # Print best performance
    logger.info(
        '=> ============ Network trained - all epochs passed... ============')

    logger.info('=> [Best performance: Epoch {} - mIoU = {} - IoU {}]'.format(
        best_record['epoch'], best_record['mIoU'], best_record['IoU']))

    logger.info(
        '=> Writing config file in output folder - deleting from config files folder')
    _cfg.finish_config()
    logger.info('=> Training routine completed...')

    exit()


if __name__ == '__main__':
    main()
