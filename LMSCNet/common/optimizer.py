import torch.optim as optim


def build_optimizer(_cfg, model):

    opt = _cfg.dict_['OPTIMIZER']['TYPE']
    lr = _cfg.dict_['OPTIMIZER']['BASE_LR']
    if 'MOMENTUM' in _cfg.dict_['OPTIMIZER']:
        momentum = _cfg.dict_['OPTIMIZER']['MOMENTUM']
    if 'WEIGHT_DECAY' in _cfg.dict_['OPTIMIZER']:
        weight_decay = _cfg.dict_['OPTIMIZER']['WEIGHT_DECAY']

    if opt == 'Adam':
        optimizer = optim.Adam(model.get_parameters(),
                               lr=lr,
                               betas=(0.9, 0.999))

    elif opt == 'SGD':
        optimizer = optim.SGD(model.get_parameters(),
                              lr=lr,
                              momentum=momentum,
                              weight_decay=weight_decay)

    return optimizer


def build_scheduler(_cfg, optimizer):

    # Constant learning rate
    if _cfg.dict_['SCHEDULER']['TYPE'] == 'constant':
        def lambda1(epoch): return 1
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)

    # Learning rate scaled by 0.98^(epoch)
    if _cfg.dict_['SCHEDULER']['TYPE'] == 'power_iteration':
        def lambda1(epoch): return (0.98) ** (epoch)
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)

    return scheduler
