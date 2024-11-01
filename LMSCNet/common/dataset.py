from torch.utils.data import DataLoader

from LMSCNet.data.SemanticKITTI import SemanticKITTIDataset


def get_dataset(_cfg):

  if _cfg.dict_['DATASET']['TYPE'] == 'SemanticKITTI':
    ds_train = SemanticKITTIDataset(_cfg.dict_['DATASET'], 'train')
    ds_val   = SemanticKITTIDataset(_cfg.dict_['DATASET'], 'val')
    ds_test  = SemanticKITTIDataset(_cfg.dict_['DATASET'], 'test')

  _cfg.dict_['DATASET']['SPLIT'] = {'TRAIN': len(ds_train), 'VAL': len(ds_val), 'TEST': len(ds_test)}

  dataset = {}

  train_batch_size = _cfg.dict_['TRAIN']['BATCH_SIZE']
  val_batch_size = _cfg.dict_['VAL']['BATCH_SIZE']
  num_workers = _cfg.dict_['DATALOADER']['NUM_WORKERS']

  dataset['train'] = DataLoader(ds_train, batch_size=train_batch_size, num_workers=num_workers, shuffle=True)
  dataset['val']   = DataLoader(ds_val,   batch_size=val_batch_size, num_workers=num_workers, shuffle=False)
  dataset['test']  = DataLoader(ds_test,   batch_size=val_batch_size, num_workers=num_workers, shuffle=False)

  return dataset