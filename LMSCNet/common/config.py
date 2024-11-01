import yaml
import os

from LMSCNet.common.time import get_date_sting


class CFG:

  def __init__(self):
    '''
    Class constructor
    :param config_path:
    '''

    # Initializing dict...
    self.dict_ = {}
    return

  def from_config_yaml(self, config_path):
    '''
    Class constructor
    :param config_path:
    '''

    # Reading config file
    self.dict_ = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)

    self.dict_['STATUS']['CONFIG'] = config_path

    if not 'OUTPUT_PATH' in self.dict_['OUTPUT'].keys():
      self.set_output_filename()
      self.init_stats()
      self.update_config()

    return

  def from_dict(self, config_dict):
    '''
    Class constructor
    :param config_path:
    '''

    # Reading config file
    self.dict_ = config_dict
    return

  def set_output_filename(self):
    '''
    Set output path in the form Model_Dataset_DDYY_HHMMSS
    '''
    datetime = get_date_sting()
    model = self.dict_['MODEL']['TYPE']
    dataset = self.dict_['DATASET']['TYPE']
    OUT_PATH = os.path.join(self.dict_['OUTPUT']['OUT_ROOT'], model + '_' + dataset + '_' + datetime)
    self.dict_['OUTPUT']['OUTPUT_PATH'] = OUT_PATH
    return

  def update_config(self, resume=False):
    '''
    Save config file
    '''
    if resume:
      self.set_resume()
    yaml.dump(self.dict_, open(self.dict_['STATUS']['CONFIG'], 'w'))
    return

  def init_stats(self):
    '''
    Initialize training stats (i.e. epoch mean time, best loss, best metrics)
    '''
    self.dict_['OUTPUT']['BEST_LOSS'] = 999999999999
    self.dict_['OUTPUT']['BEST_METRIC'] = -999999999999
    self.dict_['STATUS']['LAST'] = ''
    return

  def set_resume(self):
    '''
    Update resume status dict file
    '''
    if not self.dict_['STATUS']['RESUME']:
      self.dict_['STATUS']['RESUME'] = True
    return

  def finish_config(self):
    self.move_config(os.path.join(self.dict_['OUTPUT']['OUTPUT_PATH'], 'config.yaml'))
    return

  def move_config(self, path):
    # Remove from original path
    os.remove(self.dict_['STATUS']['CONFIG'])
    # Change ['STATUS']['CONFIG'] to new path
    self.dict_['STATUS']['CONFIG'] = path
    # Save to routine output folder
    yaml.dump(self.dict_, open(path, 'w'))

    return
