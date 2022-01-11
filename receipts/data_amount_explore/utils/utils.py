import os
import yaml
import glob
import numpy as np
import scipy.io as sio
import librosa

from utils.IO_func import read_file_list

class EarlyStopping():

    # Adopted from: https://debuggercafe.com/using-learning-rate-scheduler-and-early-stopping-with-pytorch/

    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """
    def __init__(self, patience=5, min_delta=0):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.save_model = False
    def __call__(self, val_loss):
        if self.best_loss == None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            # reset counter if validation loss improves
            self.counter = 0
            self.save_model = True
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            self.save_model = False
            print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                print('INFO: Early stopping')
                self.early_stop = True

def prepare_Haskins_lists(args):

    config = yaml.load(open(args.conf_dir, 'r'), Loader=yaml.FullLoader)
    exp_type = config['experimental_setup']['experiment_type']  
    data_path = config['corpus']['path']
    fileset_path = os.path.join(data_path, 'filesets')
    spk_list = config['data_setup']['spk_list']
    num_exp = len(spk_list)
    
    exp_train_lists = {}
    exp_valid_lists = {}
    exp_test_lists = {}
    if exp_type == 'SD':
        for i in range(len(spk_list)):
            spk_fileset_path = os.path.join(fileset_path, spk_list[i])
            exp_train_lists[i] = read_file_list(os.path.join(spk_fileset_path, 'train_id_list.scp'))
            exp_valid_lists[i] = read_file_list(os.path.join(spk_fileset_path, 'valid_id_list.scp'))
            exp_test_lists[i] = read_file_list(os.path.join(spk_fileset_path, 'test_id_list.scp'))
     
    elif exp_type == 'SI':
        for i in range(len(spk_list)):
            train_spk_list = spk_list.copy()
            train_spk_list.remove(spk_list[i])
            idx = 0
            train_lists, valid_lists = [], []
            for train_spk in train_spk_list:
                spk_fileset_path = os.path.join(fileset_path, train_spk)
                if idx == 0:
                    train_lists = read_file_list(os.path.join(spk_fileset_path, 'train_id_list.scp'))
                    valid_lists = read_file_list(os.path.join(spk_fileset_path, 'valid_id_list.scp'))
                else:
                    train_lists = train_lists + read_file_list(os.path.join(spk_fileset_path, 'train_id_list.scp'))
                    valid_lists = valid_lists + read_file_list(os.path.join(spk_fileset_path, 'valid_id_list.scp'))
                idx += 1
            test_lists = read_file_list(os.path.join(os.path.join(fileset_path, spk_list[i]), 'test_id_list.scp'))

            exp_train_lists[i] = train_lists
            exp_valid_lists[i] = valid_lists
            exp_test_lists[i] = test_lists

    elif exp_type == 'SA':
        idx = 0     
        for train_spk in spk_list:
            spk_fileset_path = os.path.join(fileset_path, train_spk)
            if idx == 0:
                train_lists = read_file_list(os.path.join(spk_fileset_path, 'train_id_list.scp'))
                valid_lists = read_file_list(os.path.join(spk_fileset_path, 'valid_id_list.scp'))
            else:
                train_lists = train_lists + read_file_list(os.path.join(spk_fileset_path, 'train_id_list.scp'))
                valid_lists = valid_lists + read_file_list(os.path.join(spk_fileset_path, 'valid_id_list.scp'))

            exp_train_lists[idx] = train_lists
            exp_valid_lists[idx] = valid_lists            
            exp_test_lists[idx] = read_file_list(os.path.join(spk_fileset_path, 'test_id_list.scp')) 
            idx += 1           
    else:
        raise ValueError('Unrecognized experiment type')

    return exp_train_lists, exp_valid_lists, exp_test_lists

