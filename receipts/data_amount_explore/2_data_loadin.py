import time
import yaml
import os
import torch
import pickle
from utils.database import HaskinsData_ATS
from torch.utils.data import Dataset, DataLoader
from utils.transforms import Pair_Transform_Compose
from utils.IO_func import read_file_list, load_binary_file, array_to_binary_file, load_Haskins_ATS_data
from utils.utils import prepare_Haskins_lists
from shutil import copyfile
from utils.transforms import padding_end, apply_EMA_MVN, apply_WAV_MVN, zero_padding_end
import random

seed = 123
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--conf_dir', default = 'conf/ATS_conf.yaml')
    parser.add_argument('--buff_dir', default = 'current_exp')
    args = parser.parse_args()
    config = yaml.load(open(args.conf_dir, 'r'), Loader=yaml.FullLoader)

    sel_sensors = config['articulatory_data']['sel_sensors']
    sel_dim = config['articulatory_data']['sel_dim'] 
    delta = config['articulatory_data']['delta']
    d = 3 if delta == True else 1
    ema_dim = len(sel_sensors)*len(sel_dim)*d

    prepared_data_path = os.path.join(args.buff_dir, 'data')
    prepared_data_CV_path = os.path.join(args.buff_dir, 'data_CV')

    normalize_input = config['articulatory_data']['normalize_input']
    normalize_output = config['acoustic_feature']['normalize_output']
    batch_size = config['training_setup']['batch_size']

    train_transforms = []
    valid_transforms = []
    test_transforms = []

    exp_train_lists, exp_valid_lists, exp_test_lists = prepare_Haskins_lists(args)

    for i in range(len(exp_test_lists)):
  #      CV = 'CV' + format(i, '02d')
        CV = exp_test_lists[i][0][:3]
        CV_data_dir = os.path.join(prepared_data_CV_path, CV)
        if not os.path.exists(CV_data_dir):
            os.makedirs(CV_data_dir)

        train_list = exp_train_lists[i]
        valid_list = exp_valid_lists[i]
        test_list = exp_test_lists[i]

        train_dataset = HaskinsData_ATS(prepared_data_path, train_list, ema_dim)
        if normalize_input != None and normalize_output != None:
            EMA_mean, EMA_std, WAV_mean, WAV_std, EMA_min, EMA_max, WAV_min, WAV_max = train_dataset.compute_stats()
            data_stats = {'X_mean': EMA_mean, 'X_std': EMA_std, 'X_min': EMA_min, 'X_max': EMA_max,
                          'Y_mean': WAV_mean, 'Y_std': WAV_std, 'Y_min': WAV_min, 'Y_max': WAV_max}
            
            stats_pkl_path = os.path.join(CV_data_dir, 'data_stats.pkl')
            with open(stats_pkl_path, 'wb') as handle:
                pickle.dump(data_stats, handle, protocol=pickle.HIGHEST_PROTOCOL)

        if normalize_input == 'MVN':
            train_transforms.append(apply_EMA_MVN(EMA_mean, EMA_std))
            valid_transforms.append(apply_EMA_MVN(EMA_mean, EMA_std))
            test_transforms.append(apply_EMA_MVN(EMA_mean, EMA_std))
        elif normalize_input == 'MINMAX':
            train_transforms.append(apply_EMA_MinMax(EMA_min, EMA_max))
            valid_transforms.append(apply_EMA_MinMax(EMA_min, EMA_max))
            test_transforms.append(apply_EMA_MinMax(EMA_min, EMA_max))

        if normalize_output == 'MVN':
            train_transforms.append(apply_WAV_MVN(WAV_mean, WAV_std))
            valid_transforms.append(apply_WAV_MVN(WAV_mean, WAV_std))

        elif normalize_output == 'MINMAX':
            train_transforms.append(apply_WAV_MinMax(WAV_min, WAV_max))
            valid_transforms.append(apply_WAV_MinMax(WAV_min, WAV_max))

        if batch_size > 1:
            valid_dataset = HaskinsData_ATS(prepared_data_path, valid_list, ema_dim)
  #          max_len = max(train_dataset.find_max_len(), valid_dataset.find_max_len())
            max_len = 340
  #          train_transforms.append(padding_end(max_len))
  #          valid_transforms.append(padding_end(max_len))    
            train_transforms.append(zero_padding_end(max_len))
            valid_transforms.append(zero_padding_end(max_len))

        print(train_transforms)
        print('#######################################')
        print(valid_transforms)
        print('#######################################')
        print(test_transforms)

        train_dataset = HaskinsData_ATS(prepared_data_path, train_list, ema_dim, transforms = Pair_Transform_Compose(train_transforms))
        valid_dataset = HaskinsData_ATS(prepared_data_path, valid_list, ema_dim, transforms = Pair_Transform_Compose(valid_transforms))
        test_dataset = HaskinsData_ATS(prepared_data_path, test_list, ema_dim, transforms = Pair_Transform_Compose(test_transforms))
        
        train_pkl_path = os.path.join(CV_data_dir, 'train_data.pkl')
        tr = open(train_pkl_path, 'wb')
        pickle.dump(train_dataset, tr)
        valid_pkl_path = os.path.join(CV_data_dir, 'valid_data.pkl')
        va = open(valid_pkl_path, 'wb')
        pickle.dump(valid_dataset, va)
        test_pkl_path = os.path.join(CV_data_dir, 'test_data.pkl')
        te = open(test_pkl_path, 'wb')
        pickle.dump(test_dataset, te)

        

