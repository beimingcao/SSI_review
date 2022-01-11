import time
import yaml
import os
import torch
import pickle
from torch.utils.data import Dataset, DataLoader
from utils.models import MyLSTM
from utils.models import RegressionLoss
from utils.models import save_model
from utils.measures import MCD
from torch.autograd import Variable
from sklearn.metrics import mean_squared_error
from math import sqrt
from scipy import ndimage

def test_LSTM(test_SPK, test_dataset, exp_output_folder, stats, args):
    config = yaml.load(open(args.conf_dir, 'r'), Loader=yaml.FullLoader)
    sel_sensors = config['articulatory_data']['sel_sensors']
    sel_dim = config['articulatory_data']['sel_dim'] 
    delta = config['articulatory_data']['delta']
    d = 3 if delta == True else 1
    D_in = len(sel_sensors)*len(sel_dim)*d
    D_out = config['acoustic_feature']['n_mel_channels']
    hidden_size = config['training_setup']['hidden_size']
    num_layers = config['training_setup']['layer_num']
    batch_size = config['training_setup']['batch_size']
    save_output = config['testing_setup']['save_output']
    synthesis_samples = config['testing_setup']['synthesis_samples']
    normalize_input = config['articulatory_data']['normalize_input']
    normalize_output = config['acoustic_feature']['normalize_output']
    metric = MCD()

    test_data = DataLoader(test_dataset, num_workers=0, batch_size=1, shuffle=False, drop_last=False)
    test_out_folder = os.path.join(exp_output_folder, 'testing')
    model_out_folder = os.path.join(exp_output_folder, 'trained_models')
    if not os.path.exists(test_out_folder):
        os.makedirs(test_out_folder)
   
    SPK_model_path = os.path.join(model_out_folder)
    model_path = os.path.join(SPK_model_path, test_SPK + '_lstm')
    model = MyLSTM(D_in, hidden_size, D_out, num_layers)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    acc_vals = []
    for file_id, x, y in test_data:
        x, y = x.type(torch.FloatTensor), y.type(torch.FloatTensor)

        h, c = model.init_hidden(x)
        with torch.no_grad():
            y_head = model(x, h, c)

        if normalize_output == 'MVN':
            y_head = y_head * stats['Y_std'] + stats['Y_mean']
        elif normalize_output == 'MINMAX':
            y_head = y_head * (stats['Y_max']-stats['Y_min']) + stats['Y_min']

        y_pt = y_head.squeeze(0).T

        if save_output == True:
            outpath = os.path.join(test_out_folder, test_SPK)
            if not os.path.exists(outpath):
                os.makedirs(outpath)
            torch.save(y_pt, os.path.join(outpath, file_id[0] + '.pt'))

        acc_vals.append(metric(y.squeeze(0), y_head.squeeze(0)))
    avg_vacc = sum(acc_vals) / len(acc_vals)

    results_out_folder = os.path.join(exp_output_folder, 'RESULTS')
    if not os.path.exists(results_out_folder):
        os.makedirs(results_out_folder)

    results = os.path.join(results_out_folder, test_SPK + '_results.txt')
    with open(results, 'w') as r:
        print('MCD = %0.3f' % avg_vacc, file = r)
    r.close()
    return avg_vacc



if __name__ == '__main__':
    import argparse
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument('--conf_dir', default = 'conf/ATS_conf.yaml')
    parser.add_argument('--buff_dir', default = 'current_exp')
    args = parser.parse_args()
    config = yaml.load(open(args.conf_dir, 'r'), Loader=yaml.FullLoader)
    normalize_input = config['articulatory_data']['normalize_input']
    normalize_output = config['acoustic_feature']['normalize_output']

    data_path = os.path.join(args.buff_dir, 'data_CV')
    SPK_list = config['data_setup']['spk_list']

    results_all = os.path.join(args.buff_dir, 'results_all.txt')
    with open(results_all, 'w') as r:
        for test_SPK in SPK_list:
            data_path_SPK = os.path.join(data_path, test_SPK)
            te = open(os.path.join(data_path_SPK, 'test_data.pkl'), 'rb')
            test_dataset = pickle.load(te)

            if normalize_input != None and normalize_output != None:
                stats_pkl_path = os.path.join(data_path_SPK, 'data_stats.pkl')
                with open(stats_pkl_path, 'rb') as handle:
                    stats = pickle.load(handle)

            avg_vacc = test_LSTM(test_SPK, test_dataset, args.buff_dir, stats, args)
            print(test_SPK, '\t', file = r)
            print('MCD = %0.3f' % avg_vacc, file = r)
    r.close()
