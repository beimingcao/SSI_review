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
from torch.optim.lr_scheduler import StepLR
from utils.utils import EarlyStopping
import random

seed = 123
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

def train_LSTM(test_SPK, train_dataset, valid_dataset, exp_output_folder, args):
    
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

    learning_rate = config['training_setup']['learning_rate']
    weight_decay = config['training_setup']['weight_decay']
    num_epoch = config['training_setup']['num_epoch']
    early_stop = config['training_setup']['early_stop']
    patient = config['training_setup']['patient']

    model = MyLSTM(D_in, hidden_size, D_out, num_layers)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_func = RegressionLoss()
    metric = MCD()

    train_data = DataLoader(train_dataset, num_workers=0, batch_size=batch_size, shuffle=True, drop_last=False)
    valid_data = DataLoader(valid_dataset, num_workers=0, batch_size=batch_size, shuffle=True, drop_last=False)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    train_out_folder = os.path.join(exp_output_folder, 'training')
    if not os.path.exists(train_out_folder):
        os.makedirs(train_out_folder)
    results = os.path.join(train_out_folder, test_SPK + '_train.txt')

    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
    if early_stop == True:
        print('Applying early stop.')
        early_stopping = EarlyStopping(patience=patient)
    with open(results, 'w') as r:
        for epoch in range(num_epoch):
            model.train()
            acc_vals = []
            for file_id, x, y in train_data:
                x, y = x.type(torch.FloatTensor).to(device), y.type(torch.FloatTensor).to(device)
                h, c = model.init_hidden(x)
                h, c = h.to(device), c.to(device)
                y_head = model(x, h, c)

                loss_val = loss_func(y_head, y)
            #    acc_val = metric(y_head.squeeze(0), y.squeeze(0))
                acc_val = loss_val
                acc_vals.append(acc_val)

                optimizer.zero_grad()
                loss_val.backward()
                optimizer.step()    
            avg_acc = sum(acc_vals) / len(acc_vals)

            model.eval()
            acc_vals = []
            for file_id, x, y in valid_data:
                x, y = x.type(torch.FloatTensor).to(device), y.type(torch.FloatTensor).to(device)
                h, c = model.init_hidden(x)
                h, c = h.to(device), c.to(device)
                acc_vals.append(metric(model(x, h, c).squeeze(0), y.squeeze(0)))
            scheduler.step()
            avg_vacc = sum(acc_vals) / len(acc_vals)
            SPK = file_id[0][:3]

            early_stopping(avg_vacc)
            if early_stopping.early_stop:
                break

            print('epoch %-3d \t acc = %0.3f \t val acc = %0.3f' % (epoch, avg_acc, avg_vacc))
            print('epoch %-3d \t acc = %0.3f \t val acc = %0.3f' % (epoch, avg_acc, avg_vacc), file = r)

            model_out_folder = os.path.join(exp_output_folder, 'trained_models')
            if not os.path.exists(model_out_folder):
                os.makedirs(model_out_folder)
            if early_stopping.save_model == True:
                save_model(model, os.path.join(model_out_folder, test_SPK + '_lstm'))
    r.close()
    print('Training for testing SPK: ' + test_SPK + ' is done.')




if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--conf_dir', default = 'conf/ATS_conf.yaml')
    parser.add_argument('--buff_dir', default = 'current_exp')
    args = parser.parse_args()
    config = yaml.load(open(args.conf_dir, 'r'), Loader=yaml.FullLoader)

    data_path = os.path.join(args.buff_dir, 'data_CV')
    SPK_list = config['data_setup']['spk_list']

    for test_SPK in SPK_list:
        data_path_SPK = os.path.join(data_path, test_SPK)

        tr = open(os.path.join(data_path_SPK, 'train_data.pkl'), 'rb') 
        va = open(os.path.join(data_path_SPK, 'valid_data.pkl'), 'rb')        
        train_dataset, valid_dataset = pickle.load(tr), pickle.load(va)

        train_LSTM(test_SPK, train_dataset, valid_dataset, args.buff_dir, args)    



