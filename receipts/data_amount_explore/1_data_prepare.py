import time
import yaml
import os
import torch
from utils.IO_func import read_file_list, load_binary_file, array_to_binary_file, load_Haskins_ATS_data
from shutil import copyfile
from utils.transforms import Pair_Transform_Compose
from utils.transforms import Fix_EMA_MissingValues_ATS, apply_delta_deltadelta_EMA_ATS, ProcrustesMatching_ATS, wav2melspec_ATS, change_wav_sampling_rate_ATS, ema_wav_length_match, padding_end, apply_EMA_MVN, zero_padding_end

def data_processing(args):

    '''
    Load in data from all speakers involved, apply feature extraction, 
    save them into binary files in the current_exp folder, 
    so that data loadin will be accelerated a lot.

    '''

    config_path = args.conf_dir       
    config = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)

    out_folder = os.path.join(args.buff_dir, 'data')
    
    transforms = [Fix_EMA_MissingValues_ATS()] # default transforms
    data_path = config['corpus']['path']
    fileset_path = os.path.join(data_path, 'filesets')
    SPK_list = config['data_setup']['spk_list']
    ################ Articulatory data processing #################
    sel_sensors = config['articulatory_data']['sel_sensors']
    sel_dim = config['articulatory_data']['sel_dim']  
    procrustes = config['articulatory_data']['Procrustes']
    delta = config['articulatory_data']['delta']

    if procrustes == True:
        lateral = config['articulatory_data']['lateral']  
        transforms.append(ProcrustesMatching_ATS(sel_sensors, sel_dim, lateral = 'xz')) 
    if delta == True:
        transforms.append(apply_delta_deltadelta_EMA_ATS())
    ################ Acoustic data processing #################
    sampling_rate = config['acoustic_feature']['sampling_rate']
    filter_length = config['acoustic_feature']['filter_length']
    hop_length = config['acoustic_feature']['hop_length']
    win_length = config['acoustic_feature']['win_length']
    n_mel_channels = config['acoustic_feature']['n_mel_channels']
    mel_fmin = config['acoustic_feature']['mel_fmin']
    mel_fmax = config['acoustic_feature']['mel_fmax']

    transforms.append(change_wav_sampling_rate_ATS())
    transforms.append(wav2melspec_ATS(sampling_rate, filter_length, hop_length, win_length, 
                 n_mel_channels, mel_fmin, mel_fmax))
    transforms.append(ema_wav_length_match())

    transforms_all = Pair_Transform_Compose(transforms)
    
    for SPK in SPK_list:
        out_folder_SPK = os.path.join(out_folder, SPK)
        if not os.path.exists(out_folder_SPK):
            os.makedirs(out_folder_SPK)

        fileset_path_SPK = os.path.join(fileset_path, SPK)
        file_id_list = read_file_list(os.path.join(fileset_path_SPK, 'file_id_list.scp'))

        for file_id in file_id_list:
            data_path_spk = os.path.join(data_path, file_id[:3])
            mat_path = os.path.join(data_path_spk, 'data/'+ file_id + '.mat')
            EMA, WAV, fs_ema, fs_wav = load_Haskins_ATS_data(mat_path, file_id, sel_sensors, sel_dim)
            EMA, WAV = transforms_all(EMA, WAV) 
            EMA_out_dir = os.path.join(out_folder_SPK, file_id + '.ema')
            WAV_out_dir = os.path.join(out_folder_SPK, file_id + '.pt')
            array_to_binary_file(EMA, EMA_out_dir)
            torch.save(WAV, WAV_out_dir)
 
            
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--conf_dir', default = 'conf/ATS_conf.yaml')
    parser.add_argument('--buff_dir', default = 'current_exp')

    args = parser.parse_args()
    data_processing(args)
