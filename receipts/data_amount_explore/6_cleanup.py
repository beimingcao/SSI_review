import time
import yaml
import os
from shutil import copyfile
from subprocess import call

def cp_dir(source, target):
    call(['cp', '-a', source, target])

def cleanup(args):

    exp_path = args.exp_dir
    buff_path = args.buff_dir   
    config_path = args.conf_dir

    config = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)

    if not os.path.exists(exp_path):
        os.makedirs(exp_path)

    experiments_folder = exp_path
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    EXP_TYPE = config['experimental_setup']['experiment_type']  

    Experiment_ID = EXP_TYPE + '_' + timestamp + '_' + args.comment
    Exp_output_folder = os.path.join(experiments_folder, Experiment_ID)

    if not os.path.exists(Exp_output_folder):
        os.makedirs(Exp_output_folder)


    conf_dst = os.path.join(Exp_output_folder, os.path.basename(config_path))        
    copyfile(config_path, conf_dst)
    
    model_dst = os.path.join(Exp_output_folder, 'models.py')
    copyfile('models.py', model_dst)

    train_dst = os.path.join(Exp_output_folder, '3_train.py')
    copyfile('3_train.py', train_dst)

    result_src = os.path.join(buff_path, 'RESULTS')
    result_dst = os.path.join(Exp_output_folder)
    cp_dir(result_src, result_dst)

    train_src = os.path.join(buff_path, 'training')
    train_dst = os.path.join(Exp_output_folder)
    cp_dir(train_src, train_dst)

    test_src = os.path.join(buff_path, 'testing')
    test_dst = os.path.join(Exp_output_folder)
    cp_dir(test_src, test_dst)

    result_all_src = os.path.join(buff_path, 'results_all.txt')
    result_all_dst = os.path.join(Exp_output_folder, 'results_all.txt')
    copyfile(result_all_src, result_all_dst)



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--conf_dir', default = 'conf/ATS_conf.yaml')
    parser.add_argument('--exp_dir', default = 'experiments')
    parser.add_argument('--buff_dir', default = 'current_exp')
    parser.add_argument('--comment', default = 'no_comment')

    args = parser.parse_args()
    cleanup(args)
