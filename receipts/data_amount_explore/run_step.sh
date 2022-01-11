#!/bin/bash -e


save_results=True
comment="No_comment"

conf_dir=conf/ATS_conf.yaml
experiments_dir=experiments
current_exp=current_exp

stage=0

if [ $stage -le 0 ];then
echo "New experiments, converting data in binary type, and put into current_exp folder, apply sample-level transforms"
rm -rf $current_exp
python3 0_setup.py --conf_dir $conf_dir --exp_dir $experiments_dir --buff_dir $current_exp
python3 1_data_prepare.py --conf_dir $conf_dir --buff_dir $current_exp
fi

if [ $stage -le 1 ];then
rm -rf $current_exp/data_CV
rm -rf $current_exp/RESULTS
rm -rf $current_exp/testing
rm -rf $current_exp/results_all.txt
echo "Load data into pickle files for future usage, applying transforms that needs information from training set only, like MVN"
python3 2_data_loadin.py --conf_dir $conf_dir --buff_dir $current_exp
fi


if [ $stage -le 2 ];then
echo "Start training"
rm -rf $current_exp/training
rm -rf $current_exp/trained_models
python3 3_train.py --conf_dir $conf_dir --buff_dir $current_exp
fi

if [ $stage -le 3 ];then
echo "Start testing"
rm -rf $current_exp/testing
rm -rf $current_exp/RESULTS
python3 4_test.py --conf_dir $conf_dir --buff_dir $current_exp
fi

if [ $stage -le 4 ];then
echo "Synthesizing acoustic features into wave samples"
./5_synthesis.sh
fi

if [ "$save_results" = "True" ];then
python3 6_cleanup.py --conf_dir $conf_dir --exp_dir $experiments_dir --buff_dir $current_exp --comment $comment
fi
