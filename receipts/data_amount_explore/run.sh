#!/bin/bash -e

new_experiment=True
training=True
testing=True

save_results=True
comment="No_comment"

conf_dir=conf/ATS_conf.yaml
experiments_dir=experiments
current_exp=current_exp

if [ "$new_experiment" = "True" ];then
echo "New experiments, loading data into current_exp folder"
#rm -rf $current_exp
#python3 0_setup.py --conf_dir $conf_dir --exp_dir $experiments_dir --buff_dir $current_exp
#python3 1_data_prepare.py --conf_dir $conf_dir --buff_dir $current_exp
python3 2_data_loadin.py --conf_dir $conf_dir --buff_dir $current_exp
fi
if [ "$training" = "True" ];then
rm -rf $current_exp/training
rm -rf $current_exp/trained_models
python3 3_train.py --conf_dir $conf_dir --buff_dir $current_exp
fi

if [ "$testing" = "True" ];then
rm -rf $current_exp/testing
rm -rf $current_exp/RESULTS
python3 4_test.py --conf_dir $conf_dir --buff_dir $current_exp
fi

./5_synthesis.sh

if [ "$save_results" = "True" ];then
python3 6_cleanup.py --conf_dir $conf_dir --exp_dir $experiments_dir --buff_dir $current_exp --comment $comment
fi
