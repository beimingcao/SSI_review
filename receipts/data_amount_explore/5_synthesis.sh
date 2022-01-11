#!/bin/bash -e


testing_path=../current_exp/testing

cd waveglow

for SPK in $testing_path/*; do 
  python3 inference.py -f <(ls $SPK/*.pt) -w waveglow_256channels_universal_v5.pt -o ./$SPK --is_fp16 -s 0.6
done

cd ..
