#!/bin/sh

model_name="cipp"

base_path="/Users/friedrichmunke/datasets/synth_v2"

python multi_train.py -df $base_path/synth_000/ --model $base_path/synth_000/$model_name
python multi_train.py -df $base_path/synth_A005/ --model $base_path/synth_A005/$model_name
python multi_train.py -df $base_path/synth_A010/ --model $base_path/synth_A010/$model_name
python multi_train.py -df $base_path/synth_A015/ --model $base_path/synth_A015/$model_name
python multi_train.py -df $base_path/synth_A020/ --model $base_path/synth_A020/$model_name
python multi_train.py -df $base_path/synth_A025/ --model $base_path/synth_A025/$model_name
python multi_train.py -df $base_path/synth_A030/ --model $base_path/synth_A030/$model_name
python multi_train.py -df $base_path/synth_A035/ --model $base_path/synth_A035/$model_name
python multi_train.py -df $base_path/synth_A040/ --model $base_path/synth_A040/$model_name
python multi_train.py -df $base_path/synth_A045/ --model $base_path/synth_A045/$model_name
python multi_train.py -df $base_path/synth_A050/ --model $base_path/synth_A050/$model_name
python multi_train.py -df $base_path/synth_A100/ --model $base_path/synth_A100/$model_name
