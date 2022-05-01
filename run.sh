#!/bin/sh

model_name="cipp"

python multi_train.py -df ~/synth_000/ --model ~/synth_000_$model_name
python multi_train.py -df ~/synth_A001/ --model ~/synth_A001_$model_name
python multi_train.py -df ~/synth_A002/ --model ~/synth_A002_$model_name
python multi_train.py -df ~/synth_A003/ --model ~/synth_A003_$model_name
python multi_train.py -df ~/synth_A004/ --model ~/synth_A004_$model_name
python multi_train.py -df ~/synth_A005/ --model ~/synth_A005_$model_name
python multi_train.py -df ~/synth_A010/ --model ~/synth_A010_$model_name
python multi_train.py -df ~/synth_A015/ --model ~/synth_A015_$model_name
python multi_train.py -df ~/synth_A020/ --model ~/synth_A020_$model_name
python multi_train.py -df ~/synth_A025/ --model ~/synth_A025_$model_name
python multi_train.py -df ~/synth_A050/ --model ~/synth_A050_$model_name
python multi_train.py -df ~/synth_A100/ --model ~/synth_A100_$model_name
