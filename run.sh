#!/bin/sh

model_name="cipp"

python multi_train.py -df ~/datasets/synth_000/ --model ~/ai_models/synth_000_$model_name
python multi_train.py -df ~/datasets/synth_A001/ --model ~/ai_models/synth_A001_$model_name
python multi_train.py -df ~/datasets/synth_A002/ --model ~/ai_models/synth_A002_$model_name
python multi_train.py -df ~/datasets/synth_A003/ --model ~/ai_models/synth_A003_$model_name
python multi_train.py -df ~/datasets/synth_A004/ --model ~/ai_models/synth_A004_$model_name
python multi_train.py -df ~/datasets/synth_A005/ --model ~/ai_models/synth_A005_$model_name
python multi_train.py -df ~/datasets/synth_A010/ --model ~/ai_models/synth_A010_$model_name
python multi_train.py -df ~/datasets/synth_A015/ --model ~/ai_models/synth_A015_$model_name
python multi_train.py -df ~/datasets/synth_A020/ --model ~/ai_models/synth_A020_$model_name
python multi_train.py -df ~/datasets/synth_A025/ --model ~/ai_models/synth_A025_$model_name
python multi_train.py -df ~/datasets/synth_A050/ --model ~/ai_models/synth_A050_$model_name
python multi_train.py -df ~/datasets/synth_A100/ --model ~/ai_models/synth_A100_$model_name
