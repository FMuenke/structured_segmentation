# Structured Segmentation
## Introduction

This repository provides a lightweight framework, 
that can be used to build semantic segmentation models based on structured classifiers.
The frameworks implements structured classifiers as independent layers, which can be easily combined with another to 
create powerful end-to-end trainable semantic segmentation models.

This framework enables the user to create models themselves but already provides pre-configured models.
- Structured Classifier: A single Structured Segmentation Algorithm.
- Encoder Decoder: An ensemble of multiple Structured Segmentation Algorithms

## Setup
Installation is done by cloning the repository
```bash
git clone https://github.com/FMuenke/structured_segmentation.git
```
All dependency can be installed with
````bash
pip install -r requirements.txt
````
It is strongly recommended to use a virtual environment (like anaconda).

## Example

An encoder-decoder model was used to segment cracks in images.

Input Image:

![Input Image of the EncoderDecoder Model](examples/example_0_image.jpg)

Ground-Truth (left) / Result (right):

![](examples/example_0_result.png)

## Tutorial
In the following we explain briefly how to train and evaluate a model yourself.
````python
import os
from data_structure.segmentation_data_set import SegmentationDataSet
from model import EncoderDecoder
from utils.utils import save_dict, check_n_make_dir

color_coding = {
    "class_1": [
      [255, 255, 255], #  Color Code of Class 1 on segmentation mask
      [255, 0, 0]      #  Color Code used for displaying results
    ],
    "class_2": [
      [4, 4, 4],       #  Color Code of Class 2 on segmentation mask
      [255, 0, 0]      #  Color Code used for displaying results
    ],
}

data_folder_train = "PATH_TO_TRAIN_DATA"
data_folder_test = "PATH_TO_TEST_DATA"
model_folder = "PATH_TO_STORE_MODEL"

model = EncoderDecoder()

data_set = SegmentationDataSet(data_folder_train, color_coding)
train_set = data_set.get_data()

check_n_make_dir(model_folder)
model.fit(train_set)
model.save(model_folder)
save_dict(color_coding, os.path.join(model_folder, "color_coding.json"))

data_set = SegmentationDataSet(data_folder_test, color_coding)
test_set = data_set.get_data()
model.evaluate(test_set, color_coding, results_folder=model_folder)

````
