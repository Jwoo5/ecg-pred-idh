# Requirements
* [PyTorch](https://pytorch.org) version >= 1.5.0
* Python version >= 3.6
* **To preprocess ECG datasets**: `pip install scipy==1.3.1`

# Getting started

## Pre-process
Before pre-processing, please ensure that your data directory structure is like:
```
path/to/ECG_digitized
├─ F00001/
├─ ...
└─ F97801
   ├─ lead_0.png
   ├─ ...
   └─ lead_12.png
```

Given the root directory, `ECG_digitized`, containing ECG images (.png) and path to `ECG_Datasheet_sess.csv`, run:
```shell script
$ python preprocess/preprocess.py \
    /path/to/ECG_digitized \
    --dest /path/to/data \
    --csv-path /path/to/ECG_Datasheet_sess.csv
```
It will convert ECG images (.png) to waveform data (.mat), and output them into `--dest` directory.

Note that it might take around 3 hours totally.

## Prepare data manifest
Given the root directory containing pre-processed data, run:
```shell script
$ python preprocess/manifest.py \
    /path/to/data \
    --dest /path/to/manifest \
    --valid-percent $N
```
`$N` should be set to reasonble number (`<= 0.5`) to split the dataset into train, valid, and test set.

# Train
```shell script
$ CUDA_VISIBLE_DEVICES=$N python train.py \
    --data /path/to/manifest \
```
Note: set `$N` to denote what gpus you will use to train the model.  
It will train the model with default configurations.

If you want to train model with your own configurations, add appropriate arguments to the command.  
The available arguments are described in `train.py`