# SeFOSS

Official repository for WACV2024 paper [Improving Open-Set Semi-Supervised Learning with Self-Supervision](https://arxiv.org/abs/2301.10127)

Additionally, this repo contains Tensorflow implementations of FixMatch and DoubleMatch with RandAugment.

![SeFOSS graph](/media/SeFOSS-graph.png)

## Requirements

Python requirements are specified in requirements.txt.

## Preparation

Make sure the ssl-tf2-sefoss directory is in your Python path when running this code:
```bash
export PYTHONPATH=$PYTHONPATH:"path to this repo"
```

Set bash variables specifying where to store data and training results:
```bash
DATADIR="directory for storing data"
TRAINDIR="directory for storing checkpoints and results"
```

Optional: set logging level = 1 to disable info messages
```bash
export TF_CPP_MIN_LOG_LEVEL=1
```
(0 prints all messages, 1 disables info messages, 2 disables info & warning messages, 3 disables all messages)


## Datasets

This code reads data from tfrecord files.

To download data and prepare the tfrecord files, run
```bash
python3 scripts/create_datasets.py \
--datadir=$DATADIR
```

To create the labeled subsets, run for example
```bash
python3 scripts/create_split.py \
--seed=1 \
--size=4000 \
$DATADIR/SSL2/cifar10 \
$DATADIR/cifar10-train.tfrecord
```
which creates a labeled subset of CIFAR-10 with 4,000 samples using random seed 1.

## Training

To run SeFOSS using CIFAR-10 with 4,000 labels as ID and CIFAR-100, use
```bash
python3 sefoss_ossl.py \
--datadir=$DATADIR \
--traindir=$TRAINDIR \
--trainsteps=400000 \
--pretrainsteps=50000 \
--dataset=cifar10 \
--datasetood=cifar100 \
--nlabeled=4000 \
--ws=5.0 \
--arch=WRN-28-2 \
--seed=1
```


