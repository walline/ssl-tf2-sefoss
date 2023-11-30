# ProSub

Official repository for WACV2024 paper [Improving Open-Set Semi-Supervised Learning with Self-Supervision](https://arxiv.org/abs/2301.10127)

Additionally, this repo contains Tensorflow implementations of FixMatch and DoubleMatch with RandAugment.

## Requirements

Python requirements are specified in requirements.txt.

Make sure the the ssl-tf2-sefoss directory is in your Python path when running this code.

## Datasets

This code reads data from tfrecord files.

To download data and prepare the tfrecord files, run
```bash
python3 scripts/generate_datasets.py \
--datadir="directory for saving tfrecord files"
```

To create the labeled subsets, run for example
```bash
DATADIR="directory containing tfrecord files"
python3 scripts/create_split.py
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
--datadir="directory containing your tfrecord files" \
--traindir="directory for storing checkpoints and results" \
--trainsteps=400000 \
--pretrainsteps=50000 \
--dataset=cifar10 \
--datasetood=cifar100 \
--nlabeled=4000 \
--ws=5.0 \
--arch=WRN-28-2 \
--seed=1
```


