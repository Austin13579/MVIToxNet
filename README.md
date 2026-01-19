# Integrating Multiview Information for Enhanced Deep Learning-Based Acute Dermal Toxicity Prediction

This is a PyTorch implementation of MVIToxNet.

## Data splitting

Before running the codes, you need to split the datasets according to your requirements.
```
cd dataset/
mkdir datas
python split.py --ds Rabbit/Rat --rs 0/1/2/3/4/5/6/7/8/9
```

`rs` denotes random seed.

For convenience, you can use the `ds_split.sh`.
```
./ds_split.sh
```

## Training

Get Fingerprints first
```
cd src/
mkdir fps
python get_fp.py
```

Training
```
python train.py --ds Rabbit/Rat --rs 0/1/2/3/4/5/6/7/8/9
```

For convenience, you can use the `run.sh`.
```
./run.sh
```
