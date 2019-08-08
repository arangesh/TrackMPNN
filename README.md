# TrackMPNN: A Message Passing Neural Network for End-to-End Multi-object Tracking

This is the Pytorch implementation for TrackMPNN - an end-to-end trainable multi-objecct tracker (MOT).

## Installation
1) Clone this repository.
2) Install PyTorch
3) Install other requirements:
```shell
pip install -r requirements.txt
```
or
```shell
conda install --file requirements.txt
```
4) Install [py-motmetrics](https://github.com/cheind/py-motmetrics) to keep track of MOT metrics during training and inference.

Note that this repository additionally uses code from [PointNet.pytorch](https://github.com/fxia22/pointnet.pytorch) and [Graph Convolutional Networks in PyTorch](https://github.com/tkipf/pygcn)

## Training
TrackMPNN can be trained using [this](https://github.com/arangesh/TrackMPNN/blob/master/train.py) script as follows:

```shell
python train.py --dataset-path=/path/to/dataset/ --timesteps=10
```
