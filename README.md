# TrackMPNN: A Message Passing Neural Network for End-to-End Multi-object Tracking

This is the Pytorch implementation for TrackMPNN - an end-to-end trainable multi-objecct tracker (MOT).

## Installation
1) Clone this repository
2) Install PyTorch
3) Install Pipenv:
```shell
pip3 install pipenv
```
4) Install other requirements and dependencies using Pipenv:
```shell
cd TrackMPNN
pipenv install
```

Note that this repository additionally uses code from [PointNet.pytorch](https://github.com/fxia22/pointnet.pytorch) and [Graph Convolutional Networks in PyTorch](https://github.com/tkipf/pygcn).

## Dataset
1) Download the detection features for the train and test split on the KITTI-MOTS dataset using [this link](https://drive.google.com/open?id=18hypBYy0pvFPUspnmZV2t8sSuZiTm3fy)
2) Unzip the data

## Training
TrackMPNN can be trained using [this](https://github.com/arangesh/TrackMPNN/blob/master/train.py) script as follows:

```shell
pipenv shell # activate virtual environment
python train.py --dataset-root-path=/path/to/kitti-mots/ --timesteps=5
```
