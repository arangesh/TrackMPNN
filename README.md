# TrackMPNN: A Message Passing Neural Network for End-to-End Multi-object Tracking

This is the Pytorch implementation for TrackMPNN - an end-to-end trainable multi-objecct tracker (MOT).

## Installation
1) Clone this repository
2) Install Pipenv:
```shell
pip3 install pipenv
```
3) Install all requirements and dependencies in a new virtual environment using Pipenv:
```shell
cd TrackMPNN
pipenv install
```
4) Get link for desired PyTorch wheel from [here](https://download.pytorch.org/whl/torch_stable.html) and install it in the Pipenv virtual environment as follows:
```shell
pipenv install https://download.pytorch.org/whl/cu100/torch-1.2.0-cp36-cp36m-manylinux1_x86_64.whl
```
Note that this repository additionally uses code from [PointNet.pytorch](https://github.com/fxia22/pointnet.pytorch) and [Graph Convolutional Networks in PyTorch](https://github.com/tkipf/pygcn).

5) Clone and make DCNv2:
```shell
cd models/dla
git clone https://github.com/CharlesShang/DCNv2
cd DCNv2
./make.sh
```

6) Download the [pre-trained detector weights](https://drive.google.com/file/d/10F9ZWpZ0SVHwg0xMKldIeKZenb7KktcH/view?usp=sharing) to the `TrackMPNN/weights` folder.

## Dataset
1) Download and extract the KITTI multi-object tracking (MOT) dataset (including images, labels, and calibration files).

## Training
TrackMPNN can be trained using [this](https://github.com/arangesh/TrackMPNN/blob/master/train.py) script as follows:
```shell
pipenv shell # activate virtual environment
python train.py --dataset-root-path=/path/to/kitti-mot/ --cur-win-size=5 --random-transforms
exit # exit virtual environment
```

## Inference
Inference can be carried out using [this](https://github.com/arangesh/TrackMPNN/blob/master/infer.py) script as follows:
```shell
pipenv shell # activate virtual environment
python infer.py --snapshot=/path/to/snapshot --dataset-root-path=/path/to/kitti-mot/ --timesteps=5
exit # exit virtual environment
```

Config files, logs, results and snapshots from running the above scripts will be stored in the `TrackMPNN/experiments` folder by default.
