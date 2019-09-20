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

## Dataset
1) Download the detection features (proposed by [MOTBeyondPixels](https://github.com/JunaidCS032/MOTBeyondPixels)) for the train and test split on the KITTI-MOTS dataset using [this link](https://drive.google.com/file/d/1xivQ4LC87vlpb4t_0nbkS_gTt81YxNdJ/view?usp=sharing)
2) Unzip the data

If you would like to re-generate these features, you can do so by using the scripts in the `utils/matlab` folder as follows:
1) Run `store_feats_train.m` after modifying the `root` variable
2) Run `store_feats_test.m` after modifying the `root` variable
3) Run `store_mean_std.m` after modifying the `root` variable

## Training
TrackMPNN can be trained using [this](https://github.com/arangesh/TrackMPNN/blob/master/train.py) script as follows:
```shell
pipenv shell # activate virtual environment
python train.py --dataset-root-path=/path/to/kitti-mots/ --timesteps=5
exit # exit virtual environment
```

## Inference
Inference can be carried out using [this](https://github.com/arangesh/TrackMPNN/blob/master/infer.py) script as follows:
```shell
pipenv shell # activate virtual environment
python infer.py --snapshot=/path/to/snapshot --dataset-root-path=/path/to/kitti-mots/ --timesteps=5
exit # exit virtual environment
```

Config files, logs, results and snapshots from running the above scripts will be stored in the `TrackMPNN/experiments` folder by default.
