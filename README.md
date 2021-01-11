# TrackMPNN: A Message Passing Neural Network for End-to-End Multi-object Tracking

This is the Pytorch implementation for TrackMPNN for the KITTI MOT dataset.

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
5) Clone and make DCNv2 (note gcc-8 is highest supported version incase you're on ubuntu 20.04 +):
```shell
cd models/dla
git clone https://github.com/CharlesShang/DCNv2
cd DCNv2
./make.sh
```
6) Download the [imagenet pre-trained embedding network weights](https://drive.google.com/file/d/1fwnWaXftLBBARN_CG-BmzrBNfMNQaiAV/view?usp=sharing) to the `TrackMPNN/weights` folder.

## Dataset
1) Download and extract the KITTI multi-object tracking (MOT) dataset (including images, labels, and calibration files).
2) Download the [RRC and CenterTrack detections](https://drive.google.com/file/d/1PIDr9GcTayXw7GtmQ_R7IMm4YiT1eU_W/view?usp=sharing) for both `training` and `testing` splits and add them to the KITTI MOT folder. The dataset should be organized as follows:
```plain
└── kitti-mot
    ├── training/
    |   └── calib/
    |   └── image_02/
    |   └── label_02/
    |   └── rrc_detections/
    |   └── centertrack_detections/
    └── testing/
        └── calib/
        └── image_02/
        └── rrc_detections/
        └── centertrack_detections/
```

## Training
TrackMPNN can be trained for RRC detections as follows:
```shell
pipenv shell # activate virtual environment
python train.py --dataset-root-path=/path/to/kitti-mot/ --cur-win-size=5 --detections=rrc --feats=2d --category=Car --no-tp-classifier --random-transforms
exit # exit virtual environment
```
TrackMPNN can also be trained for CenterTrack detections as follows:
```shell
pipenv shell # activate virtual environment
python train.py --dataset-root-path=/path/to/kitti-mot/ --cur-win-size=5 --detections=centertrack --feats=2d --category=All --no-tp-classifier --random-transforms
exit # exit virtual environment
```

## Inference
Inference on the `testing` split can be carried out using [this](https://github.com/arangesh/TrackMPNN/blob/master/infer.py) script as follows:
```shell
pipenv shell # activate virtual environment
python infer.py --snapshot=/path/to/snapshot.pth --dataset-root-path=/path/to/kitti-mot/ --hungarian
exit # exit virtual environment
```
All settings from training will be carried over for inference.

Config files, logs, results and snapshots from running the above scripts will be stored in the `TrackMPNN/experiments` folder by default.


## Visualizing Inference Results 
You can use the `utils/visualize_mot.py` script to generate a video of the tracking results after running the inference script:
```shell
pipenv shell # activate virtual environment
python utils/visualize_mot.py /path/to/testing/inference/results /path/to/kitti-mot/testing/image_02
exit
```
The videos will be stored in the same folder as the inference results.
