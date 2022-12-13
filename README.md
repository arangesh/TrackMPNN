# TrackMPNN: A Message Passing Neural Network for End-to-End Multi-object Tracking

This is the Pytorch implementation of TrackMPNN for the KITTI and BDD100K multi-object tracking (MOT) datasets.

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
git clone git@github.com:mez/DCNv2.git
cd DCNv2
./make.sh
```
6) Download the [imagenet pre-trained embedding network weights](https://drive.google.com/file/d/1fwnWaXftLBBARN_CG-BmzrBNfMNQaiAV/view?usp=sharing) to the `TrackMPNN/weights` folder.

## KITTI
### Dataset
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

### Training
TrackMPNN can be trained using RRC detections as follows:
```shell
pipenv shell # activate virtual environment
python train.py --dataset=kitti --dataset-root-path=/path/to/kitti-mot/ --cur-win-size=5 --detections=rrc --feats=2d --category=Car --no-tp-classifier --epochs=30 --random-transforms
exit # exit virtual environment
```
TrackMPNN can also be trained using CenterTrack detections as follows:
```shell
pipenv shell # activate virtual environment
python train.py --dataset=kitti --dataset-root-path=/path/to/kitti-mot/ --cur-win-size=5 --detections=centertrack --feats=2d --category=All --no-tp-classifier --epochs=50 --random-transforms
exit # exit virtual environment
```
By default, the model is trained to track `All` object categories, but you can supply the `--category` argument with any one of the following options: `['Pedestrian', 'Car', 'Cyclist', 'All']`.

### Inference
Inference on the `testing` split can be carried out using [this](https://github.com/arangesh/TrackMPNN/blob/master/infer.py) script as follows:
```shell
pipenv shell # activate virtual environment
python infer.py --snapshot=/path/to/snapshot.pth --dataset-root-path=/path/to/kitti-mot/ --hungarian
exit # exit virtual environment
```
All settings from training will be carried over for inference.

Config files, logs, results and snapshots from running the above scripts will be stored in the `TrackMPNN/experiments` folder by default.


### Visualizing Inference Results 
You can use the `utils/visualize_mot.py` script to generate a video of the tracking results after running the inference script:
```shell
pipenv shell # activate virtual environment
python utils/visualize_mot.py /path/to/testing/inference/results /path/to/kitti-mot/testing/image_02
exit
```
The videos will be stored in the same folder as the inference results.

## BDD100K
### Dataset
1) Download and extract the BDD100K multi-object tracking (MOT) dataset (including images, labels, and calibration files).
2) Download the [HIN and Libra detections](https://drive.google.com/file/d/1PIDr9GcTayXw7GtmQ_R7IMm4YiT1eU_W/view?usp=sharing) for `training`, `validation` and `testing` splits and add them to the BDD100K MOT folder. The dataset should be organized as follows:
```plain
└── bdd100k-mot
    ├── training/
    |   └── calib/
    |   └── image_02/
    |   └── label_02/
    |   └── hin_detections/
    |   └── libra_detections/
    ├── validation/
    |   └── calib/
    |   └── image_02/
    |   └── label_02/
    |   └── hin_detections/
    |   └── libra_detections/
    └── testing/
        └── calib/
        └── image_02/
        └── hin_detections/
        └── libra_detections/
```

### Training
TrackMPNN can be trained using HIN detections as follows:
```shell
pipenv shell # activate virtual environment
python train.py --dataset=bdd100k --dataset-root-path=/path/to/bdd100k-mot/ --cur-win-size=5 --detections=libra --feats=2d --category=All --no-tp-classifier --epochs=20  --random-transforms
exit # exit virtual environment
```
By default, the model is trained to track `All` object categories, but you can supply the `--category` argument with any one of the following options: `['pedestrian', 'rider', 'car', 'bus', 'truck', 'train', 'motorcycle', 'bicycle', 'All']`.

### Inference
Inference on the `testing` split can be carried out using [this](https://github.com/arangesh/TrackMPNN/blob/master/infer.py) script as follows:
```shell
pipenv shell # activate virtual environment
python infer.py --snapshot=/path/to/snapshot.pth --dataset-root-path=/path/to/bdd100k-mot/ --hungarian
exit # exit virtual environment
```
All settings from training will be carried over for inference.

Config files, logs, results and snapshots from running the above scripts will be stored in the `TrackMPNN/experiments` folder by default.


### Visualizing Inference Results 
You can use the `utils/visualize_mot.py` script to generate a video of the tracking results after running the inference script:
```shell
pipenv shell # activate virtual environment
python utils/visualize_mot.py /path/to/testing/inference/results /path/to/bdd100k-mot/testing/image_02
exit
```
The videos will be stored in the same folder as the inference results.