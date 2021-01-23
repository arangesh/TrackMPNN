import os
import argparse
from datetime import datetime
import json

import torch

parser = argparse.ArgumentParser('Options for training TrackMPNN models in PyTorch...')

parser.add_argument('--dataset', type=str, default='kitti', help='dataset to train on: kitti/bdd100k')
parser.add_argument('--dataset-root-path', type=str, default='/home/akshay/data/kitti-mot', help='path to dataset')
parser.add_argument('--output-dir', type=str, default=None, help='output directory for model and logs')
parser.add_argument('--snapshot', type=str, default=None, help='use a pre-trained model snapshot')
parser.add_argument('--category', type=str, default='All', help='category to train model for')
parser.add_argument('--detections', type=str, default='centertrack', help='detections to use')
parser.add_argument('--feats', type=str, default='2d+temp+vis', help='features used for tracking: any combination of 2d, temp and vis')
parser.add_argument('--embed-arch', type=str, default='espv2', help='architecture to use for the visual embedding network: espv2/dla34')
parser.add_argument('--cur-win-size', type=int, default=5, help='number of timesteps in curring processing window')
parser.add_argument('--ret-win-size', type=int, default=0, help='number of timesteps in the past to be retained for association')
parser.add_argument('--hungarian', action='store_true', default=False, help='decode tracks using frame-by-frame Hungarian algorithm')
parser.add_argument('--no-tp-classifier', action='store_true', default=False, help='train network to only classify edges')
parser.add_argument('--num-hidden-feats', type=int, default=64, help='number of hidden layer nodes')
parser.add_argument('--num-att-heads', type=int, default=3, help='number of attention heads')
parser.add_argument('--msg-type', type=str, default='diff', help='type of message passing operation: diff/concat')
parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train for')
parser.add_argument('--learning-rate', type=float, default=0.0001, help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum for gradient step')
parser.add_argument('--weight-decay', type=float, default=0.0005, help='weight decay')
parser.add_argument('--log-schedule', type=int, default=10, help='number of iterations to print/save log after')
parser.add_argument('--seed', type=int, default=5, help='set seed to some constant value to reproduce experiments')
parser.add_argument('--no-cuda', action='store_true', default=False, help='do not use cuda for training')
parser.add_argument('--plot-gradients', action='store_true', default=False, help='plot gradient magnitudes during training')
parser.add_argument('--random-transforms', action='store_true', default=False, help='use random transforms for data augmentation')


args = parser.parse_args()

# check args
if args.dataset == 'kitti':
    if args.category not in ['Pedestrian', 'Car', 'Cyclist', 'All']:
        assert False, 'Unrecognized object category!'
    if args.detections not in ['centertrack', 'rrc']:
        assert False, 'Unrecognized detections!'
    if args.embed_arch not in ['espv2', 'dla34']:
        assert False, 'Unrecognized architecture specified for embedding network!'
elif args.dataset == 'bdd100k':
    if args.category not in ['pedestrian', 'rider', 'car', 'bus', 'truck', 'train', 'motorcycle', 'bicycle', 'All']:
        assert False, 'Unrecognized object category!'
    if args.detections not in ['hin']:
        assert False, 'Unrecognized detections!'
    if args.embed_arch not in ['espv2', 'dla34']:
        assert False, 'Unrecognized architecture specified for embedding network!'
else:
    assert False, 'Unrecognized dataset!'
# setup args
args.tp_classifier = not args.no_tp_classifier
args.cuda = not args.no_cuda and torch.cuda.is_available()
if args.output_dir is None:
    args.output_dir = datetime.now().strftime("%Y-%m-%d-%H:%M")
    args.output_dir = os.path.join('.', 'experiments', args.output_dir)

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)
    if args.plot_gradients:
        os.makedirs(os.path.join(args.output_dir, 'gradients'))
else:
    assert False, 'Output directory already exists!'

# store config in output directory
with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
    json.dump(vars(args), f)
