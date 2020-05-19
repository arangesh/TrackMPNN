import os
import argparse
import torch
from datetime import datetime
import json

parser = argparse.ArgumentParser('Options for training Track-MPNN models in PyTorch...')

parser.add_argument('--dataset-root-path', type=str, default='/home/akshay/data/kitti-mot', help='path to dataset')
parser.add_argument('--output-dir', type=str, default=None, help='output directory for model and logs')
parser.add_argument('--snapshot', type=str, default=None, help='use a pre-trained model snapshot')
parser.add_argument('--timesteps', type=int, default=5, metavar='TS', help='number of timesteps to train on')
parser.add_argument('--hungarian', action='store_true', default=False, help='decode tracks using frame-by-frame Hungarian algorithm')
parser.add_argument('--no-tp-classifier', action='store_true', default=False, help='train network to only classify edges')
parser.add_argument('--img-feats', type=int, default=64, metavar='NIMG', help='number of image features')
parser.add_argument('--hidden', type=int, default=64, metavar='NH', help='number of hidden layer nodes')
parser.add_argument('--msg-type', type=str, default='diff', help='type of message passing operation (diff/concat)')
parser.add_argument('--epochs', type=int, default=100, metavar='EP', help='number of epochs to train for')
parser.add_argument('--learning-rate', type=float, default=0.0001, metavar='LR', help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='momentum for gradient step')
parser.add_argument('--weight-decay', type=float, default=0.0005, metavar='WD', help='weight decay')
parser.add_argument('--log-schedule', type=int, default=10, metavar='N', help='number of iterations to print/save log after')
parser.add_argument('--seed', type=int, default=1, help='set seed to some constant value to reproduce experiments')
parser.add_argument('--no-cuda', action='store_true', default=False, help='do not use cuda for training')
parser.add_argument('--random-transforms', action='store_true', default=False, help='use random transforms for data augmentation')


args = parser.parse_args()

# setup args
args.tp_classifier = not args.no_tp_classifier
args.cuda = not args.no_cuda and torch.cuda.is_available()
if args.output_dir is None:
    args.output_dir = datetime.now().strftime("%Y-%m-%d-%H:%M")
    args.output_dir = os.path.join('.', 'experiments', args.output_dir)

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)
else:
    assert False, 'Output directory already exists!'

# store config in output directory
with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
    json.dump(vars(args), f)
