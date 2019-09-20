import os
import argparse
import torch
from datetime import datetime
import json

parser = argparse.ArgumentParser('Options for testing Track-MPNN models in PyTorch...')

parser.add_argument('--snapshot', type=str, help='use a pre-trained model snapshot')
parser.add_argument('--dataset-root-path', type=str, default='/home/akshay/data/kitti-mots', help='path to dataset')
parser.add_argument('--output-dir', type=str, default=None, help='output directory for model and logs')
parser.add_argument('--timesteps', type=int, default=5, metavar='TS', help='number of timesteps to train on')
parser.add_argument('--hungarian', action='store_true', default=False, help='decode tracks using frame-by-frame Hungarian algorithm')
parser.add_argument('--no-tp-classifier', action='store_true', default=False, help='train network to only classify edges')
parser.add_argument('--seed', type=int, default=1, help='set seed to some constant value to reproduce experiments')
parser.add_argument('--no-cuda', action='store_true', default=False, help='do not use cuda for training')


args = parser.parse_args()

# setup args
args.tp_classifier = not args.no_tp_classifier
args.cuda = not args.no_cuda and torch.cuda.is_available()
if args.output_dir is None:
    args.output_dir = datetime.now().strftime("%Y-%m-%d-%H:%M-results")
    args.output_dir = os.path.join('.', 'experiments', args.output_dir)

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)
else:
    assert False, 'Output directory already exists!'

# store config in output directory
with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
    json.dump(vars(args), f)
