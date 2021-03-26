import os
import argparse
from datetime import datetime
import json

import torch

parser = argparse.ArgumentParser('Options for testing TrackMPNN models in PyTorch...')

parser.add_argument('--dataset-root-path', type=str, default='/home/akshay/data/kitti-mot', help='path to dataset')
parser.add_argument('--output-dir', type=str, default=None, help='output directory for model and logs')
parser.add_argument('--snapshot', type=str, help='use a pre-trained model snapshot')
parser.add_argument('--hungarian', action='store_true', default=False, help='decode tracks using frame-by-frame Hungarian algorithm')
parser.add_argument('--seed', type=int, default=5, help='set seed to some constant value to reproduce experiments')
parser.add_argument('--no-cuda', action='store_true', default=False, help='do not use cuda for training')
parser.add_argument('--feats', type=str, default='2d', help='features used for tracking: any combination of 2d, temp and vis')
parser.add_argument('--dataset', type=str, default='kitti', help='dataset to train on: kitti/bdd100k/mot20')
parser.add_argument('--category', type=str, default='All', help='category to train model for')
parser.add_argument('--detections', type=str, default='det', help='detections to use')
parser.add_argument('--embed-arch', type=str, default='espv2', help='architecture to use for the visual embedding network: espv2/dla34')
parser.add_argument('--cur-win-size', type=int, default=5, help='number of timesteps in curring processing window')
parser.add_argument('--ret-win-size', type=int, default=0, help='number of timesteps in the past to be retained for association')
parser.add_argument('--num-hidden-feats', type=int, default=64, help='number of hidden layer nodes')
parser.add_argument('--num-att-heads', type=int, default=0, help='number of attention heads')
parser.add_argument('--msg-type', type=str, default='diff', help='type of message passing operation: diff/concat')


args = parser.parse_args()

# setup args
args.cuda = not args.no_cuda and torch.cuda.is_available()
if args.output_dir is None:
    args.output_dir = datetime.now().strftime("%Y-%m-%d-%H:%M-infer")
    args.output_dir = os.path.join('.', 'experiments', args.output_dir)

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)
else:
    assert False, 'Output directory already exists!'

# load args used for training snapshot (if available)
if os.path.exists(os.path.join(os.path.dirname(args.snapshot), 'config.json')):
    with open(os.path.join(os.path.dirname(args.snapshot), 'config.json')) as f:
        json_args = json.load(f)
    # augment infer args with training args for model consistency
    args.dataset = json_args['dataset']
    args.category = json_args['category']
    args.detections = json_args['detections']
    args.feats = json_args['feats']
    args.embed_arch = json_args['embed_arch']
    args.cur_win_size = json_args['cur_win_size']
    args.ret_win_size = json_args['ret_win_size']
    args.no_tp_classifier = json_args['no_tp_classifier']
    args.num_hidden_feats = json_args['num_hidden_feats']
    args.num_att_heads = json_args['num_att_heads']
    args.msg_type = json_args['msg_type']
    args.tp_classifier = not args.no_tp_classifier

# store config in output directory
with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
    json.dump(vars(args), f)
