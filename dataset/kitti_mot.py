import os
import numpy as np
import json
import random

from PIL import Image
import torch
import torch.nn as nn
from torch.utils import data
import torchvision.transforms as transforms

from utils.dataset import get_tracking_data
from models.dla.pose_dla_dcn import get_pose_net


class KittiMOTDataset(data.Dataset):
    def __init__(self, dataset_root_path=None, split='train', timesteps=5, num_img_feats=None, random_transforms=False, cuda=True):
        """Initialization"""

        if dataset_root_path is None:
            raise FileNotFoundError("Dataset Path needs to be valid")
        print('Preparing ' + split + ' dataset...')

        self.split = split
        self.timesteps = timesteps
        self.num_img_feats = num_img_feats
        self.down_ratio = 4
        self.random_transforms = random_transforms
        self.cuda = cuda
        self.dropout = 0.2 # probability of a detection being dropped

        if self.split == 'test':
            self.dataset_path = os.path.join(dataset_root_path, 'testing', 'gcn_features')
            self.im_path = os.path.join(dataset_root_path, 'testing', 'image_02')
        else:
            self.dataset_path = os.path.join(dataset_root_path, 'training', 'gcn_features')
            self.im_path = os.path.join(dataset_root_path, 'training', 'image_02')

        if self.num_img_feats is not None:
            heads = {'hm': 3, 'depth': 1, 'rotation': 8, 'dim': 3, 'trk_feats': self.num_img_feats}
            self.detector = get_pose_net(num_layers=34, heads=heads, head_conv=256, down_ratio=self.down_ratio)
        else:
            self.detector = None

        if random_transforms:
            self.image_transforms = transforms.Compose([
                transforms.Resize((192, 640)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        else: 
            self.image_transforms = transforms.Compose([
                transforms.Resize((192, 640)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

        self.dataset = get_tracking_data(self.dataset_path, self.split, self.timesteps)
        with open(os.path.join(dataset_root_path, 'gcn_features_mean.json')) as json_file:
            data = json.load(json_file)
            mean = [data['score']]
            mean.extend(data['bbox_2d'])
            mean.extend(data['appearance'])
            mean.extend(data['convex_hull_3d'])
            self.mean = np.array([mean], dtype='float32')
        with open(os.path.join(dataset_root_path, 'gcn_features_std.json')) as json_file:
            data = json.load(json_file)
            std = [data['score']]
            std.extend(data['bbox_2d'])
            std.extend(data['appearance'])
            std.extend(data['convex_hull_3d'])
            self.std = np.array([std], dtype='float32')
        
        print('Finished preparing ' + self.split + ' dataset!')

    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.dataset)

    def decision(self, probability):
        """
        Function that returns True with probability
        """
        return random.random() < probability

    def get_trk_feats(self, feat_maps, bbox, down_ratio=4):
        """
        Extract image features from bbox center location
        """
        c_x = (bbox[0] + bbox[2]) / 2.0
        c_y = (bbox[1] + bbox[3]) / 2.0
        c_x, c_y = round(c_x / down_ratio), round(c_y / down_ratio)
        if c_x > 159:
            c_x = 159
        if c_y > 47:
            c_y = 47
        feat = feat_maps[:, :, c_y, c_x]
        return feat

    def __getitem__(self, index):
        """Generates one sample of data"""
        input_info = self.dataset[index]

        # time reversal with probability 0.5
        random_transforms_tr = self.random_transforms and self.decision(0.5)
        # horizontal flip with probability 0.5
        random_transforms_hf = self.random_transforms and self.decision(0.5)

        im_feats, features, labels = [], [], []
        fr_range = range(input_info[1], input_info[2], 1)
        max_fr = max(fr_range)

        for fr in fr_range:
            im = Image.open(os.path.join(self.im_path, input_info[0], '%.6d.png' % (fr,)))
            if random_transforms_hf:
                im = im.transpose(Image.FLIP_LEFT_RIGHT)
            im = self.image_transforms(im)
            if self.cuda:
                im = im.cuda()
            if self.split == 'train':
                detector_ops = self.detector(im.unsqueeze(0))[-1]
            else:
                with torch.no_grad():
                    detector_ops = self.detector(im.unsqueeze(0))[-1]
            
            with open(os.path.join(self.dataset_path, input_info[0], '%.6d.json' % (fr,))) as json_file:
                data = json.load(json_file)

                for d in range(len(data['score'])):
                    if not data['score'][d]:
                        continue
                    # random dropout
                    if self.random_transforms and self.decision(self.dropout):
                        continue

                    # each feature vector for a detection in the sequence contains:
                    # [2d_bbox_score (1), 2d_bbox_coords [x1, y1, x2, y2] (4), keypoint_appearance_feats (64), 3d_convex_hull_coords (10/14)]
                    datum = [data['score'][d]]
                    bbox_2d = data['bbox_2d'][d]
                    convex_hull_3d = data['convex_hull_3d'][d]
                    appearance = data['appearance'][d]

                    if random_transforms_hf:
                        bbox_2d = [1382 - bbox_2d[2], bbox_2d[1], 1382 - bbox_2d[0], bbox_2d[3]]
                        convex_hull_3d = [-x if i<len(convex_hull_3d)/2 else x for (i, x) in enumerate(convex_hull_3d)]
                        appearance = [appearance[x] for i in range(64, 0, -8) for x in range(i-8, i)]

                    im_feats.append(self.get_trk_feats(detector_ops['trk_feats'], bbox_2d))

                    datum.extend(bbox_2d)
                    datum.extend(appearance)
                    datum.extend(convex_hull_3d)
                    features.append(datum)
                    # target labels for each detection in the sequence contains:
                    # [frame_no (1), track_id (1)]
                    if self.split == 'test':
                        labels.append([fr, -1])
                    else:
                        if random_transforms_tr: # time reversal
                            labels.append([max_fr - fr, data['track_id'][d]])
                        else:
                            labels.append([fr, data['track_id'][d]])

        if len(features) != 0 and len(labels) != 0:
            features = (np.array(features, dtype='float32') - self.mean) / self.std # normalize/standardize features
            features = features[:, list(range(5)) + list(range(69, features.shape[1]))]
            labels = np.array(labels, dtype='int64')
        
            features = torch.from_numpy(features)
            if self.cuda:
                features = features.cuda()
            features = torch.cat((features[:, :5], torch.cat(im_feats, 0), features[:, 5:]), 1)
        return features, labels
