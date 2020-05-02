import os
import numpy as np
import json
import random
from torch.utils import data

from utils.dataset import get_tracking_data


def decision(probability):
    """
    Function that returns True with probability
    """
    return random.random() < probability


class KittiMOTDataset(data.Dataset):
    def __init__(self, dataset_root_path=None, split='train', timesteps=5, random_transforms=False):
        """Initialization"""

        if dataset_root_path is None:
            raise FileNotFoundError("Dataset Path needs to be valid")
        print('Preparing ' + split + ' dataset...')

        self.split = split
        self.timesteps = timesteps
        self.random_transforms = random_transforms
        self.dropout = 0.2 # probability of a detection being dropped
        if self.split == 'test':
            self.dataset_path = os.path.join(dataset_root_path, 'testing', 'gcn_features')
        else:
            self.dataset_path = os.path.join(dataset_root_path, 'training', 'gcn_features')

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

    def __getitem__(self, index):
        """Generates one sample of data"""
        input_info = self.dataset[index]

        # time reversal with probability 0.5
        random_transforms_tr = self.random_transforms and decision(0.5)
        # horizontal flip with probability 0.5
        random_transforms_hf = self.random_transforms and decision(0.5)

        features, labels = [], []
        fr_range = range(input_info[1], input_info[2], 1)
        max_fr = max(fr_range)

        for fr in fr_range:
            with open(os.path.join(self.dataset_path, input_info[0], '%.6d.json' % (fr,))) as json_file:
                data = json.load(json_file)
                for d in range(len(data['score'])):
                    if not data['score'][d]:
                        continue
                    # random dropout
                    if self.random_transforms and decision(self.dropout):
                        continue

                    # each feature vector for a detection in the sequence contains:
                    # [2d_bbox_score (1), 2d_bbox_coords [x1, y1, x2, y2] (4), keypoint_appearance_feats (64), 3d_convex_hull_coords (10)]
                    datum = [data['score'][d]]
                    bbox_2d = data['bbox_2d'][d]
                    appearance = data['appearance'][d]
                    convex_hull_3d = data['convex_hull_3d'][d]

                    if random_transforms_hf:
                        bbox_2d = [1382 - bbox_2d[2], bbox_2d[1], 1382 - bbox_2d[0], bbox_2d[3]]
                        appearance = [appearance[x] for i in range(64, 0, -8) for x in range(i-8, i)]
                        convex_hull_3d = [-x if i<len(convex_hull_3d)/2 else x for (i, x) in enumerate(convex_hull_3d)]

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
            labels = np.array(labels, dtype='int64')
        return features, labels
