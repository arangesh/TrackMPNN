import os
import numpy as np
import json
from torch.utils import data

from utils.dataset import get_tracking_data


class KittiMOTSDataset(data.Dataset):
    def __init__(self, dataset_root_path=None, split='train', timesteps=10):
        """Initialization"""

        if dataset_root_path is None:
            raise FileNotFoundError("Dataset Path needs to be valid")

        print('Preparing ' + split + ' dataset...')
        self.split = split
        self.timesteps = timesteps
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

        print('Finished preparing ' + split + ' dataset!')

    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.dataset)

    def __getitem__(self, index):
        """Generates one sample of data"""
        input_info = self.dataset[index]

        features, labels = [], []
        for t, fr in enumerate(range(input_info[1], input_info[2])):
            with open(os.path.join(self.dataset_path, input_info[0], '%.6d.json' % (fr,))) as json_file:
                data = json.load(json_file)
                for d in range(len(data['track_id'])):
                    if not data['track_id'][d]:
                        continue
                    # each feature vector for a detection in the sequence contains:
                    # [2d_bbox_score (1), 2d_bbox_coords (4), keypoint_appearance_feats (64), 3d_convex_hull_coords (10)]
                    datum = [data['score'][d]]
                    datum.extend(data['bbox_2d'][d])
                    datum.extend(data['appearance'][d])
                    datum.extend(data['convex_hull_3d'][d])
                    features.append(datum)
                    # target labels for each detection in the sequence contains:
                    # [frame_no (1), track_id (1)]
                    labels.append([t, data['track_id'][d]])

        if len(features) != 0 and len(labels) != 0:
            features = (np.array(features, dtype='float32') - self.mean) / self.std  # normalize/standardize features
            labels = np.array(labels, dtype='int64')
        return features, labels
