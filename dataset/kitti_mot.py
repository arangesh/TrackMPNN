import os
import json
import random
import numpy as np
from PIL import Image

import torch
from torch.utils import data
import torchvision.transforms as transforms

from models.loss import EmbeddingLoss
from models.dla.model import create_model, load_model


def get_tracking_data(dataset_path, split, timesteps):
    seqs = sorted(os.listdir(dataset_path))
    # seqs 13, 16 and 17 have very few or no cars at all
    if split == 'train':
        seqs = seqs[:-1]
        print(seqs)
    elif split == 'val':
        seqs = seqs[-1:]
        print(seqs)
    else:
        pass

    num_frames = [len(os.listdir(os.path.join(dataset_path, x))) for x in seqs]

    # Load tracking dataset; each row is [seq_no, st_fr, ed_fr]
    dataset = []
    if split == 'train':
        for i, seq in enumerate(seqs):
            for st_fr in range(0, num_frames[i], int(timesteps / 2)):
                dataset.append([seq, st_fr, min(st_fr + timesteps, num_frames[i])])
    else:
        for i, seq in enumerate(seqs):
            dataset.append([seq, 0, num_frames[i]])

    return dataset


def store_results_kitti(y_out, X, output_path):
    """
    This is a function that writes the result for the given sequence in KITTI format
    
    y_out [NUM_DETS, 2]: Array of outputs where each row is [ts, track_id]
    X [B, NUM_DETS, NUM_FEATS]: Features for all detections in a sequence
    output_path: Output file to write results in
    """
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    X = X.squeeze(0).detach().cpu().numpy().astype('float32') # (NUM_DETS, NUM_FEATS)
    times = np.sort(y_out[:, 0])
    t_st = times[0]
    t_ed = times[-1]

    with open(output_path, "w") as f:
        for t in range(t_st, t_ed+1):
            hids = np.where(np.logical_and(y_out[:, 0] == t, y_out[:, 1] != -1))[0]
            htracks = y_out[hids, 1]
            htracks = htracks.astype('int64')
            assert (htracks.size == np.unique(htracks).size), "Same track ID assigned to two detections from same timestep!"

            scores = X[hids, 0] * 0.2042707115 + 0.919026958912
            bboxs = X[hids, 1:5] * np.array([1242, 375, 1242, 375]) + np.array([1242, 375, 1242, 375]) / 2

            for i in range(scores.size):
                f.write("%d %d Car -1 -1 -10 %.2f %.2f %.2f %.2f -1 -1 -1 -1000 -1000 -1000 -10 %.2f\n" % (t, htracks[i], bboxs[i, 0], bboxs[i, 1], bboxs[i, 2], bboxs[i, 3], scores[i]))


class KittiMOTDataset(data.Dataset):
    def __init__(self, dataset_root_path=None, split='train', timesteps=5, num_img_feats=None, random_transforms=False, cuda=True):
        """Initialization"""

        if dataset_root_path is None:
            raise FileNotFoundError("Dataset Path needs to be valid")
        print('Preparing ' + split + ' dataset...')

        self.split = split
        self.timesteps = timesteps
        self.num_img_feats = num_img_feats # number of image based features to be used for tracking
        self.down_ratio = 4 # factor by which output feature maps are downsampled
        self.random_transforms = random_transforms
        self.cuda = cuda
        self.dropout = 0.2 # probability of a detection being dropped
        self.im_size_out = (384, 1280) # input image size for feature extraction

        if self.split == 'test':
            self.dataset_path = os.path.join(dataset_root_path, 'testing', 'gcn_features')
            self.im_path = os.path.join(dataset_root_path, 'testing', 'image_02')
        else:
            self.dataset_path = os.path.join(dataset_root_path, 'training', 'gcn_features')
            self.im_path = os.path.join(dataset_root_path, 'training', 'image_02')
            self.embed_loss = EmbeddingLoss()

        if self.num_img_feats is not None:
            # initialize detector with necessary heads and pretrained imagenet weights
            heads = {'hm': 3, 'wh':2, 'reg':2, 'dep': 1, 'rot': 8, 'dim': 3, 'trk': self.num_img_feats}
            self.detector = create_model(34, heads, 256)
            self.detector = load_model(self.detector, os.path.join('.', 'weights', 'model_last_kitti.pth'))
            #self.detector = get_pose_net(num_layers=34, heads=heads, head_conv=256, down_ratio=self.down_ratio)
        else:
            self.detector = None

        if random_transforms:
            self.image_transforms = transforms.Compose([
                transforms.Resize(self.im_size_out),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        else: 
            self.image_transforms = transforms.Compose([
                transforms.Resize(self.im_size_out),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

        # get tracking batch information 
        self.dataset = get_tracking_data(self.dataset_path, self.split, self.timesteps)
        # load mean and std values for each hand-crafted feature
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

    def get_trk_feats(self, feat_maps, bbox, im_size_in, down_ratio=4):
        """
        Extract image features from bbox center location
        """
        # get bbox center
        c_x = (bbox[0] + bbox[2]) / 2.0
        c_y = (bbox[1] + bbox[3]) / 2.0
        # account for image resizing
        c_x = (c_x * self.im_size_out[1]) / im_size_in[1]
        c_y = (c_y * self.im_size_out[0]) / im_size_in[0]
        # divide the center by downsampling ratio of the model
        c_x, c_y = round(c_x / down_ratio), round(c_y / down_ratio)

        # ensure center is within the feature map
        if c_x < 0:
            c_x = 0
        if c_y < 0:
            c_y = 0
        if c_x >= int(self.im_size_out[1] / down_ratio):
            c_x = int(self.im_size_out[1] / down_ratio) - 1
        if c_y >= int(self.im_size_out[0] / down_ratio):
            c_y = int(self.im_size_out[0] / down_ratio) - 1

        # extract features from center
        feat = feat_maps[:, :, c_y, c_x]
        return feat

    def __getitem__(self, index):
        """Generates one sample of data"""
        # get information for loading tracking sample
        input_info = self.dataset[index]

        # time reversal with probability 0.5
        random_transforms_tr = self.random_transforms and self.decision(0.5)
        # horizontal flip with probability 0.5
        random_transforms_hf = self.random_transforms and self.decision(0.5)

        im_feats, features, labels = [], [], []
        fr_range = range(input_info[1], input_info[2], 1)
        max_fr = max(fr_range)

        for fr in fr_range:
            # extract features for entire image
            im = Image.open(os.path.join(self.im_path, input_info[0], '%.6d.png' % (fr,)))
            im_size_in = [im.height, im.width]

            if random_transforms_hf:
                im = im.transpose(Image.FLIP_LEFT_RIGHT)
            im = self.image_transforms(im)
            if self.cuda:
                im = im.cuda()
            if self.split == 'train':
                detector_ops = self.detector(im.unsqueeze(0))[-1]
            else:
                with torch.no_grad(): # necessary to prevent gradient computation
                    detector_ops = self.detector(im.unsqueeze(0))[-1]

            with open(os.path.join(self.dataset_path, input_info[0], '%.6d.json' % (fr,))) as json_file:
                data = json.load(json_file)

                # store association features per object
                for d in range(len(data['score'])):
                    if not data['score'][d]:
                        continue

                    # random dropout
                    if self.random_transforms and self.decision(self.dropout):
                        continue

                    # hand-crafted features
                    # each feature vector for a detection in the sequence contains:
                    # [2d_bbox_score (1), 2d_bbox_coords [x1, y1, x2, y2] (4), keypoint_appearance_feats (64), 3d_convex_hull_coords (10/14)]
                    datum = [data['score'][d]]
                    bbox_2d = data['bbox_2d'][d]
                    convex_hull_3d = data['convex_hull_3d'][d]
                    appearance = data['appearance'][d]

                    # random horizontal flip
                    if random_transforms_hf:
                        bbox_2d = [1382 - bbox_2d[2], bbox_2d[1], 1382 - bbox_2d[0], bbox_2d[3]]
                        convex_hull_3d = [-x if i<len(convex_hull_3d)/2 else x for (i, x) in enumerate(convex_hull_3d)]
                        appearance = [appearance[x] for i in range(64, 0, -8) for x in range(i-8, i)]

                    # learned image features
                    im_feats.append(self.get_trk_feats(detector_ops['trk'], bbox_2d, im_size_in))

                    datum.extend(bbox_2d)
                    datum.extend(appearance)
                    datum.extend(convex_hull_3d)
                    features.append(datum)
                    # target labels for each detection in the sequence contains:
                    # [frame_no (1), track_id (1)]
                    if self.split == 'test':
                        labels.append([fr, -1]) # load dummy labels
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
            im_feats = torch.cat(im_feats, 0)

            if self.split == 'train':
                loss = self.embed_loss(im_feats, labels)
            else:
                loss = 0

            if self.cuda:
                features = features.cuda()
            # concatenate all features
            features = torch.cat((features[:, :5], im_feats, features[:, 5:]), 1)
            return features, labels, loss
        else:
            return features, labels, 0
