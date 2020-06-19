import os
import json
import random
import numpy as np
from scipy.optimize import linear_sum_assignment
import cv2

import torch
from torch.utils import data
import torchvision.transforms as transforms

from utils.misc import vectorized_iou
from models.loss import EmbeddingLoss
from models.dla.ddd import DddDetector


def store_kitti_results(bbox_pred, y_out, class_dict, output_path):
    """
    This is a function that writes the result for the given sequence in KITTI format
    
    bbox_pred [num_dets, (cat_id, alpha, x1, y1, x2, y2, h, w, l, x, y, z, rotation_y, score)]: Predicted bboxes in a sequence
    y_out [num_dets, (frame, track_id)]: Predicted tracks where each row is [ts, track_id]
    class_dict = {'Pedestrian': 1, 'Car': 2, 'Cyclist': 3}
    output_path: Output file to write results in
    """
    class_dict = {v: k for k, v in class_dict.items()}
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    times = np.sort(y_out[:, 0])
    t_st = times[0]
    t_ed = times[-1]

    with open(output_path, "w") as f:
        for t in range(t_st, t_ed+1):
            hids = np.where(np.logical_and(y_out[:, 0] == t, y_out[:, 1] != -1))[0]
            htracks = y_out[hids, 1]
            htracks = htracks.astype('int64')
            assert (htracks.size == np.unique(htracks).size), "Same track ID assigned to two detections from same timestep!"

            cat_ids = bbox_pred[hids, 0]
            alphas = bbox_pred[hids, 1]
            bboxs = bbox_pred[hids, 2:6]
            heights = bbox_pred[hids, 6]
            widths = bbox_pred[hids, 7]
            lengths = bbox_pred[hids, 8]
            locs_x = bbox_pred[hids, 9]
            locs_y = bbox_pred[hids, 10]
            locs_z = bbox_pred[hids, 11]
            r_ys = bbox_pred[hids, 12]
            scores = bbox_pred[hids, 13]            

            for i in range(scores.size):
                f.write("%d %d %s -1 -1 %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f\n" % \
                        (t, htracks[i], class_dict[int(cat_ids[i])], alphas[i], bboxs[i, 0], bboxs[i, 1], bboxs[i, 2], \
                        bboxs[i, 3], heights[i], widths[i], lengths[i], locs_x[i], locs_y[i], locs_z[i], r_ys[i], scores[i]))


class KittiMOTDataset(data.Dataset):
    def __init__(self, dataset_root_path=None, split='train', cat='Car', timesteps=5, num_img_feats=4, random_transforms=False, cuda=True):
        """Initialization"""

        if dataset_root_path is None:
            raise FileNotFoundError("Dataset path needs to be valid")
        print('Preparing ' + split + ' dataset...')

        self.split = split
        self.class_dict = {'Pedestrian': 1, 'Car': 2, 'Cyclist': 3}
        self.cat = cat
        self.timesteps = timesteps
        self.num_img_feats = num_img_feats # number of image based features to be used for tracking
        self.down_ratio = 4 # factor by which output feature maps are downsampled
        self.random_transforms = random_transforms
        self.cuda = cuda
        self.dropout_ratio = 0.2 # probability of a detection being dropped

        if self.split == 'test':
            self.feature_path = os.path.join(dataset_root_path, 'testing', 'gcn_features')
            self.im_path = os.path.join(dataset_root_path, 'testing', 'image_02')
            self.label_path = None
        else:
            self.feature_path = os.path.join(dataset_root_path, 'training', 'gcn_features')
            self.im_path = os.path.join(dataset_root_path, 'training', 'image_02')
            self.label_path = os.path.join(dataset_root_path, 'training', 'label_02')

        if self.split == 'train':
            self.embed_loss = EmbeddingLoss()

        # initialize detector with necessary heads and pretrained weights
        if self.split == 'train':
            self.detector = DddDetector(self.cuda, 'train', num_img_feats=num_img_feats, dataset='kitti')
        elif self.split == 'val':
            # do not initialize a second detector for val (will use the same one as train)
            self.detector = None
        elif self.split == 'test':
            self.detector = DddDetector(self.cuda, 'test', num_img_feats=num_img_feats, dataset='kitti')

        # get tracking batch information 
        self.chunks = self.get_tracking_chunks()
        # load mean values for each feature
        mean = [0.90] + [621, 187.5, 621, 187.5] # 2d features
        mean = mean + [1.53, 1.63, 3.88] + [0.0, 0.8, 35.70] + [0.0] # 3d features
        mean = mean + [0.0 for _ in range(self.num_img_feats)] # image features
        self.mean = self.convert_to_tensor(np.array([mean], dtype='float32'))
        # load std values for each feature
        std = [0.20] + [1242.0, 375.0, 1242.0, 375.0] # 2d features
        std = std + [0.14, 0.10, 0.43] + [123.95, 1.0, 395.67] + [np.pi] # 3d features
        std = std + [1.0 for _ in range(self.num_img_feats)] # image features
        self.std = self.convert_to_tensor(np.array([std], dtype='float32'))

        print('Finished preparing ' + self.split + ' dataset!')

    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.chunks)

    def convert_to_tensor(self, var):
        """
        Handles conversion of ndarrays/datatypes to cuda/normal tensors
        """
        if isinstance(var, np.ndarray):
            var = torch.from_numpy(var)
        if self.cuda:
            var = var.cuda()
        return var

    def decision(self, probability):
        """
        Function that returns True with probability
        """
        return random.random() < probability

    def get_tracking_chunks(self):
        seqs = sorted(os.listdir(self.feature_path))
        # seqs 13, 16 and 17 have very few or no cars at all
        if self.split == 'train':
            seqs = seqs[:-1]
            print(seqs)
        elif self.split == 'val':
            seqs = seqs[-1:]
            print(seqs)
        else:
            pass

        num_frames = [len(os.listdir(os.path.join(self.feature_path, x))) for x in seqs]

        # Load tracking chunks; each row is [seq_no, st_fr, ed_fr]
        chunks = []
        if self.split == 'train':
            for i, seq in enumerate(seqs):
                for st_fr in range(0, num_frames[i], int(self.timesteps / 2)):
                    chunks.append([seq, st_fr, min(st_fr + self.timesteps, num_frames[i])])
        else:
            for i, seq in enumerate(seqs):
                chunks.append([seq, 0, num_frames[i]])

        return chunks

    def load_kitti_labels(self, seq, fr):
        """
        Values    Name      Description
        ----------------------------------------------------------------------------
           1    frame        Frame within the sequence where the object appearers
           1    track id     Unique tracking id of this object within this sequence
           1    type         Describes the type of object: 'Car', 'Van', 'Truck',
                             'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram',
                             'Misc' or 'DontCare'
           1    truncated    Integer (0,1,2) indicating the level of truncation.
                             Note that this is in contrast to the object detection
                             benchmark where truncation is a float in [0,1].
           1    occluded     Integer (0,1,2,3) indicating occlusion state:
                             0 = fully visible, 1 = partly occluded
                             2 = largely occluded, 3 = unknown
           1    alpha        Observation angle of object, ranging [-pi..pi]
           4    bbox         2D bounding box of object in the image (0-based index):
                             contains left, top, right, bottom pixel coordinates
           3    dimensions   3D object dimensions: height, width, length (in meters)
           3    location     3D object location x,y,z in camera coordinates (in meters)
           1    rotation_y   Rotation ry around Y-axis in camera coordinates [-pi..pi]
           1    score        Only for results: Float, indicating confidence in
                             detection, needed for p/r curves, higher is better.
        """
        annotations = []
        bbox_gt = np.zeros((0, 16), dtype=np.float32)
        if self.label_path is None:
            return annotations, bbox_gt

        label_file = open(os.path.join(self.label_path, seq + '.txt'), 'r')
        for line in label_file:
            tmp = line[:-1].split(' ')
            if int(tmp[0]) < fr:
                continue
            elif int(tmp[0]) > fr:
                break

            cat = tmp[2]
            if cat != self.cat:
                continue

            ann = { 'frame': fr,
                    'track_id': int(tmp[1]),
                    'category_id': self.class_dict[self.cat],
                    'dim': [float(tmp[10]), float(tmp[11]), float(tmp[12])],
                    'bbox': [float(tmp[6]), float(tmp[7]), float(tmp[8]), float(tmp[9])],
                    'depth': float(tmp[15]),
                    'alpha': float(tmp[5]),
                    'truncated': int(float(tmp[3])),
                    'occluded': int(tmp[4]),
                    'location': [float(tmp[13]), float(tmp[14]), float(tmp[15])],
                    'rotation_y': float(tmp[16])}
            annotations.append(ann)
            # [fr, trk_id, cat_id, alpha, x1, y1, x2, y2, h, w, l, x, y, z, rotation_y, score]
            b = [ann['frame'], ann['track_id'], self.class_dict[self.cat]] \
                + [ann['alpha']] + ann['bbox'] + ann['dim'] + ann['location'] \
                + [ann['rotation_y']] + [1]
            bbox_gt = np.concatenate((bbox_gt, np.array([b], dtype=np.float32)), axis=0)
        return annotations, bbox_gt

    def get_trk_feats(self, feat_maps, bboxes, im_size):
        """
        Extract image features from bbox center location
        """
        feats = []
        for bbox in bboxes:
            # get bbox center
            c_x = (bbox[0] + bbox[2]) / 2.0
            c_y = (bbox[1] + bbox[3]) / 2.0
            # account for image resizing
            c_x = (c_x * self.detector.opt.input_w) / im_size[0]
            c_y = (c_y * self.detector.opt.input_h) / im_size[1]
            # divide the center by downsampling ratio of the model
            c_x = int(c_x / self.detector.opt.down_ratio)
            c_y = int(c_y / self.detector.opt.down_ratio)

            # extract features from center
            feats.append(feat_maps[:, :, c_y, c_x])
        if len(feats) == 0:
            return self.convert_to_tensor(np.zeros((0, self.num_img_feats), dtype='float32'))
        feats = torch.cat(feats, dim=0)
        return feats

    def get_track_ids(self, bbox_pred, bbox_gt, iou_thresh=0.5):
        """
        Function to assign each detection a track_id based on GT or -1 if false positive
        bbox_pred: [N_pred, (fr, -1, cat_id, alpha, x1, y1, x2, y2, h, w, l, x, y, z, rotation_y, score)]
        bbox_gt: (N_gt, (fr, trk_id, cat_id, alpha, x1, y1, x2, y2, h, w, l, x, y, z, rotation_y, score)]
        """
        if bbox_pred.size == 0 or bbox_gt.size == 0:
            return bbox_pred

        if not np.all(np.equal(bbox_pred[:, 0:1], bbox_gt[:, 0:1].T)):
            assert False, "Detections and GT boxes not from same frame!"

        # calculate iou matrix
        ious = vectorized_iou(bbox_pred[:, 4:8], bbox_gt[:, 4:8])
        # cost matrix
        C = 1.0 - ious
        # optimal assignment
        row_ind, col_ind = linear_sum_assignment(C)

        for row, col in zip(row_ind, col_ind):
            if C[row, col] < iou_thresh:
                if bbox_pred[row, 1] < 0: # if unassigned
                    bbox_pred[row, 1] = bbox_gt[col, 1]
                else: # if already assigned
                    assert False, "Same detection assigned to two tracks!"

        return bbox_pred

    def __getitem__(self, index):
        """Generates one sample of data"""
        # get information for loading tracking sample
        input_info = self.chunks[index]

        # time reversal with probability 0.5
        random_transforms_tr = self.random_transforms and self.decision(0.5)
        # horizontal flip with probability 0.5
        random_transforms_hf = self.random_transforms and self.decision(0.5)

        # intiliaze empty arrays to store predicted and GT bboxes
        bbox_pred = np.zeros((0, 16), dtype=np.float32)
        bbox_gt = np.zeros((0, 16), dtype=np.float32)
        im_feats = []

        for fr in range(input_info[1], input_info[2], 1):
            # load image
            im = cv2.imread(os.path.join(self.im_path, input_info[0], '%.6d.png' % (fr,)))
            
            # load GT annotations
            # [fr, trk_id, cat_id, alpha, x1, y1, x2, y2, h, w, l, x, y, z, rotation_y, score]
            annotations, bbox_gt_fr = self.load_kitti_labels(input_info[0], fr)

            # apply horizontal flip transform
            if random_transforms_hf:
                # for predicted bboxes, just flipping the images to suffice
                im = cv2.flip(im, 1)
                # transform GT bboxes
                bbox_gt_fr_old = bbox_gt_fr
                bbox_gt_fr[:, 3] = -bbox_gt_fr_old[:, 3] # alpha
                bbox_gt_fr[:, 4] = im.shape[1] - bbox_gt_fr[:, 6] # x1
                bbox_gt_fr[:, 6] = im.shape[1] - bbox_gt_fr[:, 4] # x2
                bbox_gt_fr[:, 11] = -bbox_gt_fr_old[:, 11] # X
                bbox_gt_fr[:, 14] = np.pi - bbox_gt_fr_old[:, 14] # rotation_y

            # run forward pass through detector
            detector_ops = self.detector.run(im)
            # object classes = {1: 'Pedestrian', 2: 'Car', 3: 'Cyclist'}
            # [num_dets, (alpha, x1, y1, x2, y2, h, w, l, x, y, z, rotation_y, score)]
            detections = detector_ops['results'][self.class_dict[self.cat]]
            # find bboxes for the frame
            # [fr, -1, cat_id, alpha, x1, y1, x2, y2, h, w, l, x, y, z, rotation_y, score]
            bbox_pred_fr = np.zeros((detections.shape[0], 16), dtype=np.float32)
            bbox_pred_fr[:, 0] = fr
            bbox_pred_fr[:, 1] = -1
            bbox_pred_fr[:, 2] = self.class_dict[self.cat]
            bbox_pred_fr[:, 3:] = detections

            # apply time reversal transform
            if random_transforms_tr:
                bbox_pred_fr[:, 0] = input_info[2] - bbox_pred_fr[:, 0]
                bbox_gt_fr[:, 0] = input_info[2] - bbox_gt_fr[:, 0]

            # assign GT track ids to each predicted bbox
            bbox_pred_fr = self.get_track_ids(bbox_pred_fr, bbox_gt_fr)

            # random dropout of bboxes
            if self.random_transforms:
                ret_idx = [not self.decision(self.dropout_ratio) for _ in range(bbox_pred_fr.shape[0])]
                bbox_pred_fr = bbox_pred_fr[ret_idx, :]

            # append to existing bboxes in the sequence
            bbox_pred = np.concatenate((bbox_pred, bbox_pred_fr), axis=0)
            bbox_gt = np.concatenate((bbox_gt, bbox_gt_fr), axis=0)
            im_feats.append(self.get_trk_feats(detector_ops['outputs']['trk'], bbox_pred_fr[:, 4:8], 
                (im.shape[1], im.shape[0])))

        # features for tracker
        # [score, x1, y1, x2, y2]
        two_d_feats = self.convert_to_tensor(bbox_pred[:, [15, 4, 5, 6, 7]])
        # [h, w, l, X, Y, Z, rotation_y]
        three_d_feats = self.convert_to_tensor(bbox_pred[:, [8, 9, 10, 11, 12, 13, 14]])
        # image features
        im_feats = torch.cat(im_feats, 0)
        # (num_dets, 5 + 7 + num_img_feats)
        features = torch.cat((two_d_feats, three_d_feats, im_feats), 1)
        print(features.size())

        if features.size()[0] != 0:
            features = (features - self.mean) / self.std # normalize/standardize features
            if self.split == 'train':
                loss = self.embed_loss(features[:, -self.num_img_feats:], bbox_pred[:, :2].astype('int64'))
            else:
                loss = self.convert_to_tensor(torch.tensor(0.0))
        else:
            loss = self.convert_to_tensor(torch.tensor(0.0))
        return features, bbox_pred, bbox_gt, loss
