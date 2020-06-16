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


def get_tracking_data(feature_path, split, timesteps):
    seqs = sorted(os.listdir(feature_path))
    # seqs 13, 16 and 17 have very few or no cars at all
    if split == 'train':
        seqs = seqs[:-1]
        print(seqs)
    elif split == 'val':
        seqs = seqs[-1:]
        print(seqs)
    else:
        pass

    num_frames = [len(os.listdir(os.path.join(feature_path, x))) for x in seqs]

    # Load tracking chunks; each row is [seq_no, st_fr, ed_fr]
    chunks = []
    if split == 'train':
        for i, seq in enumerate(seqs):
            for st_fr in range(0, num_frames[i], int(timesteps / 2)):
                chunks.append([seq, st_fr, min(st_fr + timesteps, num_frames[i])])
    else:
        for i, seq in enumerate(seqs):
            chunks.append([seq, 0, num_frames[i]])

    return chunks


def store_results_kitti(bbox_pred, y_out, output_path):
    """
    This is a function that writes the result for the given sequence in KITTI format
    
    bbox_pred [NUM_DETS_PRED, (x1, y1, x2, y2, score)]: Predicted bboxes in a sequence
    y_out [NUM_DETS_PRED, (frame, track_id)]: Predicted tracks where each row is [ts, track_id]
    output_path: Output file to write results in
    """
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

            scores = bbox_pred[hids, 4]
            bboxs = bbox_pred[hids, :4]

            for i in range(scores.size):
                f.write("%d %d Car -1 -1 -10 %.2f %.2f %.2f %.2f -1 -1 -1 -1000 -1000 -1000 -10 %.2f\n" % (t, htracks[i], bboxs[i, 0], bboxs[i, 1], bboxs[i, 2], bboxs[i, 3], scores[i]))


def load_kitti_labels(label_path, input_info):
    """
    label_path: /path/to/kitti/split/label_02
    info: [seq, st_fr, ed_fr]
    #Values    Name      Description
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
    annotations = [[] for fr in range(input_info[1], input_info[2])]
    bbox_gt = np.zeros((0, 7), dtype=np.float32)
    if label_path is None:
        return annotations, bbox_gt

    label_file = open(os.path.join(label_path, input_info[0] + '.txt'), 'r')
    for line in label_file:
        tmp = line[:-1].split(' ')

        fr = int(tmp[0])
        if fr < input_info[1]:
            continue
        if fr >= input_info[2]:
            break
        cat_id = tmp[2]
        if cat_id != 'Car':
            continue

        ann = { 'frame': fr,
                'track_id': int(tmp[1]),
                'category_id': 0,
                'dim': [float(tmp[10]), float(tmp[11]), float(tmp[12])],
                'bbox': [float(tmp[6]), float(tmp[7]), float(tmp[8]), float(tmp[9])],
                'depth': float(tmp[15]),
                'alpha': float(tmp[5]),
                'truncated': int(float(tmp[3])),
                'occluded': int(tmp[4]),
                'location': [float(tmp[13]), float(tmp[14]), float(tmp[15])],
                'rotation_y': float(tmp[16])}
        annotations[fr - input_info[1]].append(ann)
        b = [ann['frame'], ann['track_id']] + ann['bbox'] + [1]
        bbox_gt = np.concatenate((bbox_gt, np.array([b], dtype=np.float32)), axis=0)
    return annotations, bbox_gt


class KittiMOTDataset(data.Dataset):
    def __init__(self, dataset_root_path=None, split='train', timesteps=5, num_img_feats=4, random_transforms=False, cuda=True):
        """Initialization"""

        if dataset_root_path is None:
            raise FileNotFoundError("Dataset path needs to be valid")
        print('Preparing ' + split + ' dataset...')

        self.split = split
        self.timesteps = timesteps
        self.num_img_feats = num_img_feats # number of image based features to be used for tracking
        self.down_ratio = 4 # factor by which output feature maps are downsampled
        self.random_transforms = random_transforms
        self.cuda = cuda
        self.dropout_ratio = 0.2 # probability of a detection being dropped
        self.im_size_out = (384, 1280) # input image size for feature extraction

        if self.split == 'test':
            self.feature_path = os.path.join(dataset_root_path, 'testing', 'gcn_features')
            self.im_path = os.path.join(dataset_root_path, 'testing', 'image_02')
            self.label_path = None
        else:
            self.feature_path = os.path.join(dataset_root_path, 'training', 'gcn_features')
            self.im_path = os.path.join(dataset_root_path, 'training', 'image_02')
            self.label_path = os.path.join(dataset_root_path, 'training', 'label_02')
            self.embed_loss = EmbeddingLoss()

        if self.split == 'train':
            # initialize detector with necessary heads and pretrained weights
            # object classes = ['Pedestrian', 'Car', 'Cyclist']
            heads = {'hm': 3, 'wh':2, 'reg':2, 'dep': 1, 'rot': 8, 'dim': 3, 'trk': self.num_img_feats}
            self.detector = create_model(34, heads, 256)
            self.detector = load_model(self.detector, os.path.join('.', 'weights', 'model_last_kitti.pth'))
            self.detector = self.convert_to_tensor(self.detector)
        else:
            self.detector = None

        self.image_transforms = transforms.Compose([
            transforms.Resize(self.im_size_out),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

        # get tracking batch information 
        self.chunks = get_tracking_data(self.feature_path, self.split, self.timesteps)
        # load mean and std values for each hand-crafted feature
        with open(os.path.join(dataset_root_path, 'gcn_features_mean.json')) as json_file:
            data = json.load(json_file)
            mean = [0 for _ in range(self.num_img_feats)]
            mean.extend([data['score']])
            mean.extend(data['bbox_2d'])
            mean.extend(data['convex_hull_3d'])
            self.mean = self.convert_to_tensor(np.array([mean], dtype='float32'))
        with open(os.path.join(dataset_root_path, 'gcn_features_std.json')) as json_file:
            data = json.load(json_file)
            std = [1 for _ in range(self.num_img_feats)]
            std.extend([data['score']])
            std.extend(data['bbox_2d'])
            std.extend(data['convex_hull_3d'])
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

    def get_trk_feats(self, feat_maps, bbox, im_size_in):
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
        c_x, c_y = round(c_x / self.down_ratio), round(c_y / self.down_ratio)

        # ensure center is within the feature map
        if c_x < 0:
            c_x = 0
        if c_y < 0:
            c_y = 0
        if c_x >= int(self.im_size_out[1] / self.down_ratio):
            c_x = int(self.im_size_out[1] / self.down_ratio) - 1
        if c_y >= int(self.im_size_out[0] / self.down_ratio):
            c_y = int(self.im_size_out[0] / self.down_ratio) - 1

        # extract features from center
        feat = feat_maps[:, :, c_y, c_x]
        return feat

    def __getitem__(self, index):
        """Generates one sample of data"""
        # get information for loading tracking sample
        input_info = self.chunks[index]

        # time reversal with probability 0.5
        random_transforms_tr = self.random_transforms and self.decision(0.5)
        # horizontal flip with probability 0.5
        random_transforms_hf = self.random_transforms and self.decision(0.5)

        im_feats, features, labels, bbox_pred = [], [], [], []
        fr_range = range(input_info[1], input_info[2], 1)

        # load GT annotations
        annotations, bbox_gt = load_kitti_labels(self.label_path, input_info)
        # apply transforms to GT boxes
        if random_transforms_hf:
            bbox_gt = np.stack((bbox_gt[:, 0], bbox_gt[:, 1], 1382 - bbox_gt[:, 4], 
                bbox_gt[:, 3], 1382 - bbox_gt[:, 2], bbox_gt[:, 5], bbox_gt[:, 6]), axis=-1)
        if random_transforms_tr:
            bbox_gt = np.stack((input_info[2] - bbox_gt[:, 0], bbox_gt[:, 1], bbox_gt[:, 2], 
                bbox_gt[:, 3], bbox_gt[:, 4], bbox_gt[:, 5], bbox_gt[:, 6]), axis=-1)

        for fr in fr_range:
            # load and preprocess image
            im = Image.open(os.path.join(self.im_path, input_info[0], '%.6d.png' % (fr,)))
            im_size_in = [im.height, im.width]
            im = self.convert_to_tensor(self.image_transforms(im))

            # run forward pass through detector
            if self.split == 'train':
                detector_ops = self.detector(im.unsqueeze(0))[-1]
            else:
                with torch.no_grad(): # necessary to prevent gradient computation
                    detector_ops = self.detector(im.unsqueeze(0))[-1]

            # get tracking features for each detection
            with open(os.path.join(self.feature_path, input_info[0], '%.6d.json' % (fr,))) as json_file:
                data = json.load(json_file)

                # store association features per object
                for d in range(len(data['score'])):
                    if not data['score'][d]:
                        continue

                    # random dropout
                    if self.random_transforms and self.decision(self.dropout_ratio):
                        continue

                    # load features
                    score = [data['score'][d]]
                    bbox_2d = data['bbox_2d'][d]
                    convex_hull_3d = data['convex_hull_3d'][d]
                    im_feats = self.get_trk_feats(detector_ops['trk'], bbox_2d, im_size_in)

                    # random horizontal flip
                    if random_transforms_hf:
                        bbox_2d = [1382 - bbox_2d[2], bbox_2d[1], 1382 - bbox_2d[0], bbox_2d[3]]
                        convex_hull_3d = [-x if i<len(convex_hull_3d)/2 else x for (i, x) in enumerate(convex_hull_3d)]

                    # target labels for each detection in the sequence contains:
                    # [frame_no (1), track_id (1)]
                    if self.split == 'test':
                        labels.append([fr, -1]) # load dummy labels
                    else:
                        # time reversal
                        if random_transforms_tr:
                            labels.append([input_info[2] - fr, data['track_id'][d]])
                        else:
                            labels.append([fr, data['track_id'][d]])

                    # create bbox_pred array for MOT metrics calculation
                    bbox_pred.append([labels[-1][0], -1] + bbox_2d + score)

                    # each feature vector for a detection in the sequence contains:
                    # [image_feats(self.num_img_feats), 2d_bbox_score (1), 2d_bbox_coords [x1, y1, x2, y2] (4), 3d_convex_hull_coords (10/14)]
                    score = self.convert_to_tensor(np.array([score], dtype=np.float32))
                    bbox_2d = self.convert_to_tensor(np.array([bbox_2d], dtype=np.float32))
                    convex_hull_3d = self.convert_to_tensor(np.array([convex_hull_3d], dtype=np.float32))
                    features.append(torch.cat((im_feats, score, bbox_2d, convex_hull_3d), 1))

        if len(features) != 0 and len(labels) != 0:
            features = (torch.cat(features, 0) - self.mean) / self.std # normalize/standardize features
            labels = np.array(labels, dtype='int64')
            if self.split == 'train':
                loss = self.embed_loss(features[:, :self.num_img_feats], labels)
            else:
                loss = self.convert_to_tensor(torch.tensor(0.0))
            bbox_pred = np.array(bbox_pred, dtype=np.float32)
        else:
            loss = self.convert_to_tensor(torch.tensor(0.0))
            bbox_pred = np.zeros((0, 7), dtype=np.float32)
        return features, labels, loss, bbox_pred, bbox_gt
