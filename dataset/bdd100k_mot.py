import os
import glob
import json
import random
import numpy as np
from scipy.optimize import linear_sum_assignment
import cv2
import PIL

import torch
import torch.nn.functional as F
from torch.utils import data
import torch.optim as optim
import torchvision.transforms as transforms

from utils.misc import vectorized_iou, vectorized_iom
from models.dla.pose_dla_dcn import get_pose_net
from models.espv2.SegmentationModel import EESPNet_Seg
from models.loss import EmbeddingLoss, FairMOTLoss


def store_bdd100k_results(bbox_pred, y_out, class_dict, output_path):
    """
    This is a function that writes the result for the given sequence in BDD100k format
    
    bbox_pred [num_dets, (cat_id, alpha, x1, y1, x2, y2, h, w, l, x, y, z, rotation_y, score)]: Predicted bboxes in a sequence
    y_out [num_dets, (frame, track_id)]: Predicted tracks where each row is [ts, track_id]
    class_dict = {'pedestrian': 1, 'rider': 2, 'car': 3, 'bus': 4, 'truck': 5, 'train': 6, 'motorcycle':7, 'bicycle':8}
    output_path: Output file to write results in
    """
    class_dict = {v: k for k, v in class_dict.items()}
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    times = np.sort(y_out[:, 0])
    t_st = times[0]
    t_ed = times[-1]
    data = []

    for t in range(t_st, t_ed+1):
        hids = np.where(np.logical_and(y_out[:, 0] == t, y_out[:, 1] != -1))[0]
        htracks = y_out[hids, 1]
        htracks = htracks.astype('int32')
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

        labels = []
        for i in range(scores.size):
            box2d = {'x1':bboxs[i, 0], 'y1':bboxs[i, 1], 'x2':bboxs[i, 2], 'y2':bboxs[i, 3]}
            labels.append({'id':htracks[i], 'category':class_dict[int(cat_ids[i])], 'box2d':box2d})
        d = {'name':os.path.basename(output_path), 'videoName':os.path.basename(output_path), 'frameIndex':int(t), 'labels':labels}
        data.append(d)

    with open(output_path, "w") as f:
        json.dump(data, f)


class BDD100kMOTDataset(data.Dataset):
    def __init__(self, dataset_root_path=None, split='train', cat='All', detections='hin', feats='2d+temp+vis', embed_arch='espv2', cur_win_size=5, ret_win_size=10, snapshot=None, random_transforms=False, cuda=True):
        """Initialization"""

        if dataset_root_path is None:
            raise FileNotFoundError("Dataset path needs to be valid")
        print('Preparing ' + split + ' dataset...')

        self.split = split
        self.class_dict = {'pedestrian': 1, 'rider': 2, 'car': 3, 'bus': 4, 'truck': 5, 'train': 6, 'motorcycle':7, 'bicycle':8}
        self.distractors = {'other person':9, 'trailer':9, 'other vehicle':9, 'crowd':-1}
        if cat == 'All':
            self.cats = list(self.class_dict.keys()) + list(self.distractors.keys())
        else:
            self.cats = [cat] + list(self.distractors.keys())
        self.detections = detections
        self.feats = feats
        self.embed_arch = embed_arch
        self.cur_win_size = cur_win_size
        self.ret_win_size = ret_win_size
        self.num_vis_feats = 128 # number of visual features to be used for tracking
        self.input_h, self.input_w = 720, 1280
        self.snapshot = snapshot
        self.random_transforms = random_transforms
        self.cuda = cuda
        self.dropout_ratio = 0.2 # probability of a detection being dropped
        self.fr_range = 30

        if self.split == 'test':
            self.im_path = os.path.join(dataset_root_path, 'testing', 'image_02')
            self.label_path = None
            self.detections_path = os.path.join(dataset_root_path, 'testing', self.detections + '_detections') 
        elif self.split == 'train':
            self.im_path = os.path.join(dataset_root_path, 'training', 'image_02')
            self.label_path = os.path.join(dataset_root_path, 'training', 'label_02')
            self.detections_path = os.path.join(dataset_root_path, 'training', self.detections + '_detections')
        elif self.split == 'val':
            self.im_path = os.path.join(dataset_root_path, 'validation', 'image_02')
            self.label_path = os.path.join(dataset_root_path, 'validation', 'label_02')
            self.detections_path = os.path.join(dataset_root_path, 'validation', self.detections + '_detections')
        else:
            assert False, 'Incorrect split provided to dataloader!'

        # initialize detector with necessary heads and pretrained weights
        if 'vis' in self.feats:
            if self.split == 'train':
                if self.embed_arch == 'espv2':
                    self.down_ratio = 1
                    self.embed_net = EESPNet_Seg(classes=self.num_vis_feats, s=1, pretrained='./weights/espnetv2_s_1.0.pth')
                    # optimizer for detector
                    self.optimizer = optim.Adam(self.embed_net.parameters(), 5e-4, (0.9, 0.999), eps=1e-08, weight_decay=5e-4)
                elif self.embed_arch == 'dla34':
                    self.down_ratio = 4
                    self.embed_net = get_pose_net(num_layers=34, heads={'trk': self.num_vis_feats}, head_conv=256, down_ratio=self.down_ratio)
                    # optimizer for detector
                    self.optimizer = optim.Adam(self.embed_net.parameters(), lr=1.25e-4)
                if self.snapshot is not None:
                     self.embed_net.load_state_dict(torch.load(self.snapshot), strict=True)
                if self.cuda:
                    self.embed_net.cuda()
                #self.embed_loss = EmbeddingLoss()
                self.embed_loss = FairMOTLoss(self.num_vis_feats)
            elif self.split == 'val':
                if self.embed_arch == 'espv2':
                    self.down_ratio = 1
                elif self.embed_arch == 'dla34':
                    self.down_ratio = 4
                # do not initialize a second detector for val (will use the same one as train)
                self.embed_net = None
            elif self.split == 'test':
                if self.embed_arch == 'espv2':
                    self.down_ratio = 1
                    self.embed_net = EESPNet_Seg(classes=self.num_vis_feats, s=1, pretrained='./weights/espnetv2_s_1.0.pth')
                elif self.embed_arch == 'dla34':
                    self.down_ratio = 4
                    self.embed_net = get_pose_net(num_layers=34, heads={'trk': self.num_vis_feats}, head_conv=256, down_ratio=self.down_ratio)
                if self.snapshot is not None:
                     self.embed_net.load_state_dict(torch.load(self.snapshot), strict=True)
                if self.cuda:
                    self.embed_net.cuda()

        # get tracking batch information 
        self.chunks = self.get_tracking_chunks()
        # load mean values for each feature
        mean = [0.5 for _ in range(len(self.class_dict))] # one-hot category IDs
        if '2d' in self.feats:
            if self.detections == 'hin':
                mean = mean + [0.94] + [545.84, 329.28, 85.19, 71.47] # 2d features
            elif self.detections == 'libra':
                mean = mean + [0.94] + [545.84, 329.28, 85.19, 71.47] # 2d features
        if 'temp' in self.feats:
            mean = mean + [0.0 for _ in range(2*1)] # temporal features
        if 'vis' in self.feats:
            mean = mean + [0.5 for _ in range(self.num_vis_feats)] # visual features
        self.mean = self._convert_to_tensor([mean])
        # load std values for each feature
        std = [0.5 for _ in range(len(self.class_dict))] # one-hot category IDs
        if '2d' in self.feats:
            if self.detections == 'hin':
                std = std + [0.07] + [294.88, 81.51, 93.51, 75.72] # 2d features
            elif self.detections == 'libra':
                std = std + [0.07] + [294.88, 81.51, 93.51, 75.72] # 2d features
        if 'temp' in self.feats:
            std = std + [1.0 for _ in range(2*1)] # temporal features
        if 'vis' in self.feats:
            std = std + [0.5 for _ in range(self.num_vis_feats)] # visual features
        self.std = self._convert_to_tensor([std])

        print('Finished preparing ' + self.split + ' dataset!')

    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.chunks)

    def _convert_to_tensor(self, var):
        """
        Handles conversion of ndarrays/datatypes to cuda/normal tensors
        """
        if type(var) == type([]):
            var = np.array(var, dtype=np.float32)
        if isinstance(var, np.ndarray):
            var = torch.from_numpy(var)
        if self.cuda:
            var = var.cuda()
        return var

    def _decision(self, probability):
        """
        Function that returns True with probability
        """
        return random.random() < probability

    def get_tracking_chunks(self):
        seqs = sorted(os.listdir(self.im_path))
        num_frames = [len(glob.glob(os.path.join(self.im_path, x, '*.jpg'))) for x in seqs]

        # Load tracking chunks; each row is [seq_no, st_fr, ed_fr]
        chunks = []
        if self.split == 'train':
            for i, seq in enumerate(seqs):
                for st_fr in range(0, num_frames[i], int(self.cur_win_size)):
                    fr_list = [fr for fr in range(st_fr, min(st_fr + self.cur_win_size, num_frames[i]))]
                    skip_fr = random.randint(st_fr + self.cur_win_size, st_fr + self.cur_win_size + self.ret_win_size)
                    if skip_fr < num_frames[i] - 1:
                        fr_list = fr_list + [skip_fr, skip_fr + 1]
                    chunks.append([seq, fr_list])
        else:
            for i, seq in enumerate(seqs):
                chunks.append([seq, [fr for fr in range(0, num_frames[i])]])

        return chunks

    def load_bdd100k_labels(self, seq, fr, im_shape, random_transforms_hf):
        """
        Values    Name      Description
        ----------------------------------------------------------------------------
           1    frame        Frame within the sequence where the object appearers
           1    track id     Unique tracking id of this object within this sequence
           1    type         Describes the type of object
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

        cat_ids = {**self.class_dict, **self.distractors}
        label_file = open(os.path.join(self.label_path, seq + '.txt'), 'r')
        for line in label_file:
            tmp = line[:-1].split(' ')
            if int(tmp[0]) < fr:
                continue
            elif int(tmp[0]) > fr:
                break

            ann = { 'frame': fr,
                    'track_id': int(tmp[1]),
                    'category_id': cat_ids[tmp[2]],
                    'dim': [float(tmp[10]), float(tmp[11]), float(tmp[12])],
                    'bbox': [float(tmp[6]), float(tmp[7]), float(tmp[8]), float(tmp[9])],
                    'depth': float(tmp[15]),
                    'alpha': float(tmp[5]),
                    'truncated': int(float(tmp[3])),
                    'occluded': int(tmp[4]),
                    'location': [float(tmp[13]), float(tmp[14]), float(tmp[15])],
                    'rotation_y': float(tmp[16])}
            if random_transforms_hf:
                # transform GT bboxes to account for horizontal flip
                ann['alpha'] = -ann['alpha'] # alpha
                ann['bbox'] = [im_shape[1]-ann['bbox'][2]-1, ann['bbox'][1], 
                               im_shape[1]-ann['bbox'][0]-1, ann['bbox'][3]] # x1 and x2
                ann['location'] = [-ann['location'][0], ann['location'][1], ann['location'][2]] # X
                if ann['rotation_y'] >= -np.pi and ann['rotation_y'] <= -np.pi/2:
                    ann['rotation_y'] = np.pi/2 + ann['rotation_y']
                elif ann['rotation_y'] > -np.pi/2 and ann['rotation_y'] <= 0:
                    ann['rotation_y'] =  -np.pi/2 + ann['rotation_y']
                elif ann['rotation_y'] > 0 and ann['rotation_y'] <= np.pi/2:
                    ann['rotation_y'] = np.pi/2 + ann['rotation_y']
                elif ann['rotation_y'] > np.pi/2 and ann['rotation_y'] <= np.pi:
                    ann['rotation_y'] = -np.pi/2 + ann['rotation_y']
            annotations.append(ann)

            if tmp[2] not in self.cats:
                continue

            # [fr, trk_id, cat_id, alpha, x1, y1, x2, y2, h, w, l, x, y, z, rotation_y, score]
            b = [ann['frame'], ann['track_id'], ann['category_id']] \
                + [ann['alpha']] + ann['bbox'] + ann['dim'] \
                + ann['location'] + [ann['rotation_y']] + [1]
            bbox_gt = np.concatenate((bbox_gt, np.array([b], dtype=np.float32)), axis=0)
        return annotations, bbox_gt

    def load_detections(self, seq, fr, im_shape, random_transforms_hf):
        """
        Values    Name      Description
        ----------------------------------------------------------------------------
           1    frame        Frame within the sequence where the object appearers
           1    track id     Unique tracking id of this object within this sequence
           1    type         Describes the type of object
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
        bbox_pred = np.zeros((0, 16), dtype=np.float32)
        if self.detections_path is None:
            return bbox_pred

        cat_ids = {**self.class_dict, **self.distractors}
        try:
            det_file = open(os.path.join(self.detections_path, seq, '%.4d.txt' % (fr,)), 'r')
        except:
            return bbox_pred
        for line in det_file:
            tmp = line[:-1].split(',')

            ann = { 'frame': fr,
                    'category_id': cat_ids[tmp[0]],
                    'bbox': [float(tmp[1]), float(tmp[2]), float(tmp[3]), float(tmp[4])],
                    'score': float(tmp[5])}
            if random_transforms_hf:
                # transform GT bboxes to account for horizontal flip
                ann['bbox'] = [im_shape[1]-ann['bbox'][2]-1, ann['bbox'][1], 
                               im_shape[1]-ann['bbox'][0]-1, ann['bbox'][3]] # x1 and x2

            if tmp[0] not in self.cats:
                continue
            if tmp[0] in list(self.distractors.keys()): # remove boxes related to distractors
                continue
            if ann['score'] <= 0.2:
                continue

            # [fr, -1, cat_id, -10, x1, y1, x2, y2, -1, -1, -1, -1000, -1000, -1000, -10, score]
            b = [ann['frame'], -1, ann['category_id']] \
                + [-10] + ann['bbox'] + [-1, -1, -1] \
                + [-1000, -1000, -1000] + [-10] + [ann['score']]
            bbox_pred = np.concatenate((bbox_pred, np.array([b], dtype=np.float32)), axis=0)
        return bbox_pred

    def get_embed_net_outputs(self, im):
        """
        Run forward pass through embedding network
        """
        # resizing transform
        _transform = transforms.Resize((self.input_h, self.input_w))
        # convert PIL Image (H, W, C), uint8 --> torch tensor (C, H, W), float32
        _to_tensor = transforms.ToTensor()
        # normalization transform for input images
        _normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        im_tensor = _normalize(_to_tensor(_transform(im)))
        if self.cuda:
            im_tensor = im_tensor.cuda()

        if self.split == 'train':
            outputs = self.embed_net(im_tensor.unsqueeze(0))
        else:
            with torch.no_grad():
                outputs = self.embed_net(im_tensor.unsqueeze(0))
        if self.embed_arch == 'dla34':
            outputs = outputs[-1]['trk']
        return outputs

    def get_vis_feats(self, feat_maps, bboxes, im_shape):
        """
        Extract visual features from bbox center location
        """
        feats = []
        for bbox in bboxes:
            # get bbox center
            c_x = (bbox[0] + bbox[2]) / 2.0
            c_y = (bbox[1] + bbox[3]) / 2.0
            # account for image resizing
            c_x = (c_x * self.input_w) / im_shape[1]
            c_y = (c_y * self.input_h) / im_shape[0]
            # divide the center by downsampling ratio of the model
            c_x = int(c_x / self.down_ratio)
            c_y = int(c_y / self.down_ratio)

            # extract features from center
            feats.append(feat_maps[:, :, c_y, c_x])
        if len(feats) == 0:
            return self._convert_to_tensor(np.zeros((0, self.num_vis_feats), dtype='float32'))
        feats = torch.cat(feats, dim=0)
        return feats

    def get_temp_feats(self, frames):
        """
        Extract bounded, cyclic features representing frame
        """
        feats = np.mod(frames, self.fr_range) * np.pi / self.fr_range
        feats = np.concatenate((np.sin(feats), np.cos(feats)), axis=1)
        return feats

    def get_track_ids(self, bbox_pred, bbox_gt, iou_thresh=0.5, iom_thresh=0.8):
        """
        Function to assign each detection a track_id based on GT or -1 if false positive
        bbox_pred: [N_pred, (fr, -1, cat_id, alpha, x1, y1, x2, y2, h, w, l, x, y, z, rotation_y, score)]
        bbox_gt: (N_gt, (fr, trk_id, cat_id, alpha, x1, y1, x2, y2, h, w, l, x, y, z, rotation_y, score)]
        """
        if bbox_gt.size == 0:
            return bbox_pred, bbox_gt

        bbox_crowd = bbox_gt[bbox_gt[:, 2] == -1, :] # get crowd regions in image
        bbox_gt = bbox_gt[bbox_gt[:, 2] != -1, :] # remove crowd regions from GT
        bbox_distract = bbox_gt[bbox_gt[:, 2] == 9, :] # get distractor regions in image
        bbox_gt = bbox_gt[bbox_gt[:, 2] != 9, :] # remove distractor regions from GT

        if bbox_pred.size == 0:
            return bbox_pred, bbox_gt

        if not np.all(np.equal(bbox_pred[:, 0:1], bbox_gt[:, 0:1].T)):
            assert False, "Detections and GT boxes not from same frame!"

        # assign track IDs based on IoU with GT in descending order
        if bbox_gt.size > 0:
            # calculate iou matrix
            ious = vectorized_iou(bbox_pred[:, 4:8], bbox_gt[:, 4:8])
            # indices sorted by iou
            (rows, cols) = np.unravel_index(np.argsort(ious, axis=None), ious.shape)

            gt_assigned = -1*np.ones((ious.shape[1],))
            for row, col in zip(rows[::-1], cols[::-1]):
                if ious[row, col] >= iou_thresh:
                    if bbox_pred[row, 1] < 0: # if unassigned pred bbox
                        if gt_assigned[col] < 0: # if unassigned GT bbox
                            if bbox_pred[row, 2] == bbox_gt[col, 2]: # if belongs to the same class
                                bbox_pred[row, 1] = bbox_gt[col, 1]
                                gt_assigned[col] = 1

        # discard predicted boxes in crowd region
        if bbox_crowd.size > 0:
            # calculate iom matrix
            ioms = vectorized_iom(bbox_pred[:, 4:8], bbox_crowd[:, 4:8])
            max_ioms = np.amax(ioms, axis=1)
            retain_ids = []
            for i in range(bbox_pred.shape[0]):
                if bbox_pred[i, 1] < 0 and max_ioms[i] >= iom_thresh:
                    pass
                else:
                    retain_ids.append(i)
            # remove predicted detections from crowd regions
            bbox_pred = bbox_pred[retain_ids, :]

        # discard predicted boxes that are distractors
        if bbox_distract.size > 0:
            # calculate iou matrix
            ious = vectorized_iou(bbox_pred[:, 4:8], bbox_distract[:, 4:8])
            max_ious = np.amax(ious, axis=1)
            retain_ids = []
            for i in range(bbox_pred.shape[0]):
                if bbox_pred[i, 1] < 0 and max_ious[i] >= iou_thresh:
                    pass
                else:
                    retain_ids.append(i)
            # remove predicted detections that are distractors
            bbox_pred = bbox_pred[retain_ids, :]

        return bbox_pred, bbox_gt

    def __getitem__(self, index):
        """Generates one sample of data"""
        # get information for loading tracking sample
        input_info = self.chunks[index]

        # time reversal with probability 0.5
        random_transforms_tr = self.random_transforms and self._decision(0.5)
        # horizontal flip with probability 0.5
        random_transforms_hf = self.random_transforms and self._decision(0.5)

        # intiliaze empty arrays to store predicted and GT bboxes
        bbox_pred = np.zeros((0, 16), dtype=np.float32)
        bbox_gt = np.zeros((0, 16), dtype=np.float32)
        vis_feats = []
        tot_loss = self._convert_to_tensor(torch.tensor(0.0))
        if self.split == 'train' and 'vis' in self.feats:
            self.optimizer.zero_grad()

        for fr in input_info[1]:
            # load image
            im = PIL.Image.open(os.path.join(self.im_path, input_info[0], '%.4d.jpg' % (fr,)))
            # apply horizontal flip to image
            if random_transforms_hf:
                im = im.transpose(PIL.Image.FLIP_LEFT_RIGHT)
            
            # load GT annotations
            # [fr, trk_id, cat_id, alpha, x1, y1, x2, y2, h, w, l, x, y, z, rotation_y, score]
            annotations, bbox_gt_fr = self.load_bdd100k_labels(input_info[0], fr, (im.size[1], im.size[0]), random_transforms_hf)

            # load detections
            # [fr, -1, cat_id, -10, x1, y1, x2, y2, -1, -1, -1, -1000, -1000, -1000, -10, score]
            bbox_pred_fr = self.load_detections(input_info[0], fr, (im.size[1], im.size[0]), random_transforms_hf)

            # apply time reversal transform
            if random_transforms_tr:
                bbox_gt_fr[:, 0] = input_info[1][-1] - bbox_gt_fr[:, 0] + input_info[1][0]
                bbox_pred_fr[:, 0] = input_info[1][-1] - bbox_pred_fr[:, 0] + input_info[1][0]

            # assign GT track ids to each predicted bbox
            bbox_pred_fr, bbox_gt_fr = self.get_track_ids(bbox_pred_fr, bbox_gt_fr)

            # random dropout of bboxes
            if self.random_transforms:
                ret_idx = [not self._decision(self.dropout_ratio) for _ in range(bbox_pred_fr.shape[0])]
                bbox_pred_fr = bbox_pred_fr[ret_idx, :]

            # get visual embeddings from bbox centers
            if 'vis' in self.feats:
                # run forward pass through detector
                outputs = self.get_embed_net_outputs(im)
                vis_feats.append(self.get_vis_feats(outputs, 
                    bbox_pred_fr[:, 4:8], (im.size[1], im.size[0])))

            # append to existing bboxes in the sequence
            bbox_pred = np.concatenate((bbox_pred, bbox_pred_fr), axis=0)
            bbox_gt = np.concatenate((bbox_gt, bbox_gt_fr), axis=0)

        # features for tracker
        # one-hot category IDs
        features = self._convert_to_tensor(np.eye(len(self.class_dict), dtype=np.float32)[bbox_pred[:, 2].astype('int64') - 1])
        # 2d features [score, xc, yc, w, h]
        if '2d' in self.feats:
            two_d_feats = self._convert_to_tensor(np.stack((bbox_pred[:, 15], (bbox_pred[:, 4] + bbox_pred[:, 6])/2.0, 
                (bbox_pred[:, 5] + bbox_pred[:, 7])/2.0, bbox_pred[:, 6] - bbox_pred[:, 4], bbox_pred[:, 7] - bbox_pred[:, 5]), axis=1))
            features = torch.cat((features, two_d_feats), 1)
        # temporal feats
        if 'temp' in self.feats:
            temp_feats = self._convert_to_tensor(self.get_temp_feats(bbox_pred[:, 0:1]))
            features = torch.cat((features, temp_feats), 1)
        # visual features
        if 'vis' in self.feats:
            vis_feats = torch.cat(vis_feats, 0)
            # calculate embedding loss
            if self.split == 'train':
                tot_loss += self.embed_loss(vis_feats, bbox_pred[:, :2].astype('int64'))
            features = torch.cat((features, F.softmax(vis_feats, dim=1)), 1)

        if features.size()[0] != 0:
            features = (features - self.mean) / self.std # normalize/standardize features
                
        return features.detach(), bbox_pred, bbox_gt, tot_loss
