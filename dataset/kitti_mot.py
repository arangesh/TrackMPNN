import os
import json
import random
import numpy as np
from scipy.optimize import linear_sum_assignment
import cv2

import torch
from torch.utils import data
import torch.optim as optim
import torchvision.transforms as transforms

from utils.misc import vectorized_iou
from models.dla.ddd import DddDetector
from models.dla.loss import DddLoss
from models.dla.utils.image import gaussian_radius, draw_umich_gaussian, draw_msra_gaussian
from models.dla.utils.image import get_affine_transform, affine_transform
from models.loss import EmbeddingLoss


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
    def __init__(self, dataset_root_path=None, split='train', cat='Car', timesteps=5, num_img_feats=4, snapshot=None, random_transforms=False, cuda=True):
        """Initialization"""

        if dataset_root_path is None:
            raise FileNotFoundError("Dataset path needs to be valid")
        print('Preparing ' + split + ' dataset...')

        self.split = split
        self.class_dict = {'Pedestrian': 1, 'Car': 2, 'Cyclist': 3}
        self.cat = cat
        self.timesteps = timesteps
        self.num_img_feats = num_img_feats # number of image based features to be used for tracking
        self.snapshot = snapshot
        self.random_transforms = random_transforms
        self.cuda = cuda
        self.dropout_ratio = 0.2 # probability of a detection being dropped

        if self.split == 'test':
            self.im_path = os.path.join(dataset_root_path, 'testing', 'image_02')
            self.label_path = None
        else:
            self.im_path = os.path.join(dataset_root_path, 'training', 'image_02')
            self.label_path = os.path.join(dataset_root_path, 'training', 'label_02')

        # initialize detector with necessary heads and pretrained weights
        if self.split == 'train':
            self.detector = DddDetector(self.snapshot, self.cuda, 'train', num_img_feats=num_img_feats, dataset='kitti')
            self.det_loss = DddLoss(self.detector.opt)
            self.embed_loss = EmbeddingLoss()
            # optimizer for detector
            self.optimizer = optim.Adam(self.detector.model.parameters(), lr=1.25e-5)
        elif self.split == 'val':
            # do not initialize a second detector for val (will use the same one as train)
            self.detector = None
        elif self.split == 'test':
            self.detector = DddDetector(self.snapshot, self.cuda, 'test', num_img_feats=num_img_feats, dataset='kitti')

        # get tracking batch information 
        self.chunks = self.get_tracking_chunks()
        # load mean values for each feature
        mean = [0.90] + [621, 187.5, 621, 187.5] # 2d features
        mean = mean + [1.53, 1.63, 3.88] + [0.0, 0.8, 35.70] + [0.0] # 3d features
        mean = mean + [0.0 for _ in range(self.num_img_feats)] # image features
        self.mean = self._convert_to_tensor(np.array([mean], dtype='float32'))
        # load std values for each feature
        std = [0.20] + [1242.0, 375.0, 1242.0, 375.0] # 2d features
        std = std + [0.14, 0.10, 0.43] + [123.95, 1.0, 395.67] + [np.pi] # 3d features
        std = std + [1.0 for _ in range(self.num_img_feats)] # image features
        self.std = self._convert_to_tensor(np.array([std], dtype='float32'))

        print('Finished preparing ' + self.split + ' dataset!')

    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.chunks)

    def _convert_to_tensor(self, var):
        """
        Handles conversion of ndarrays/datatypes to cuda/normal tensors
        """
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

    def _convert_alpha(self, alpha):
        return alpha

    def _alpha_to_8(self, alpha):
        # return [alpha, 0, 0, 0, 0, 0, 0, 0]
        ret = [0, 0, 0, 1, 0, 0, 0, 1]
        if alpha < np.pi / 6. or alpha > 5 * np.pi / 6.:
            r = alpha - (-0.5 * np.pi)
            ret[1] = 1
            ret[2], ret[3] = np.sin(r), np.cos(r)
        if alpha > -np.pi / 6. or alpha < -5 * np.pi / 6.:
            r = alpha - (0.5 * np.pi)
            ret[5] = 1
            ret[6], ret[7] = np.sin(r), np.cos(r)
        return ret

    def get_tracking_chunks(self):
        seqs = sorted(os.listdir(self.im_path))
        # seqs 13, 16 and 17 have very few or no cars at all
        if self.split == 'train':
            seqs = seqs[:-1]
            print(seqs)
        elif self.split == 'val':
            seqs = seqs[-1:]
            print(seqs)
        else:
            pass

        num_frames = [len(os.listdir(os.path.join(self.im_path, x))) for x in seqs]

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

    def load_kitti_labels(self, seq, fr, im_shape, random_transforms_hf):
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

        cats = ['Pedestrian', 'Car', 'Cyclist', 'Van', 'Truck',  'Person',
        'Tram', 'Misc', 'DontCare']
        cat_ids = {cat: i + 1 for i, cat in enumerate(cats)}
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

            if self.cat != tmp[2]:
                continue
            # [fr, trk_id, cat_id, alpha, x1, y1, x2, y2, h, w, l, x, y, z, rotation_y, score]
            b = [ann['frame'], ann['track_id'], ann['category_id']] \
                + [ann['alpha']] + ann['bbox'] + ann['dim'] \
                + ann['location'] + [ann['rotation_y']] + [1]
            bbox_gt = np.concatenate((bbox_gt, np.array([b], dtype=np.float32)), axis=0)
        return annotations, bbox_gt

    def get_trk_feats(self, feat_maps, bboxes, im_shape):
        """
        Extract image features from bbox center location
        """
        feats = []
        for bbox in bboxes:
            # get bbox center
            c_x = (bbox[0] + bbox[2]) / 2.0
            c_y = (bbox[1] + bbox[3]) / 2.0
            # account for image resizing
            c_x = (c_x * self.detector.opt.input_w) / im_shape[1]
            c_y = (c_y * self.detector.opt.input_h) / im_shape[0]
            # divide the center by downsampling ratio of the model
            c_x = int(c_x / self.detector.opt.down_ratio)
            c_y = int(c_y / self.detector.opt.down_ratio)

            # extract features from center
            feats.append(feat_maps[:, :, c_y, c_x])
        if len(feats) == 0:
            return self._convert_to_tensor(np.zeros((0, self.num_img_feats), dtype='float32'))
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

    def create_detector_labels(self, anns, im_shape):
        cat_ids = {1:0, 2:1, 3:2, 4:-3, 5:-3, 6:-2, 7:-99, 8:-99, 9:-1}
        max_objs = 50
        num_classes = self.detector.opt.num_classes

        height, width = im_shape[0:2]
        c = np.array([width / 2, height / 2], dtype=np.float32)
        if self.detector.opt.keep_res:
            inp_height, inp_width = self.detector.opt.input_h, self.detector.opt.input_w
            s = np.array([inp_width, inp_height], dtype=np.int32)
        else:
            s = np.array([width, height], dtype=np.int32)
        out_height = self.detector.opt.output_h
        out_width = self.detector.opt.output_w
        trans_output = get_affine_transform(c, s, 0, [out_width, out_height])

        hm = np.zeros(
            (num_classes, self.detector.opt.output_h, self.detector.opt.output_w), dtype=np.float32)
        wh = np.zeros((max_objs, 2), dtype=np.float32)
        reg = np.zeros((max_objs, 2), dtype=np.float32)
        dep = np.zeros((max_objs, 1), dtype=np.float32)
        rotbin = np.zeros((max_objs, 2), dtype=np.int64)
        rotres = np.zeros((max_objs, 2), dtype=np.float32)
        dim = np.zeros((max_objs, 3), dtype=np.float32)
        ind = np.zeros((max_objs), dtype=np.int64)
        reg_mask = np.zeros((max_objs), dtype=np.uint8)
        rot_mask = np.zeros((max_objs), dtype=np.uint8)

        num_objs = min(len(anns), max_objs)
        draw_gaussian = draw_msra_gaussian if self.detector.opt.mse_loss else \
                        draw_umich_gaussian
        gt_det = []
        for k in range(num_objs):
            ann = anns[k]
            bbox = np.array(ann['bbox'], dtype=np.float32)
            cls_id = int(cat_ids[ann['category_id']])
            if cls_id <= -99:
                continue
            bbox[:2] = affine_transform(bbox[:2], trans_output)
            bbox[2:] = affine_transform(bbox[2:], trans_output)
            bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, self.detector.opt.output_w - 1)
            bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, self.detector.opt.output_h - 1)
            h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
            if h > 0 and w > 0:
                radius = gaussian_radius((h, w))
                radius = max(0, int(radius))
                ct = np.array(
                    [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
                ct_int = ct.astype(np.int32)
                if cls_id < 0:
                    ignore_id = [_ for _ in range(num_classes)] \
                                if cls_id == - 1 else  [- cls_id - 2]
                    if self.detector.opt.rect_mask:
                        hm[ignore_id, int(bbox[1]): int(bbox[3]) + 1, 
                           int(bbox[0]): int(bbox[2]) + 1] = 0.9999
                    else:
                        for cc in ignore_id:
                            draw_gaussian(hm[cc], ct, radius)
                        hm[ignore_id, ct_int[1], ct_int[0]] = 0.9999
                    continue
                draw_gaussian(hm[cls_id], ct, radius)

                wh[k] = 1. * w, 1. * h
                gt_det.append([ct[0], ct[1], 1] + \
                               self._alpha_to_8(self._convert_alpha(ann['alpha'])) + \
                               [ann['depth']] + (np.array(ann['dim']) / 1).tolist() + [cls_id])
                if self.detector.opt.reg_bbox:
                    gt_det[-1] = gt_det[-1][:-1] + [w, h] + [gt_det[-1][-1]]
                if 1:
                    alpha = self._convert_alpha(ann['alpha'])
                    # print('img_id cls_id alpha rot_y', img_path, cls_id, alpha, ann['rotation_y'])
                    if alpha < np.pi / 6. or alpha > 5 * np.pi / 6.:
                        rotbin[k, 0] = 1
                        rotres[k, 0] = alpha - (-0.5 * np.pi)    
                    if alpha > -np.pi / 6. or alpha < -5 * np.pi / 6.:
                        rotbin[k, 1] = 1
                        rotres[k, 1] = alpha - (0.5 * np.pi)
                    dep[k] = ann['depth']
                    dim[k] = ann['dim']
                    # print('        cat dim', cls_id, dim[k])
                    ind[k] = ct_int[1] * self.detector.opt.output_w + ct_int[0]
                    reg[k] = ct - ct_int
                    reg_mask[k] = 1
                    rot_mask[k] = 1

        ret = {'hm': hm, 'dep': dep, 'dim': dim, 'ind': ind, 
               'rotbin': rotbin, 'rotres': rotres, 'reg_mask': reg_mask,
               'rot_mask': rot_mask}
        if self.detector.opt.reg_bbox:
            ret.update({'wh': wh})
        if self.detector.opt.reg_offset:
            ret.update({'reg': reg})
        # convert to tensor and add batch dimension
        for k, v in ret.items():
            ret[k] = self._convert_to_tensor(v).unsqueeze(0)
        return ret

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
        im_feats = []
        tot_loss = self._convert_to_tensor(torch.tensor(0.0))
        if self.split == 'train':
            self.optimizer.zero_grad()

        for fr in range(input_info[1], input_info[2], 1):
            # load image
            im = cv2.imread(os.path.join(self.im_path, input_info[0], '%.6d.png' % (fr,)))
            
            # load GT annotations
            # [fr, trk_id, cat_id, alpha, x1, y1, x2, y2, h, w, l, x, y, z, rotation_y, score]
            annotations, bbox_gt_fr = self.load_kitti_labels(input_info[0], fr, im.shape, random_transforms_hf)

            # apply horizontal flip transform
            if random_transforms_hf:
                im = cv2.flip(im, 1)

            # run forward pass through detector
            detector_ops = self.detector.run(im)
            # apply detector losses
            if self.split == 'train':
                det_labels = self.create_detector_labels(annotations, im.shape)
                loss, loss_stats = self.det_loss(detector_ops['outputs'], det_labels)
                tot_loss += loss.mean() / self.timesteps

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
                ret_idx = [not self._decision(self.dropout_ratio) for _ in range(bbox_pred_fr.shape[0])]
                bbox_pred_fr = bbox_pred_fr[ret_idx, :]

            # append to existing bboxes in the sequence
            bbox_pred = np.concatenate((bbox_pred, bbox_pred_fr), axis=0)
            bbox_gt = np.concatenate((bbox_gt, bbox_gt_fr), axis=0)
            im_feats.append(self.get_trk_feats(detector_ops['outputs']['trk'], 
                bbox_pred_fr[:, 4:8], im.shape))

        # features for tracker
        # [score, x1, y1, x2, y2]
        two_d_feats = self._convert_to_tensor(bbox_pred[:, [15, 4, 5, 6, 7]])
        # [h, w, l, X, Y, Z, rotation_y]
        three_d_feats = self._convert_to_tensor(bbox_pred[:, [8, 9, 10, 11, 12, 13, 14]])
        # image features
        im_feats = torch.cat(im_feats, 0)
        # (num_dets, 5 + 7 + num_img_feats)
        features = torch.cat((two_d_feats, three_d_feats, im_feats), 1)

        if features.size()[0] != 0:
            features = (features - self.mean) / self.std # normalize/standardize features
            if self.split == 'train':
                # add embedding loss
                tot_loss += self.embed_loss(features[:, -self.num_img_feats:], bbox_pred[:, :2].astype('int64'))
                tot_loss.backward()
                self.optimizer.step()
        return features.detach(), bbox_pred, bbox_gt, tot_loss.detach()
