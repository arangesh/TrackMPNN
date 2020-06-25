from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from types import SimpleNamespace
import time

import cv2
import numpy as np
import torch

from .model import create_model, load_model
from .decode import ddd_decode
from .model import flip_tensor
from .utils.image import get_affine_transform
from .utils.post_process import ddd_post_process
from .utils.debugger import Debugger
from .utils.ddd_utils import compute_box_3d, project_to_image, alpha2rot_y
from .utils.ddd_utils import draw_box_3d, unproject_2d_to_3d


class DddDetector(object):
    def __init__(self, snapshot, cuda, split, num_img_feats=4, dataset='kitti'):
        if dataset == 'kitti':
            self.opt = self.create_options_kitti(snapshot, cuda, split, num_img_feats)

        print('Creating model...')
        self.model = create_model(34, self.opt.heads, 256)
        self.model = load_model(self.model, self.opt.load_model)
        self.model = self.model.to(self.opt.device)
        if self.opt.split == 'train':
            self.model.train()
        else:
            self.model.eval()

        self.mean = np.array(self.opt.mean, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array(self.opt.std, dtype=np.float32).reshape(1, 1, 3)
        self.num_classes = self.opt.num_classes
        self.scales = self.opt.test_scales
        self.pause = True
        self.calib = np.array([[707.0493, 0, 604.0814, 45.75831],
                               [0, 707.0493, 180.5066, -0.3454157],
                               [0, 0, 1., 0.004981016]], dtype=np.float32)

    def create_options_kitti(self, snapshot, cuda, split, num_img_feats):
        opt_dict = dict()
        opt_dict['cuda'] = cuda
        opt_dict['split'] = split
        opt_dict['dataset'] = 'kitti'
        if cuda:
            opt_dict['device'] = torch.device('cuda')
        else:
            opt_dict['device'] = torch.device('cpu')
        opt_dict['reg_bbox'] = True
        opt_dict['reg_offset'] = True
        opt_dict['heads'] = {'hm': 3, 'wh':2, 'reg':2, 'dep': 1, 'rot': 8, 'dim': 3, 'trk': num_img_feats}
        opt_dict['load_model'] = snapshot
        opt_dict['test_scales'] = [1]
        opt_dict['num_classes'] = 3
        opt_dict['mean'] = [0.485, 0.456, 0.406]
        opt_dict['std'] = [0.229, 0.224, 0.225]
        opt_dict['input_h'] = 384
        opt_dict['input_w'] = 1280
        opt_dict['down_ratio'] = 4
        opt_dict['output_h'] = opt_dict['input_h'] // opt_dict['down_ratio']
        opt_dict['output_w'] = opt_dict['input_w'] // opt_dict['down_ratio']
        opt_dict['input_res'] = max(opt_dict['input_h'], opt_dict['input_w'])
        opt_dict['output_res'] = max(opt_dict['output_h'], opt_dict['output_w'])
        opt_dict['keep_res'] = False
        opt_dict['K'] = 100
        opt_dict['peak_thresh'] = 0.2
        opt_dict['vis_thresh'] = 0.3
        opt_dict['debug'] = 0
        opt_dict['debugger_theme'] = 'white'

        opt_dict['mse_loss'] = False
        opt_dict['hm_weight'] = 1.0
        opt_dict['dep_weight'] = 1.0
        opt_dict['dim_weight'] = 1.0
        opt_dict['rot_weight'] = 1.0
        opt_dict['wh_weight'] = 0.1
        opt_dict['off_weight'] = 1.0
        opt_dict['rect_mask'] = False

        return SimpleNamespace(**opt_dict)

    def pre_process(self, image, scale, calib=None):
        height, width = image.shape[0:2]

        inp_height, inp_width = self.opt.input_h, self.opt.input_w
        c = np.array([width / 2, height / 2], dtype=np.float32)
        if self.opt.keep_res:
            s = np.array([inp_width, inp_height], dtype=np.int32)
        else:
            s = np.array([width, height], dtype=np.int32)

        trans_input = get_affine_transform(c, s, 0, [inp_width, inp_height])
        resized_image = image #cv2.resize(image, (width, height))
        inp_image = cv2.warpAffine(resized_image, trans_input, (inp_width, inp_height), 
            flags=cv2.INTER_LINEAR)
        inp_image = (inp_image.astype(np.float32) / 255.)
        inp_image = (inp_image - self.mean) / self.std
        images = inp_image.transpose(2, 0, 1)[np.newaxis, ...]
        calib = np.array(calib, dtype=np.float32) if calib is not None else self.calib
        images = torch.from_numpy(images)
        meta = {'c': c, 's': s, 
                'out_height': inp_height // self.opt.down_ratio, 
                'out_width': inp_width // self.opt.down_ratio,
                'calib': calib}
        return images, meta
  
    def process(self, images, return_time=False):
        if self.opt.split == 'train':
            outputs = self.model(images)[-1]
            wh = outputs['wh'] if self.opt.reg_bbox else None
            reg = outputs['reg'] if self.opt.reg_offset else None
            forward_time = time.time()
            with torch.no_grad():
                dets = ddd_decode(outputs['hm'].sigmoid(), outputs['rot'], 
                                    1. / (outputs['dep'].sigmoid() + 1e-6) - 1., 
                                    outputs['dim'], wh=wh, reg=reg, K=self.opt.K)
        else:
            with torch.no_grad():
                outputs = self.model(images)[-1]
                outputs['hm'] = outputs['hm'].sigmoid_()
                outputs['dep'] = 1. / (outputs['dep'].sigmoid() + 1e-6) - 1.
                wh = outputs['wh'] if self.opt.reg_bbox else None
                reg = outputs['reg'] if self.opt.reg_offset else None
                forward_time = time.time()
                dets = ddd_decode(outputs['hm'], outputs['rot'], outputs['dep'], 
                                outputs['dim'], wh=wh, reg=reg, K=self.opt.K)
        if return_time:
            return outputs, dets, forward_time
        else:
            return outputs, dets

    def post_process(self, dets, meta, scale=1):
        dets = dets.detach().cpu().numpy()
        detections = ddd_post_process(
            dets.copy(), [meta['c']], [meta['s']], [meta['calib']], self.opt)
        self.this_calib = meta['calib']
        return detections[0]

    def merge_outputs(self, detections):
        results = detections[0]
        for j in range(1, self.num_classes + 1):
            if len(results[j] > 0):
                keep_inds = (results[j][:, -1] > self.opt.peak_thresh)
                results[j] = results[j][keep_inds]
        return results

    def debug(self, debugger, images, dets, outputs, scale=1):
        dets = dets.detach().cpu().numpy()
        img = images[0].detach().cpu().numpy().transpose(1, 2, 0)
        img = ((img * self.std + self.mean) * 255).astype(np.uint8)
        pred = debugger.gen_colormap(outputs['hm'][0].detach().cpu().numpy())
        debugger.add_blend_img(img, pred, 'pred_hm')
        debugger.add_ct_detection(
            img, dets[0], show_box=self.opt.reg_bbox, 
            center_thresh=self.opt.vis_thresh, img_id='det_pred')
  
    def show_results(self, debugger, image, results):
        debugger.add_3d_detection(
            image, results, self.this_calib,
            center_thresh=self.opt.vis_thresh, img_id='add_pred')
        debugger.add_bird_view(
            results, center_thresh=self.opt.vis_thresh, img_id='bird_pred')
        debugger.show_all_imgs(pause=self.pause)

    def run(self, image_or_path_or_tensor, meta=None):
        load_time, pre_time, net_time, dec_time, post_time = 0, 0, 0, 0, 0
        merge_time, tot_time = 0, 0
        debugger = Debugger(dataset=self.opt.dataset, ipynb=(self.opt.debug==3),
                            theme=self.opt.debugger_theme)
        start_time = time.time()
        pre_processed = False
        if isinstance(image_or_path_or_tensor, np.ndarray):
            image = image_or_path_or_tensor
        elif type(image_or_path_or_tensor) == type (''): 
            image = cv2.imread(image_or_path_or_tensor)
        else:
            image = image_or_path_or_tensor['image'][0].numpy()
            pre_processed_images = image_or_path_or_tensor
            pre_processed = True

        loaded_time = time.time()
        load_time += (loaded_time - start_time)

        detections = []
        for scale in self.scales:
            scale_start_time = time.time()
            if not pre_processed:
                images, meta = self.pre_process(image, scale, meta)
            else:
            # import pdb; pdb.set_trace()
                images = pre_processed_images['images'][scale][0]
                meta = pre_processed_images['meta'][scale]
                meta = {k: v.numpy()[0] for k, v in meta.items()}
            images = images.to(self.opt.device)
            torch.cuda.synchronize()
            pre_process_time = time.time()
            pre_time += pre_process_time - scale_start_time

            outputs, dets, forward_time = self.process(images, return_time=True)

            torch.cuda.synchronize()
            net_time += forward_time - pre_process_time
            decode_time = time.time()
            dec_time += decode_time - forward_time
          
            if self.opt.debug >= 2:
                self.debug(debugger, images, dets, outputs, scale)

            dets = self.post_process(dets, meta, scale)
            torch.cuda.synchronize()
            post_process_time = time.time()
            post_time += post_process_time - decode_time

            detections.append(dets)

        results = self.merge_outputs(detections)
        torch.cuda.synchronize()
        end_time = time.time()
        merge_time += end_time - post_process_time
        tot_time += end_time - start_time

        if self.opt.debug >= 1:
            self.show_results(debugger, image, results)

        return {'results': results, 'outputs':outputs, 'tot': tot_time, 'load': load_time,
                'pre': pre_time, 'net': net_time, 'dec': dec_time,
                'post': post_time, 'merge': merge_time}
