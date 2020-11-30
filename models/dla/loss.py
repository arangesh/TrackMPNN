from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from .decode import ddd_decode
from .model import _sigmoid, _transpose_and_gather_feat
from .utils.debugger import Debugger
from .utils.post_process import ddd_post_process


def _slow_neg_loss(pred, gt):
    '''focal loss from CornerNet'''
    pos_inds = gt.eq(1)
    neg_inds = gt.lt(1)

    neg_weights = torch.pow(1 - gt[neg_inds], 4)

    loss = 0
    pos_pred = pred[pos_inds]
    neg_pred = pred[neg_inds]

    pos_loss = torch.log(pos_pred) * torch.pow(1 - pos_pred, 2)
    neg_loss = torch.log(1 - neg_pred) * torch.pow(neg_pred, 2) * neg_weights

    num_pos  = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if pos_pred.nelement() == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos
    return loss

def _neg_loss(pred, gt):
    ''' Modified focal loss. Exactly the same as CornerNet.
      Runs faster and costs a little bit more memory
    Arguments:
      pred (batch x c x h x w)
      gt_regr (batch x c x h x w)
    '''
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()

    neg_weights = torch.pow(1 - gt, 4)

    loss = 0

    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

    num_pos  = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos
    return loss

def _not_faster_neg_loss(pred, gt):
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()    
    num_pos  = pos_inds.float().sum()
    neg_weights = torch.pow(1 - gt, 4)

    loss = 0
    trans_pred = pred * neg_inds + (1 - pred) * pos_inds
    weight = neg_weights * neg_inds + pos_inds
    all_loss = torch.log(1 - trans_pred) * torch.pow(trans_pred, 2) * weight
    all_loss = all_loss.sum()

    if num_pos > 0:
        all_loss /= num_pos
    loss -=  all_loss
    return loss

def _slow_reg_loss(regr, gt_regr, mask):
    num  = mask.float().sum()
    mask = mask.unsqueeze(2).expand_as(gt_regr)

    regr    = regr[mask]
    gt_regr = gt_regr[mask]
    
    regr_loss = nn.functional.smooth_l1_loss(regr, gt_regr, size_average=False)
    regr_loss = regr_loss / (num + 1e-4)
    return regr_loss

def _reg_loss(regr, gt_regr, mask):
    ''' L1 regression loss
    Arguments:
      regr (batch x max_objects x dim)
      gt_regr (batch x max_objects x dim)
      mask (batch x max_objects)
    '''
    num = mask.float().sum()
    mask = mask.unsqueeze(2).expand_as(gt_regr).float()

    regr = regr * mask
    gt_regr = gt_regr * mask

    regr_loss = nn.functional.smooth_l1_loss(regr, gt_regr, size_average=False)
    regr_loss = regr_loss / (num + 1e-4)
    return regr_loss

class FocalLoss(nn.Module):
    '''nn.Module warpper for focal loss'''
    def __init__(self):
        super(FocalLoss, self).__init__()
        self.neg_loss = _neg_loss

    def forward(self, out, target):
        return self.neg_loss(out, target)

class RegLoss(nn.Module):
    '''Regression loss for an output tensor
    Arguments:
      output (batch x dim x h x w)
      mask (batch x max_objects)
      ind (batch x max_objects)
      target (batch x max_objects x dim)
    '''
    def __init__(self):
        super(RegLoss, self).__init__()
  
    def forward(self, output, mask, ind, target):
        pred = _transpose_and_gather_feat(output, ind)
        loss = _reg_loss(pred, target, mask)
        return loss

class RegL1Loss(nn.Module):
    def __init__(self):
        super(RegL1Loss, self).__init__()
  
    def forward(self, output, mask, ind, target):
        pred = _transpose_and_gather_feat(output, ind)
        mask = mask.unsqueeze(2).expand_as(pred).float()
        # loss = F.l1_loss(pred * mask, target * mask, reduction='elementwise_mean')
        loss = F.l1_loss(pred * mask, target * mask, size_average=False)
        loss = loss / (mask.sum() + 1e-4)
        return loss

class NormRegL1Loss(nn.Module):
    def __init__(self):
        super(NormRegL1Loss, self).__init__()
  
    def forward(self, output, mask, ind, target):
        pred = _transpose_and_gather_feat(output, ind)
        mask = mask.unsqueeze(2).expand_as(pred).float()
        # loss = F.l1_loss(pred * mask, target * mask, reduction='elementwise_mean')
        pred = pred / (target + 1e-4)
        target = target * 0 + 1
        loss = F.l1_loss(pred * mask, target * mask, size_average=False)
        loss = loss / (mask.sum() + 1e-4)
        return loss

class RegWeightedL1Loss(nn.Module):
    def __init__(self):
        super(RegWeightedL1Loss, self).__init__()
  
    def forward(self, output, mask, ind, target):
        pred = _transpose_and_gather_feat(output, ind)
        mask = mask.float()
        # loss = F.l1_loss(pred * mask, target * mask, reduction='elementwise_mean')
        loss = F.l1_loss(pred * mask, target * mask, size_average=False)
        loss = loss / (mask.sum() + 1e-4)
        return loss

class L1Loss(nn.Module):
    def __init__(self):
        super(L1Loss, self).__init__()
  
    def forward(self, output, mask, ind, target):
        pred = _transpose_and_gather_feat(output, ind)
        mask = mask.unsqueeze(2).expand_as(pred).float()
        loss = F.l1_loss(pred * mask, target * mask, reduction='elementwise_mean')
        return loss

class BinRotLoss(nn.Module):
    def __init__(self):
        super(BinRotLoss, self).__init__()
  
    def forward(self, output, mask, ind, rotbin, rotres):
        pred = _transpose_and_gather_feat(output, ind)
        loss = compute_rot_loss(pred, rotbin, rotres, mask)
        return loss

def compute_res_loss(output, target):
    return F.smooth_l1_loss(output, target, reduction='elementwise_mean')

# TODO: weight
def compute_bin_loss(output, target, mask):
    mask = mask.expand_as(output)
    output = output * mask.float()
    return F.cross_entropy(output, target, reduction='elementwise_mean')

def compute_rot_loss(output, target_bin, target_res, mask):
    # output: (B, 128, 8) [bin1_cls[0], bin1_cls[1], bin1_sin, bin1_cos, 
    #                 bin2_cls[0], bin2_cls[1], bin2_sin, bin2_cos]
    # target_bin: (B, 128, 2) [bin1_cls, bin2_cls]
    # target_res: (B, 128, 2) [bin1_res, bin2_res]
    # mask: (B, 128, 1)
    # import pdb; pdb.set_trace()
    output = output.view(-1, 8)
    target_bin = target_bin.view(-1, 2)
    target_res = target_res.view(-1, 2)
    mask = mask.view(-1, 1)
    loss_bin1 = compute_bin_loss(output[:, 0:2], target_bin[:, 0], mask)
    loss_bin2 = compute_bin_loss(output[:, 4:6], target_bin[:, 1], mask)
    loss_res = torch.zeros_like(loss_bin1)
    if target_bin[:, 0].nonzero().shape[0] > 0:
        idx1 = target_bin[:, 0].nonzero()[:, 0]
        valid_output1 = torch.index_select(output, 0, idx1.long())
        valid_target_res1 = torch.index_select(target_res, 0, idx1.long())
        loss_sin1 = compute_res_loss(
          valid_output1[:, 2], torch.sin(valid_target_res1[:, 0]))
        loss_cos1 = compute_res_loss(
          valid_output1[:, 3], torch.cos(valid_target_res1[:, 0]))
        loss_res += loss_sin1 + loss_cos1
    if target_bin[:, 1].nonzero().shape[0] > 0:
        idx2 = target_bin[:, 1].nonzero()[:, 0]
        valid_output2 = torch.index_select(output, 0, idx2.long())
        valid_target_res2 = torch.index_select(target_res, 0, idx2.long())
        loss_sin2 = compute_res_loss(
          valid_output2[:, 6], torch.sin(valid_target_res2[:, 1]))
        loss_cos2 = compute_res_loss(
          valid_output2[:, 7], torch.cos(valid_target_res2[:, 1]))
        loss_res += loss_sin2 + loss_cos2
    return loss_bin1 + loss_bin2 + loss_res

class DddLoss(torch.nn.Module):
    def __init__(self, opt):
        super(DddLoss, self).__init__()
        self.crit = torch.nn.MSELoss() if opt.mse_loss else FocalLoss()
        self.crit_reg = L1Loss()
        self.crit_rot = BinRotLoss()
        self.opt = opt

    def forward(self, outputs, batch):
        opt = self.opt

        hm_loss, dep_loss, rot_loss, dim_loss = 0, 0, 0, 0
        wh_loss, off_loss = 0, 0
        outputs['hm'] = _sigmoid(outputs['hm'])
        outputs['dep'] = 1. / (outputs['dep'].sigmoid() + 1e-6) - 1.
      
        hm_loss += self.crit(outputs['hm'], batch['hm'])
        if opt.dep_weight > 0:
            dep_loss += self.crit_reg(outputs['dep'], batch['reg_mask'], 
                batch['ind'], batch['dep'])
        if opt.dim_weight > 0:
            dim_loss += self.crit_reg(outputs['dim'], batch['reg_mask'], 
                batch['ind'], batch['dim'])
        if opt.rot_weight > 0:
            rot_loss += self.crit_rot(outputs['rot'], batch['rot_mask'], 
                batch['ind'], batch['rotbin'], batch['rotres'])
        if opt.reg_bbox and opt.wh_weight > 0:
            wh_loss += self.crit_reg(outputs['wh'], batch['rot_mask'], 
                batch['ind'], batch['wh'])
        if opt.reg_offset and opt.off_weight > 0:
            off_loss += self.crit_reg(outputs['reg'], batch['rot_mask'], 
                batch['ind'], batch['reg'])
        loss = opt.hm_weight * hm_loss + opt.dep_weight * dep_loss + \
               opt.dim_weight * dim_loss + opt.rot_weight * rot_loss + \
               opt.wh_weight * wh_loss + opt.off_weight * off_loss

        loss_stats = {'loss': loss, 'hm_loss': hm_loss, 'dep_loss': dep_loss, 
                      'dim_loss': dim_loss, 'rot_loss': rot_loss, 
                      'wh_loss': wh_loss, 'off_loss': off_loss}
        return loss, loss_stats

class ModelWithLoss(torch.nn.Module):
    def __init__(self, model, loss):
        super(ModelWithLoss, self).__init__()
        self.model = model
        self.loss = loss

    def forward(self, batch):
        outputs = self.model(batch['input'])
        loss, loss_stats = self.loss(outputs, batch)
        return outputs[-1], loss, loss_stats

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        if self.count > 0:
          self.avg = self.sum / self.count

class DddTrainer(object):
    def __init__(self, opt, model, optimizer=None):
        self.opt = opt
        self.optimizer = optimizer
        self.loss_stats, self.loss = self._get_losses(opt)
        self.model_with_loss = ModelWithLoss(model, self.loss)

    def _get_losses(self, opt):
        loss_states = ['loss', 'hm_loss', 'dep_loss', 'dim_loss', 'rot_loss', 
                       'wh_loss', 'off_loss']
        loss = DddLoss(opt)
        return loss_states, loss

    def debug(self, batch, output, iter_id):
        opt = self.opt
        wh = output['wh'] if opt.reg_bbox else None
        reg = output['reg'] if opt.reg_offset else None
        dets = ddd_decode(output['hm'], output['rot'], output['dep'],
                          output['dim'], wh=wh, reg=reg, K=opt.K)

        # x, y, score, r1-r8, depth, dim1-dim3, cls
        dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
        calib = batch['meta']['calib'].detach().numpy()
        # x, y, score, rot, depth, dim1, dim2, dim3
        # if opt.dataset == 'gta':
        #   dets[:, 12:15] /= 3
        dets_pred = ddd_post_process(
            dets.copy(), batch['meta']['c'].detach().numpy(), 
            batch['meta']['s'].detach().numpy(), calib, opt)
        dets_gt = ddd_post_process(
            batch['meta']['gt_det'].detach().numpy().copy(),
            batch['meta']['c'].detach().numpy(), 
            batch['meta']['s'].detach().numpy(), calib, opt)
        #for i in range(input.size(0)):
        for i in range(1):
            debugger = Debugger(dataset=opt.dataset, ipynb=(opt.debug==3), 
                theme=opt.debugger_theme)
            img = batch['input'][i].detach().cpu().numpy().transpose(1, 2, 0)
            img = ((img * self.opt.std + self.opt.mean) * 255.).astype(np.uint8)
            pred = debugger.gen_colormap(output['hm'][i].detach().cpu().numpy())
            gt = debugger.gen_colormap(batch['hm'][i].detach().cpu().numpy())
            debugger.add_blend_img(img, pred, 'hm_pred')
            debugger.add_blend_img(img, gt, 'hm_gt')
            # decode
            debugger.add_ct_detection(
                img, dets[i], show_box=opt.reg_bbox, center_thresh=opt.center_thresh, 
                img_id='det_pred')
            debugger.add_ct_detection(
                img, batch['meta']['gt_det'][i].cpu().numpy().copy(), 
                show_box=opt.reg_bbox, img_id='det_gt')
            debugger.add_3d_detection(
                batch['meta']['image_path'][i], dets_pred[i], calib[i],
                center_thresh=opt.center_thresh, img_id='add_pred')
            debugger.add_3d_detection(
                batch['meta']['image_path'][i], dets_gt[i], calib[i],
                center_thresh=opt.center_thresh, img_id='add_gt')
            # debugger.add_bird_view(
                #   dets_pred[i], center_thresh=opt.center_thresh, img_id='bird_pred')
            # debugger.add_bird_view(dets_gt[i], img_id='bird_gt')
            debugger.add_bird_views(
                dets_pred[i], dets_gt[i], 
                center_thresh=opt.center_thresh, img_id='bird_pred_gt')
        
            # debugger.add_blend_img(img, pred, 'out', white=True)
            debugger.compose_vis_add(
                batch['meta']['image_path'][i], dets_pred[i], calib[i],
                opt.center_thresh, pred, 'bird_pred_gt', img_id='out')
            # debugger.add_img(img, img_id='out')
            if opt.debug ==4:
                debugger.save_all_imgs(opt.debug_dir, prefix='{}'.format(iter_id))
            else:
                debugger.show_all_imgs(pause=True)

    def save_result(self, output, batch, results):
        opt = self.opt
        wh = output['wh'] if opt.reg_bbox else None
        reg = output['reg'] if opt.reg_offset else None
        dets = ddd_decode(output['hm'], output['rot'], output['dep'], 
            output['dim'], wh=wh, reg=reg, K=opt.K)

        # x, y, score, r1-r8, depth, dim1-dim3, cls
        dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
        calib = batch['meta']['calib'].detach().numpy()
        # x, y, score, rot, depth, dim1, dim2, dim3
        dets_pred = ddd_post_process(
            dets.copy(), batch['meta']['c'].detach().numpy(), 
            batch['meta']['s'].detach().numpy(), calib, opt)
        img_id = batch['meta']['img_id'].detach().numpy()[0]
        results[img_id] = dets_pred[0]
        for j in range(1, opt.num_classes + 1):
            keep_inds = (results[img_id][j][:, -1] > opt.center_thresh)
            results[img_id][j] = results[img_id][j][keep_inds]

    def set_device(self, gpus, chunk_sizes, device):
        self.model_with_loss = self.model_with_loss.to(device)
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device=device, non_blocking=True)

    def run_epoch(self, phase, epoch, data_loader):
        model_with_loss = self.model_with_loss
        if phase == 'train':
            model_with_loss.train()
        else:
            model_with_loss.eval()
            torch.cuda.empty_cache()

        opt = self.opt
        results = {}
        avg_loss_stats = {l: AverageMeter() for l in self.loss_stats}
        num_iters = len(data_loader) if opt.num_iters < 0 else opt.num_iters
        for iter_id, batch in enumerate(data_loader):
            if iter_id >= num_iters:
                break

            for k in batch:
                if k != 'meta':
                    batch[k] = batch[k].to(device=opt.device, non_blocking=True)    
            output, loss, loss_stats = model_with_loss(batch)
            loss = loss.mean()
            if phase == 'train':
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            for l in avg_loss_stats:
                avg_loss_stats[l].update(
                    loss_stats[l].mean().item(), batch['input'].size(0))
      
            if opt.debug > 0:
                self.debug(batch, output, iter_id)
      
            if opt.test:
                self.save_result(output, batch, results)
            del output, loss, loss_stats

        bar.finish()
        ret = {k: v.avg for k, v in avg_loss_stats.items()}
        return ret, results

    def val(self, epoch, data_loader):
        return self.run_epoch('val', epoch, data_loader)

    def train(self, epoch, data_loader):
        return self.run_epoch('train', epoch, data_loader)