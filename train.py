import os
import statistics
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

import torch
from torch.utils.data import DataLoader
import torch.optim as optim

from models.track_mpnn import TrackMPNN
from models.loss import create_targets, FocalLoss, CELoss
from models.dla.model import save_model
from dataset.kitti_mot import KittiMOTDataset
from utils.graph import initialize_graph, update_graph, prune_graph, decode_tracks
from utils.metrics import create_mot_accumulator, calc_mot_metrics, compute_map
from utils.training_options import args
from utils.gradients import plot_grad_flow


kwargs_train = {'batch_size': 1, 'shuffle': True}
train_loader = DataLoader(KittiMOTDataset(args.dataset_root_path, 'train', args.category, args.cur_win_size, args.ret_win_size, args.num_img_feats, 
                os.path.join('.', 'weights', 'model_last_kitti.pth'), args.random_transforms, args.cuda), **kwargs_train)
kwargs_val = {'batch_size': 1, 'shuffle': False}
val_loader = DataLoader(KittiMOTDataset(args.dataset_root_path, 'val', args.category, args.cur_win_size, args.ret_win_size, args.num_img_feats, 
                None, False, args.cuda), **kwargs_val)

# global var to store best MOTA across all epochs
best_mota = -float('Inf')
# create file handles
f_log = open(os.path.join(args.output_dir, "logs.txt"), "w")


# random seed function (https://docs.fast.ai/dev/test.html#getting-reproducible-results)
def random_seed(seed_value, use_cuda):
    torch.manual_seed(seed_value)
    if use_cuda:
        torch.backends.cudnn.deterministic = True  #needed


# training function
def train(model, epoch):
    epoch_loss_d, epoch_loss_c, epoch_loss_f, epoch_loss, epoch_f1 = list(), list(), list(), list(), list()
    model.train() # set TrackMPNN model to train mode
    train_loader.dataset.detector.opt.split = 'train' # set split to train
    for b_idx, (X_seq, bbox_pred, _, loss_d) in enumerate(train_loader):
        # if no detections in sequence
        if X_seq.size()[1] == 0:
            print('No detections available for sequence...')
            continue
        y_seq = bbox_pred[:, :, :2]

        # train the network
        optimizer_trk.zero_grad()

        # intialize graph and run first forward pass
        y_pred, feats, node_adj, edge_adj, labels, t_st, t_end = initialize_graph(X_seq, y_seq, mode='train', cuda=args.cuda)
        if y_pred is None:
            continue
        scores, logits, states = model(feats, None, node_adj, edge_adj)
        # compute the loss
        idx_edge = torch.nonzero((y_pred[:, 0] == -1))[:, 0]
        idx_node = torch.nonzero((y_pred[:, 0] != -1))[:, 0]
        # calculate targets for CE and BCE(Focal) loss
        targets = create_targets(labels, node_adj, idx_node)
        # calculate CE loss
        loss_c = ce_loss(logits, targets, node_adj, idx_node)
        if args.tp_classifier:
            loss_f = focal_loss_node(scores[idx_node, 0], targets[idx_node]) + focal_loss_edge(scores[idx_edge, 0], targets[idx_edge])
            scores = torch.cat((1 - scores, scores), dim=1)
            idx = torch.cat((idx_node, idx_edge))
        else:
            loss_f = focal_loss_edge(scores[idx_edge, 0], targets[idx_edge])
            scores = torch.cat((1 - scores, scores), dim=1)
            scores[idx_node, 0] = 0
            scores[idx_node, 1] = 1
            idx = idx_edge
        # compute the f1 score
        pred = scores.data.max(1)[1]  # get the index of the max log-probability
        epoch_f1.append(f1_score(targets[idx].detach().cpu().numpy(), pred[idx].detach().cpu().numpy()))

        # loop through all frames
        for t_cur in range(t_st, t_end):
            # update graph for next timestep and run forward pass
            y_pred, feats, node_adj, edge_adj, labels = update_graph(node_adj, labels, scores, y_pred, X_seq, y_seq, t_cur, 
                use_hungraian=args.hungarian, mode='train', cuda=args.cuda)
            if feats.size()[0] == 0:
                continue
            scores, logits, states = model(feats, states, node_adj, edge_adj)
            # compute the loss
            idx_edge = torch.nonzero((y_pred[:, 0] == -1))[:, 0]
            idx_node = torch.nonzero((y_pred[:, 0] != -1))[:, 0]
            # calculate targets for CE and BCE(Focal) loss
            targets = create_targets(labels, node_adj, idx_node)
            # calculate CE loss
            loss_c += ce_loss(logits, targets, node_adj, idx_node)
            if args.tp_classifier:
                loss_f += focal_loss_node(scores[idx_node, 0], targets[idx_node]) + focal_loss_edge(scores[idx_edge, 0], targets[idx_edge])
                scores = torch.cat((1 - scores, scores), dim=1)
                idx = torch.cat((idx_node, idx_edge))
            else:
                loss_f += focal_loss_edge(scores[idx_edge, 0], targets[idx_edge])
                scores = torch.cat((1 - scores, scores), dim=1)
                scores[idx_node, 0] = 0
                scores[idx_node, 1] = 1
                idx = idx_edge
            # compute the F1 score
            pred = scores.data.max(1)[1]  # get the index of the max log-probability
            epoch_f1.append(f1_score(targets[idx].detach().cpu().numpy(), pred[idx].detach().cpu().numpy()))

        epoch_loss_d.append(loss_d.item())
        epoch_loss_c.append(loss_c.item())
        epoch_loss_f.append(loss_f.item())
        loss = loss_d + loss_c + loss_f
        epoch_loss.append(loss.item())
        loss.backward()
        optimizer_trk.step()
        train_loader.dataset.optimizer.step()

        # save gradient flow image through detector and tracker model
        if (b_idx % 100 == 0) and args.plot_gradients:
            plot_grad_flow([list(param for param in train_loader.dataset.detector.model.named_parameters() if 'trk' in param[0]),
                model.named_parameters()], os.path.join(args.output_dir, 'gradients', 'epoch%.3d_iter%.6d.jpg' % (epoch, b_idx)))

        if b_idx % args.log_schedule == 0:
            print('Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.5f}'.format(
                epoch, (b_idx + 1), len(train_loader.dataset),
                100. * (b_idx + 1) / len(train_loader.dataset), loss.item()))
            f_log.write('Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.5f}\n'.format(
                epoch, (b_idx + 1), len(train_loader.dataset),
                100. * (b_idx + 1) / len(train_loader.dataset), loss.item()))

    # now that the epoch is completed calculate statistics and store logs
    avg_loss_d = statistics.mean(epoch_loss_d)
    avg_loss_c = statistics.mean(epoch_loss_c)
    avg_loss_f = statistics.mean(epoch_loss_f)
    avg_loss = statistics.mean(epoch_loss)
    avg_f1 = statistics.mean(epoch_f1)
    print("------------------------\nAverage detector loss for epoch = {:.2f}".format(avg_loss_d))
    f_log.write("------------------------\nAverage detector loss for epoch = {:.2f}\n".format(avg_loss_d))
    print("Average cross-entropy loss for epoch = {:.2f}".format(avg_loss_c))
    f_log.write("Average cross-entropy loss for epoch = {:.2f}\n".format(avg_loss_c))
    print("Average focal loss for epoch = {:.2f}".format(avg_loss_f))
    f_log.write("Average focal loss for epoch = {:.2f}\n".format(avg_loss_f))
    print("Average loss for epoch = {:.2f}".format(avg_loss))
    f_log.write("Average loss for epoch = {:.2f}\n".format(avg_loss))
    print("Average F1 score for epoch = {:.4f}\n------------------------".format(avg_f1))
    f_log.write("Average F1 score for epoch = {:.4f}\n".format(avg_f1))

    return model, avg_loss_d, avg_loss_c, avg_loss_f, avg_loss, avg_f1


# validation function
def val(model, epoch):
    global best_mota
    epoch_f1 = list()
    accs = []
    model.eval() # set TrackMPNN model to eval mode
    val_loader.dataset.detector = train_loader.dataset.detector # use trained detector for the val loader
    train_loader.dataset.detector = None # set trained detector to None to save memory
    val_loader.dataset.detector.opt.split = 'val' # set split to val

    bbox_pred_dict, bbox_gt_dict = {}, {} # initialize dictionaries for computing mAP
    for b_idx, (X_seq, bbox_pred, bbox_gt, _) in enumerate(val_loader):
        # if no detections in sequence
        if X_seq.size()[1] == 0:
            print('No detections available for sequence...')
            continue
        y_seq = bbox_pred[:, :, :2]

        # initaialize output array tracks to -1s
        y_out = y_seq.squeeze(0).detach().cpu().numpy().astype('int64')
        y_out[:, 1] = -1

        # intialize graph and run first forward pass
        y_pred, feats, node_adj, edge_adj, labels, t_st, t_end = initialize_graph(X_seq, y_seq, mode='test', cuda=args.cuda)
        if y_pred is None:
            continue
        # compute the classification scores
        scores, logits, states = model(feats, None, node_adj, edge_adj)
        scores = torch.cat((1-scores, scores), dim=1)

        idx_edge = torch.nonzero((y_pred[:, 0] == -1))[:, 0]
        idx_node = torch.nonzero((y_pred[:, 0] != -1))[:, 0]
        # calculate targets for computing metrics
        targets = create_targets(labels, node_adj, idx_node)
        if args.tp_classifier:
            idx = torch.cat((idx_node, idx_edge))
        else:
            scores[idx_node, 0] = 0
            scores[idx_node, 1] = 1
            idx = idx_edge
        # compute the f1 score
        pred = scores.data.max(1)[1]  # get the index of the max log-probability
        epoch_f1.append(f1_score(targets[idx].detach().cpu().numpy(), pred[idx].detach().cpu().numpy()))

        # loop through all frames
        for t_cur in range(t_st, t_end):
            # update graph for next timestep and run forward pass
            y_pred, feats, node_adj, edge_adj, labels = update_graph(node_adj, labels, scores, y_pred, X_seq, y_seq, t_cur, 
                use_hungraian=args.hungarian, mode='test', cuda=args.cuda)
            scores, logits, states = model(feats, states, node_adj, edge_adj)
            scores = torch.cat((1-scores, scores), dim=1)

            idx_edge = torch.nonzero((y_pred[:, 0] == -1))[:, 0]
            idx_node = torch.nonzero((y_pred[:, 0] != -1))[:, 0]
            # calculate targets for computing metrics
            targets = create_targets(labels, node_adj, idx_node)
            if args.tp_classifier:
                idx = torch.cat((idx_node, idx_edge))
            else:
                scores[idx_node, 0] = 0
                scores[idx_node, 1] = 1
                idx = idx_edge
            # compute the f1 score
            pred = scores.data.max(1)[1]  # get the index of the max log-probability
            epoch_f1.append(f1_score(targets[idx].detach().cpu().numpy(), pred[idx].detach().cpu().numpy()))

            # if no new detections are added, don't remove detections either
            if feats.size()[0] == 0:
                continue

            if t_cur == t_end - 1:
                y_pred, y_out, states, node_adj, labels, scores = decode_tracks(states, node_adj, labels, scores, y_pred, y_out, t_end, 
                    args.ret_win_size, use_hungraian=args.hungarian, cuda=args.cuda)
            else:
                y_pred, y_out, states, node_adj, labels, scores = decode_tracks(states, node_adj, labels, scores, y_pred, y_out, 
                    t_cur - args.cur_win_size + 2, args.ret_win_size, use_hungraian=args.hungarian, cuda=args.cuda)
            print("Sequence {}, generated tracks upto t = {}/{}...".format(b_idx + 1, max(0, t_cur - args.cur_win_size + 1), t_end))
        print("Sequence {}, generated tracks upto t = {}/{}...".format(b_idx + 1, t_end, t_end))

        # create results accumulator using predictions and GT for evaluation
        bbox_pred = bbox_pred[0, :, 2:].detach().cpu().numpy().astype('float32')
        y_gt = bbox_gt[0, :, :2].detach().cpu().numpy().astype('int64')
        bbox_gt = bbox_gt[0, :, 2:].detach().cpu().numpy().astype('float32')
        acc = create_mot_accumulator(bbox_pred, bbox_gt, y_out, y_gt)
        if acc is not None:
            accs.append(acc)
        # store values for computing mAP
        bbox_pred_dict[str(b_idx)] = (y_out, bbox_pred)
        bbox_gt_dict[str(b_idx)] = (y_gt, bbox_gt)

        print('Done with sequence {} of {}...'.format(b_idx + 1, len(val_loader.dataset)))

    # Calculate F1-score
    val_f1 = statistics.mean(epoch_f1)
    # Calculate MOTA
    if len(accs) > 0:
        val_motas = [100.0 * calc_mot_metrics([_])['mota'] for _ in accs]
        val_mota = 100.0 * calc_mot_metrics(accs)['mota']
    else:
        mota = -1
    # Calculate mAP
    val_map = 100.0 * compute_map(bbox_pred_dict, bbox_gt_dict)

    print("------------------------\nValidation F1 score = {:.4f}".format(val_f1))
    for seq_num, _ in enumerate(val_motas):
        print("Validation MOTA for sequence {:d} = {:.2f}%".format(seq_num, val_motas[seq_num]))
    print("Validation MOTA = {:.2f}%".format(val_mota))
    print("Validation mAP = {:.2f}%\n------------------------".format(val_map))
    f_log.write("\nValidation F1 score = {:.4f}\n".format(val_f1))
    for seq_num, _ in enumerate(val_motas):
        f_log.write("Validation MOTA for sequence {:d} = {:.2f}%\n".format(seq_num, val_motas[seq_num]))
    f_log.write("Validation MOTA = {:.2f}%\n".format(val_mota))
    f_log.write("Validation mAP = {:.2f}%\n------------------------\n\n".format(val_map))

    # now save the model if it has better MOTA than the best model seen so forward
    if val_mota > best_mota:
        best_mota = val_mota
        # save the TrackMPNN model and the detector
        torch.save(model.state_dict(), os.path.join(args.output_dir, 'track-mpnn_' + '%.4d' % (epoch,) + '.pth'))
        save_model(os.path.join(args.output_dir, 'dla-detector_' + '%.4d' % (epoch,) + '.pth'), epoch, val_loader.dataset.detector.model, optimizer=None)

    train_loader.dataset.detector = val_loader.dataset.detector # copy back the trained detector from the val loader
    val_loader.dataset.detector = None # set detector from val loader to None to save memory

    return val_f1, val_mota, val_map


if __name__ == '__main__':
    # for reproducibility
    random_seed(args.seed, args.cuda)

    # get the model, load pretrained weights, and convert it into cuda for if necessary
    model = TrackMPNN(nfeatures=args.num_img_feats + 5 + 4 + 5, nimgfeatures=args.num_img_feats, nhidden=args.num_hidden_feats, msg_type=args.msg_type)

    if args.snapshot is not None:
        model.load_state_dict(torch.load(args.snapshot), strict=True)
    if args.cuda:
        model.cuda()
    print(model)

    # optimizer for tracker
    optimizer_trk = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    # BCE(Focal) loss applied to each node/edge individually
    focal_loss_node = FocalLoss(gamma=0, alpha=None, size_average=True)
    focal_loss_edge = FocalLoss(gamma=0, alpha=None, size_average=True)
    # CE loss applied to edges collectively
    ce_loss = CELoss()

    fig1, ax1 = plt.subplots()
    plt.grid(True)
    ax1.plot([], 'r', label='Detector loss')
    ax1.plot([], 'g', label='Cross-entropy loss')
    ax1.plot([], 'b', label='Focal loss')
    ax1.plot([], 'k', label='Total loss')
    ax1.legend()
    train_loss_d, train_loss_c, train_loss_f, train_loss = list(), list(), list(), list()

    fig2, ax2 = plt.subplots()
    plt.grid(True)
    ax2.plot([], 'g', label='Train F1 score')
    ax2.plot([], 'b', label='Validation F1 score')
    ax2.legend()

    fig3, ax3 = plt.subplots()
    plt.grid(True)
    ax3.plot([], 'b', label='Validation MOTA')
    ax3.plot([], 'r', label='Validation mAP')
    ax3.legend()

    train_f1, val_f1, val_mota, val_map  = list(), list(), list(), list()

    for i in range(1, args.epochs + 1):
        model, avg_loss_d, avg_loss_c, avg_loss_f, avg_loss, avg_f1 = train(model, i)
        train_loss_d.append(avg_loss_d)
        train_loss_c.append(avg_loss_c)
        train_loss_f.append(avg_loss_f)
        train_loss.append(avg_loss)
        train_f1.append(avg_f1)

        # plot the loss
        ax1.plot(train_loss_d, 'r', label='Detector loss')
        ax1.plot(train_loss_c, 'g', label='Cross-entropy loss')
        ax1.plot(train_loss_f, 'b', label='Focal loss')
        ax1.plot(train_loss, 'k', label='Total loss')
        fig1.savefig(os.path.join(args.output_dir, "train_loss.jpg"))

        # clear GPU cahce and free up memory
        torch.cuda.empty_cache()

        # plot the train and val F1 scores and MOTAs
        f1, mota, mAP = val(model, i)
        val_f1.append(f1)
        val_mota.append(mota)
        val_map.append(mAP)

        # clear GPU cahce and free up memory
        torch.cuda.empty_cache()

        ax2.plot(train_f1, 'g', label='Train F1 score')
        ax2.plot(val_f1, 'b', label='Validation F1 score')
        fig2.savefig(os.path.join(args.output_dir, 'train_val_f1.jpg'))

        ax3.plot(val_mota, 'b', label='Validation MOTA')
        ax3.plot(val_map, 'r', label='Validation mAP')
        fig3.savefig(os.path.join(args.output_dir, 'val_mota+map.jpg'))

    plt.close('all')
    f_log.close()
