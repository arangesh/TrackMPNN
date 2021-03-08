import os
import pickle
import statistics
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
import glob

import torch
from torch.utils.data import DataLoader

from models.track_mpnn import TrackMPNN
from models.loss import create_targets
from dataset.kitti_mot import KittiMOTDataset, store_kitti_results
from utils.graph import initialize_graph, update_graph, prune_graph, decode_tracks
from utils.metrics import create_mot_accumulator, calc_mot_metrics, compute_map
from utils.infer_options import args


kwargs_train = {'batch_size': 1, 'shuffle': True}
kwargs_val = {'batch_size': 1, 'shuffle': False}
if 'vis' in args.feats:
    vis_snapshot = os.path.join(os.path.dirname(args.snapshot), 'vis-net_' + args.snapshot[-8:])
    train_loader = DataLoader(KittiMOTDataset(args.dataset_root_path, 'train', args.category, args.detections, args.feats, 
        args.embed_arch, args.cur_win_size, args.ret_win_size, vis_snapshot, False, args.cuda), **kwargs_train)
    val_loader = DataLoader(KittiMOTDataset(args.dataset_root_path, 'val', args.category, args.detections, args.feats, 
        args.embed_arch, args.cur_win_size, args.ret_win_size, vis_snapshot, False, args.cuda), **kwargs_val)
    val_loader.dataset.embed_net = train_loader.dataset.embed_net # use trained embedding net for the val loader
    train_loader.dataset.embed_net = None # set trained embedding net to None to save memory
else:
    val_loader = DataLoader(KittiMOTDataset(args.dataset_root_path, 'val', args.category, args.detections, args.feats, 
        args.embed_arch, args.cur_win_size, args.ret_win_size, None, False, args.cuda), **kwargs_val)

# create file handles
f_log = open(os.path.join(args.output_dir, "logs.txt"), "w")

# random seed function (https://docs.fast.ai/dev/test.html#getting-reproducible-results)
def random_seed(seed_value, use_cuda):
    torch.manual_seed(seed_value)
    if use_cuda:
        torch.backends.cudnn.deterministic = True  #needed

def store_att_weights(folder, sequence_index, data):
    # labels, y_pred
    labels = data[0]
    y_pred = data[1]

    # if using multiple sets of features, choose which attention weights to save
    feature_set = 0
    if len(data) == 3:
        attention = [att.cpu().detach().numpy() for att in data[2][feature_set]]
        dict_to_pickle = {'labels' : labels.cpu().numpy(), 'y_pred' : y_pred.cpu().numpy(),
                        'attention' : attention}
    else:
        dict_to_pickle = {'labels' : labels.cpu().numpy(), 'y_pred' : y_pred.cpu().numpy()}
    path = os.path.join(folder, f"{sequence_index}.p")

    with open(path, "wb" ) as f:
        pickle.dump( dict_to_pickle, f)

def plot_att_distribution():
    results = [{'tp' : [], 'fp' : []} for i in range(args.num_att_heads)]
    filelist = glob.glob(os.path.join(args.output_dir, '*.p'))

    for count, file in enumerate(filelist):
        with open(file, 'rb') as f:
            data = pickle.load(f)

        labels = data['labels']
        y_pred = data['y_pred']
        attention = data['attention']

        N = attention[0].shape[0]
        for row_index in range(N):
            if y_pred[row_index][0] != -1:
                for col_index in range(N):
                    for i, attention_i in enumerate(attention):
                        if attention_i[row_index][col_index] > 0:
                            if labels[col_index] == 1:
                                results[i]['tp'].append(attention_i[row_index][col_index])
                            else:
                                results[i]['fp'].append(attention_i[row_index][col_index])
        print("Completed processing file %d/%d..." % (count, len(filelist)))

    fig_ax = [plt.subplots(2, 1, figsize=(6.4, 4.8*len(results))) for _ in range(args.num_att_heads)]
    for i, (fig, ax) in enumerate(fig_ax):
        num_bins = 100
        ax[0].hist(results[i]['tp'], num_bins, range=(0.0, 0.7), density=True, stacked=True)
        ax[1].hist(results[i]['fp'], num_bins, range=(0.0, 0.7), density=True, stacked=True)
        fig.savefig(os.path.join(args.output_dir, 'att_dist_%d.jpg' % (i,)))
    plt.close('all')

def val(model):
    epoch_f1 = list()
    accs = []
    model.eval() # set TrackMPNN model to eval mode
    if 'vis' in args.feats:
        val_loader.dataset.embed_net.eval()

    bbox_pred_dict, bbox_gt_dict = {}, {} # initialize dictionaries for computing mAP
    _att_ind = 0
    for b_idx, (X_seq, bbox_pred, bbox_gt, _) in enumerate(val_loader):
        # if no detections in sequence
        if X_seq.size()[1] == 0 or bbox_gt.shape[1] == 0:
            print('No detections available for sequence...')
            continue
        y_seq = bbox_pred[:, :, :2]

        # initaialize output array tracks to -1s
        y_out = y_seq.squeeze(0).detach().cpu().numpy().astype('int64')
        y_out[:, 1] = -1

        # intialize graph and run first forward pass
        y_pred, feats, node_adj, edge_adj, labels, t_st, t_end = initialize_graph(X_seq, y_seq, t_st=0, mode='test', cuda=args.cuda)
        if y_pred is None:
            continue
        # compute the classification scores
        scores, logits, states, _ = model(feats, None, node_adj, edge_adj)
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
        epoch_f1.append(f1_score(targets[idx].detach().cpu().numpy(), pred[idx].detach().cpu().numpy(), zero_division=0))

        # loop through all frames
        t_skip = t_st
        for t_cur in range(t_st, t_end):
            if t_cur < t_skip: # if timestep has already been processed
                continue
            # if no new detections found and no carried over detections
            if feats.size()[0] == 0 and states.size()[0] == 0:
                # reinitialize graph
                y_pred, feats, node_adj, edge_adj, labels, t_skip, _ = initialize_graph(X_seq, y_seq, t_st=t_cur, mode='test', cuda=args.cuda)
                if y_pred is None:
                    break
                states = None
            else:
                # update graph for next timestep
                y_pred, feats, node_adj, edge_adj, labels = update_graph(node_adj, labels, scores, y_pred, X_seq, y_seq, t_cur, 
                    use_hungraian=args.hungarian, mode='test', cuda=args.cuda)
            # run forward pass
            scores, logits, states, attention = model(feats, states, node_adj, edge_adj)
            store_att_weights(args.output_dir, _att_ind, [labels, y_pred, attention])
            _att_ind += 1
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
            epoch_f1.append(f1_score(targets[idx].detach().cpu().numpy(), pred[idx].detach().cpu().numpy(), zero_division=0))

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
        bbox_pred_dict[str(b_idx)] = (y_out[y_out[:, 1] >= 0, :], bbox_pred[y_out[:, 1] >= 0, :])
        bbox_gt_dict[str(b_idx)] = (y_gt, bbox_gt)

        print('Done with sequence {} of {}...'.format(b_idx + 1, len(val_loader.dataset)))

    # Calculate F1-score
    val_f1 = statistics.mean(epoch_f1)
    # Calculate MOTA
    if len(accs) > 0:
        val_motas = [100.0 * calc_mot_metrics([_])['mota'] for _ in accs]
        val_metrics = calc_mot_metrics(accs)
    else:
        mota = -1
    # Calculate mAP
    val_map = 100.0 * compute_map(bbox_pred_dict, bbox_gt_dict)

    print("------------------------\nValidation F1 score = {:.4f}".format(val_f1))
    for seq_num, _ in enumerate(val_motas):
        print("Validation MOTA for sequence {:d} = {:.2f}%".format(seq_num, val_motas[seq_num]))
    print("Validation MOTA = {:.2f}".format(100.*val_metrics['mota']))
    print("Validation MOTP = {:.4f}".format(val_metrics['motp']))
    print("Validation MT = {:.2f}%".format(100.*val_metrics['mostly_tracked']/val_metrics['num_unique_objects']))
    print("Validation ML = {:.2f}%".format(100.*val_metrics['mostly_lost']/val_metrics['num_unique_objects']))
    print("Validation IDS = {:d}".format(val_metrics['num_switches']))
    print("Validation FRAG = {:d}".format(val_metrics['num_fragmentations']))
    print("Validation mAP = {:.2f}\n------------------------\n".format(val_map))
    f_log.write("------------------------\nValidation F1 score = {:.4f}\n".format(val_f1))
    for seq_num, _ in enumerate(val_motas):
        f_log.write("Validation MOTA for sequence {:d} = {:.2f}%\n".format(seq_num, val_motas[seq_num]))
    f_log.write("Validation MOTA = {:.2f}\n".format(100.*val_metrics['mota']))
    f_log.write("Validation MOTP = {:.4f}\n".format(val_metrics['motp']))
    f_log.write("Validation MT = {:.2f}%\n".format(100.*val_metrics['mostly_tracked']/val_metrics['num_unique_objects']))
    f_log.write("Validation ML = {:.2f}%\n".format(100.*val_metrics['mostly_lost']/val_metrics['num_unique_objects']))
    f_log.write("Validation IDS = {:d}\n".format(val_metrics['num_switches']))
    f_log.write("Validation FRAG = {:d}\n".format(val_metrics['num_fragmentations']))
    f_log.write("Validation mAP = {:.2f}\n------------------------\n\n".format(val_map))

    return

if __name__ == '__main__': 
    # for reproducibility
    random_seed(args.seed, args.cuda)
    # get the model, load pretrained weights, and convert it into cuda for if necessary
    model = TrackMPNN(features=args.feats, ncategories=len(val_loader.dataset.class_dict), 
        nhidden=args.num_hidden_feats, nattheads=args.num_att_heads, msg_type=args.msg_type)
    if args.snapshot is not None:
        model.load_state_dict(torch.load(args.snapshot), strict=True)
    if args.cuda:
        model.cuda()
    print(model)

    val(model)
    plot_att_distribution()
