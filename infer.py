import os

import torch
from torch.utils.data import DataLoader
import torch.optim as optim

from models.track_mpnn import TrackMPNN
from dataset.kitti_mot import KittiMOTDataset, store_kitti_results
from utils.graph import initialize_graph, update_graph, prune_graph, decode_tracks
from utils.infer_options import args


kwargs_infer = {'batch_size': 1, 'shuffle': False}
if 'vis' in args.feats:
    vis_snapshot = os.path.join(os.path.dirname(args.snapshot), 'vis-net_' + args.snapshot[-8:])
else:
    vis_snapshot = None
infer_loader = DataLoader(KittiMOTDataset(args.dataset_root_path, 'test', args.category, args.detections, args.feats, 
    args.embed_arch, args.cur_win_size, args.ret_win_size, vis_snapshot, False, args.cuda), **kwargs_infer)


# random seed function (https://docs.fast.ai/dev/test.html#getting-reproducible-results)
def random_seed(seed_value, use_cuda):
    torch.manual_seed(seed_value)
    if use_cuda:
        torch.backends.cudnn.deterministic = True  #needed


# inference function
def infer(model):
    model.eval()
    for b_idx, (X_seq, bbox_pred, _, _) in enumerate(infer_loader):
        if X_seq.size()[1] == 0:
            print('No detections available for sequence...')
            continue
        y_seq = bbox_pred[:, :, :2]

        # initaialize output array tracks to -1s
        y_out = y_seq.squeeze(0).detach().cpu().numpy().astype('int64')
        y_out[:, 1] = -1

        # intialize graph and run first forward pass
        y_pred, feats, node_adj, edge_adj, labels, t_st, t_end = initialize_graph(X_seq, y_seq, t_st=0, mode='test', cuda=args.cuda)
        
        # compute the classification scores
        scores, logits, states = model(feats, None, node_adj, edge_adj)
        scores = torch.cat((1-scores, scores), dim=1)
        if not args.tp_classifier:
            idx_node = torch.nonzero((y_pred[:, 0] != -1))[:, 0]
            scores[idx_node, 0] = 0
            scores[idx_node, 1] = 1

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
            scores, logits, states = model(feats, states, node_adj, edge_adj)
            scores = torch.cat((1-scores, scores), dim=1)
            if not args.tp_classifier:
                idx_node = torch.nonzero((y_pred[:, 0] != -1))[:, 0]
                scores[idx_node, 0] = 0
                scores[idx_node, 1] = 1

            if t_cur == t_end - 1:
                y_pred, y_out, states, node_adj, labels, scores = decode_tracks(states, node_adj, labels, scores, y_pred, y_out, t_end, 
                    args.ret_win_size, use_hungraian=args.hungarian, cuda=args.cuda)
            else:
                y_pred, y_out, states, node_adj, labels, scores = decode_tracks(states, node_adj, labels, scores, y_pred, y_out, 
                    t_cur - args.cur_win_size + 2, args.ret_win_size, use_hungraian=args.hungarian, cuda=args.cuda)
            print("Sequence {}, generated tracks upto t = {}/{}...".format(b_idx + 1, max(0, t_cur - args.cur_win_size + 1), t_end))
        print("Sequence {}, generated tracks upto t = {}/{}...".format(b_idx + 1, t_end, t_end))

        # store results in KITTI format
        bbox_pred = bbox_pred[0, :, 2:].detach().cpu().numpy().astype('float32')
        store_kitti_results(bbox_pred, y_out, infer_loader.dataset.class_dict, os.path.join(args.output_dir, '%.4d.txt' % (b_idx,)))
        print('Done with sequence {} out {}...\n'.format(b_idx + 1, len(infer_loader.dataset)))

    return


if __name__ == '__main__':
    # for reproducibility
    random_seed(args.seed, args.cuda)

    # get the model, load pretrained weights, and convert it into cuda for if necessary
    num_features = 3# for one-hot category IDs
    if '2d' in args.feats:
        num_features += 5
    if 'temp' in args.feats:
        num_features += 2
    if 'vis' in args.feats:
        num_features += 16
    model = TrackMPNN(nfeatures=num_features, nhidden=args.num_hidden_feats, nattheads=args.num_att_heads, msg_type=args.msg_type)
    model.load_state_dict(torch.load(args.snapshot), strict=True)
    if args.cuda:
        model.cuda()
    print(model)

    infer(model)
