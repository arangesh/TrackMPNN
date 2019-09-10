import os
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import statistics
from sklearn.metrics import f1_score

import torch
from torch.utils.data import DataLoader
import torch.optim as optim

from models.track_mpnn import TrackMPNN
from dataset.kitti_mots import KittiMOTSDataset
from utils.graph import initialize_graph, update_graph, prune_graph, decode_tracks
from utils.metrics import create_mot_accumulator, calc_mot_metrics
from utils.training_options import args
from models.loss import FocalLoss


kwargs = {'batch_size': 1, 'shuffle': True, 'num_workers': 1}
train_loader = DataLoader(KittiMOTSDataset(args.dataset_root_path, 'train', args.timesteps), **kwargs)
val_loader = DataLoader(KittiMOTSDataset(args.dataset_root_path, 'val', args.timesteps), **kwargs)

# global var to store best MOTA across all epochs
best_mota = -float('Inf')


# random seed function (https://docs.fast.ai/dev/test.html#getting-reproducible-results)
def random_seed(seed_value, use_cuda):
    torch.manual_seed(seed_value)
    if use_cuda:
        torch.backends.cudnn.deterministic = True  #needed


# training function
def train(model, epoch):
    epoch_loss, epoch_f1 = list(), list()
    model.train()
    for b_idx, (X, y) in enumerate(train_loader):
        if type(X) == type([]) or type(y) == type([]):
            continue
        if args.cuda:
            X, y = X.cuda(), y.cuda()
        # backpropagate gradient through feature matrix
        X.requires_grad = True

        # train the network
        optimizer.zero_grad()

        # intialize graph and run first forward pass
        y_pred, feats, node_adj, edge_adj, labels, t_init, t_end = initialize_graph(X, y, cuda=args.cuda)
        if y_pred is None:
            continue
        scores = model.forward(feats, node_adj, edge_adj)
        # compute the loss
        idx_edge = torch.nonzero((y_pred[:, 0] == -1))[:, 0]
        idx_node = torch.nonzero((y_pred[:, 0] != -1))[:, 0]
        if args.tp_classifier:
            loss = focal_loss_node(scores[idx_node, 0], labels[idx_node]) + focal_loss_edge(scores[idx_edge, 0], labels[idx_edge])
            scores = torch.cat((1-scores, scores), dim=1)
            idx = torch.cat((idx_node, idx_edge))
        else:
            loss = focal_loss_edge(scores[idx_edge, 0], labels[idx_edge])
            scores = torch.cat((1-scores, scores), dim=1)
            scores[idx_node, 0] = 0
            scores[idx_node, 1] = 1
            idx = idx_edge
        # compute the f1 score
        pred = scores.data.max(1)[1]  # get the index of the max log-probability
        epoch_f1.append(f1_score(labels[idx].detach().cpu().numpy(), pred[idx].detach().cpu().numpy()))

        # loop through all frames
        for t in range(t_init, t_end):
            # update graph for next timestep and run forward pass
            y_pred, feats, node_adj, edge_adj, labels = update_graph(feats, node_adj, labels, scores, y_pred, X, y, t, use_hungraian=args.hungarian, mode='train', cuda=args.cuda)
            scores = model.forward(feats, node_adj, edge_adj)
            # compute the loss
            idx_edge = torch.nonzero((y_pred[:, 0] == -1))[:, 0]
            idx_node = torch.nonzero((y_pred[:, 0] != -1))[:, 0]
            if args.tp_classifier:
                loss += (focal_loss_node(scores[idx_node, 0], labels[idx_node]) + focal_loss_edge(scores[idx_edge, 0], labels[idx_edge]))
                scores = torch.cat((1-scores, scores), dim=1)
                idx = torch.cat((idx_node, idx_edge))
            else:
                loss += focal_loss_edge(scores[idx_edge, 0], labels[idx_edge])
                scores = torch.cat((1-scores, scores), dim=1)
                scores[idx_node, 0] = 0
                scores[idx_node, 1] = 1
                idx = idx_edge
            # compute the F1 score
            pred = scores.data.max(1)[1]  # get the index of the max log-probability
            epoch_f1.append(f1_score(labels[idx].detach().cpu().numpy(), pred[idx].detach().cpu().numpy()))

        epoch_loss.append(loss.item())
        loss.backward()
        optimizer.step()

        if b_idx % args.log_schedule == 0:
            print('Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.5f}'.format(
                epoch, (b_idx + 1), len(train_loader.dataset),
                100. * (b_idx + 1) / len(train_loader.dataset), loss.item()))
            with open(os.path.join(args.output_dir, "logs.txt"), "a") as f:
                f.write('Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.5f}\n'.format(
                epoch, (b_idx + 1), len(train_loader.dataset),
                100. * (b_idx + 1) / len(train_loader.dataset), loss.item()))

    # now that the epoch is completed calculate statistics and store logs
    avg_loss = statistics.mean(epoch_loss)
    avg_f1 = statistics.mean(epoch_f1)
    print("------------------------\nAverage loss for epoch = {:.2f}".format(avg_loss))
    with open(os.path.join(args.output_dir, "logs.txt"), "a") as f:
        f.write("------------------------\nAverage loss for epoch = {:.2f}\n".format(avg_loss))
    
    print("Average F1 score for epoch = {:.4f}\n------------------------".format(avg_f1))
    with open(os.path.join(args.output_dir, "logs.txt"), "a") as f:
        f.write("Average F1 score for epoch = {:.4f}\n".format(avg_f1))

    return model, avg_loss, avg_f1


# validation function
def val(model, epoch):
    global best_mota
    epoch_f1 = list()
    accs = []
    model.eval()

    for b_idx, (X, y) in enumerate(val_loader):
        if type(X) == type([]) or type(y) == type([]):
            continue
        if args.cuda:
            X, y = X.cuda(), y.cuda()

        # initaialize output array tracks to -1s
        y_out = y.squeeze(0).detach().cpu().numpy().astype('int64')
        y_out[:, 1] = -1

        # intialize graph and run first forward pass
        y_pred, feats, node_adj, edge_adj, labels, t_init, t_end = initialize_graph(X, y, cuda=args.cuda)
        if y_pred is None:
            continue
        # compute the classification scores
        scores = model.forward(feats, node_adj, edge_adj)
        scores = torch.cat((1-scores, scores), dim=1)

        idx_edge = torch.nonzero((y_pred[:, 0] == -1))[:, 0]
        idx_node = torch.nonzero((y_pred[:, 0] != -1))[:, 0]
        if args.tp_classifier:
            idx = torch.cat((idx_node, idx_edge))
        else:
            scores[idx_node, 0] = 0
            scores[idx_node, 1] = 1
            idx = idx_edge
        # compute the f1 score
        pred = scores.data.max(1)[1]  # get the index of the max log-probability
        epoch_f1.append(f1_score(labels[idx].detach().cpu().numpy(), pred[idx].detach().cpu().numpy()))

        # loop through all frames
        for t in range(t_init, t_end):
            # update graph for next timestep and run forward pass
            y_pred, feats, node_adj, edge_adj, labels = update_graph(feats, node_adj, labels, scores, y_pred, X, y, t, use_hungraian=args.hungarian, mode='test', cuda=args.cuda)
            scores = model.forward(feats, node_adj, edge_adj)
            scores = torch.cat((1-scores, scores), dim=1)

            idx_edge = torch.nonzero((y_pred[:, 0] == -1))[:, 0]
            idx_node = torch.nonzero((y_pred[:, 0] != -1))[:, 0]
            if args.tp_classifier:
                idx = torch.cat((idx_node, idx_edge))
            else:
                scores[idx_node, 0] = 0
                scores[idx_node, 1] = 1
                idx = idx_edge
            # compute the f1 score
            pred = scores.data.max(1)[1]  # get the index of the max log-probability
            epoch_f1.append(f1_score(labels[idx].detach().cpu().numpy(), pred[idx].detach().cpu().numpy()))
            if t == t_end - 1:
                y_pred, y_out, feats, node_adj, labels, scores = decode_tracks(feats, node_adj, labels, scores, y_pred, y_out, t_end, use_hungraian=args.hungarian, cuda=args.cuda)
            else:
                y_pred, y_out, feats, node_adj, labels, scores = decode_tracks(feats, node_adj, labels, scores, y_pred, y_out, t - args.timesteps + 2, use_hungraian=args.hungarian, cuda=args.cuda)
            print("Sequence {}, generated tracks upto t = {}/{}...".format(b_idx + 1, max(0, t - args.timesteps + 1), t_end))
        print("Sequence {}, generated tracks upto t = {}/{}...".format(b_idx + 1, t_end, t_end))
        # create results accumulator using predictions and GT for evaluation
        acc = create_mot_accumulator(y_out, X, y)
        if acc is not None:
            accs.append(acc)

        print('Done with sequence {} out {}...'.format(min(b_idx + 1, len(val_loader.dataset)), len(val_loader.dataset)))

    if len(accs) > 0:
        mota = calc_mot_metrics(accs)['mota']

    val_f1 = statistics.mean(epoch_f1)
    val_mota = 100.0 * mota
    print("------------------------\nValidation F1 score = {:.4f}".format(val_f1))
    print("Validation MOTA = {:.2f}%\n------------------------".format(val_mota))
    with open(os.path.join(args.output_dir, "logs.txt"), "a") as f:
        f.write("\nValidation F1 score = {:.4f}\n".format(val_f1))
        f.write("Validation MOTA = {:.2f}%\n------------------------\n\n".format(val_mota))

    # now save the model if it has better MOTA than the best model seen so forward
    if val_mota > best_mota:
        best_mota = val_mota
        # save the model
        torch.save(model.state_dict(), os.path.join(args.output_dir, 'track-mpnn_' + '%.4d' % (epoch,) + '.pth'))

    return val_f1, val_mota


if __name__ == '__main__':
    # for reproducibility
    random_seed(args.seed, args.cuda)

    # get the model, load pretrained weights, and convert it into cuda for if necessary
    model = TrackMPNN(nfeat=1 + 4 + 64 + 10 - 10 + 64, nhid=args.hidden)

    if args.snapshot is not None:
        model.load_state_dict(torch.load(args.snapshot), strict=False)
    if args.cuda:
        model.cuda()
    print(model)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    focal_loss_node = FocalLoss(gamma=2, alpha=0.25, size_average=True)
    focal_loss_edge = FocalLoss(gamma=2, alpha=0.25, size_average=True)

    fig1, ax1 = plt.subplots()
    plt.grid(True)
    train_loss = list()

    fig2, ax2 = plt.subplots()
    plt.grid(True)
    ax2.plot([], 'g', label='Train F1 score')
    ax2.plot([], 'b', label='Validation F1 score')
    ax2.legend()

    fig3, ax3 = plt.subplots()
    plt.grid(True)
    ax3.plot([], 'b', label='Validation MOTA')
    ax3.legend()

    train_f1, val_f1, val_mota  = list(), list(), list()

    for i in range(1, args.epochs + 1):
        model, avg_loss, avg_f1 = train(model, i)
        train_f1.append(avg_f1)

        # plot the loss
        train_loss.append(avg_loss)
        ax1.plot(train_loss, 'k')
        fig1.savefig(os.path.join(args.output_dir, "train_loss.jpg"))

        # plot the train and val F1 scores and MOTAs
        f1, mota = val(model, i)
        val_f1.append(f1)
        val_mota.append(mota)

        ax2.plot(train_f1, 'g', label='Train F1 score')
        ax2.plot(val_f1, 'b', label='Validation F1 score')
        fig2.savefig(os.path.join(args.output_dir, 'train_val_f1.jpg'))

        ax3.plot(val_mota, 'b', label='Validation MOTA')
        fig3.savefig(os.path.join(args.output_dir, 'val_mota.jpg'))

    plt.close('all')
