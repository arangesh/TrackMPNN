import os
import matplotlib.pyplot as plt
import statistics

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
#import torch.nn.functional as F

from models.track_mpnn import TrackMPNN
from dataset.kitti_mots import KittiMOTSDataset
from utils.graph import initialize_graph, update_graph, prune_graph, decode_tracks
from utils.metrics import create_mot_accumulator, calc_mot_metrics
from utils.training_options import args
from utils.loss import FocalLoss


# This will set both cpu and gpu: https://pytorch.org/docs/stable/notes/randomness.html
torch.manual_seed(args.seed)

kwargs = {'batch_size': 1, 'shuffle': True, 'num_workers': 1}
train_loader = DataLoader(KittiMOTSDataset(args.dataset_root_path, 'train', args.timesteps), **kwargs)
val_loader = DataLoader(KittiMOTSDataset(args.dataset_root_path, 'val', args.timesteps), **kwargs)

# global var to store best validation accuracy across all epochs
best_mota = -float('Inf')
# get float type for label conversion
if args.cuda:
    float_type = 'torch.cuda.FloatTensor'
else:
    float_type = 'torch.FloatTensor'


# training function
def train(model, epoch):
    epoch_loss = list()
    correct = 0.
    total = 0.
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
        y_pred, feats, node_adj, edge_adj, labels, t_init, t_end = initialize_graph(X, y, mode='train', cuda=args.cuda)
        if y_pred is None:
            continue
        scores = model.forward(feats, node_adj, edge_adj)
        # compute the loss
        if args.tp_classifier:
            idx = torch.nonzero(labels != -1)[:, 0]
            loss = focal_loss(scores[idx, 0], labels[idx])
            #loss = F.binary_cross_entropy(scores[idx, 0], labels[idx].type(float_type))
            scores = torch.cat((1-scores, scores), dim=1)
        else:
            idx = torch.nonzero((y_pred[:, 0] == -1) & (labels != -1))[:, 0]
            loss = focal_loss(scores[idx, 0], labels[idx])
            #loss = F.binary_cross_entropy(scores[idx, 0], labels[idx].type(float_type))
            scores = torch.cat((1-scores, scores), dim=1)
            ids = torch.nonzero(y_pred[:, 0] != -1)[:, 0]
            scores[ids, 0] = 0
            scores[ids, 1] = 1

        # compute the accuracy
        pred = scores.data.max(1)[1]  # get the index of the max log-probability
        correct += float(pred[idx].eq(labels[idx].data).cpu().sum())
        total += float(labels[idx].size()[0])
        # intialize graph and run first forward pass
        for t in range(t_init, t_end):
            # update graph for next timestep and run forward pass
            y_pred, feats, node_adj, edge_adj, labels = update_graph(feats, node_adj, labels, scores, y_pred, X, y, t, use_hungraian=args.hungarian, mode='train', cuda=args.cuda)
            scores = model.forward(feats, node_adj, edge_adj)
            # compute the loss
            if args.tp_classifier:
                idx = torch.nonzero(labels != -1)[:, 0]
                loss += focal_loss(scores[idx, 0], labels[idx])
                #loss += F.binary_cross_entropy(scores[idx, 0], labels[idx].type(float_type))
                scores = torch.cat((1-scores, scores), dim=1)
            else:
                idx = torch.nonzero((y_pred[:, 0] == -1) & (labels != -1))[:, 0]
                loss += focal_loss(scores[idx, 0], labels[idx])
                #loss += F.binary_cross_entropy(scores[idx, 0], labels[idx].type(float_type))
                scores = torch.cat((1-scores, scores), dim=1)
                ids = torch.nonzero(y_pred[:, 0] != -1)[:, 0]
                scores[ids, 0] = 0
                scores[ids, 1] = 1
            # compute the accuracy
            pred = scores.data.max(1)[1]  # get the index of the max log-probability
            correct += float(pred[idx].eq(labels[idx].data).cpu().sum())
            total += float(labels[idx].size()[0])

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
    print("------------------------\nAverage loss for epoch = {:.2f}".format(avg_loss))
    with open(os.path.join(args.output_dir, "logs.txt"), "a") as f:
        f.write("\n------------------------\nAverage loss for epoch = {:.2f}\n".format(avg_loss))
    
    train_accuracy = 100.0 * correct / total
    print("Accuracy for epoch = {:.2f}%\n------------------------".format(train_accuracy))
    with open(os.path.join(args.output_dir, "logs.txt"), "a") as f:
        f.write("Accuracy for epoch = {:.2f}%\n------------------------\n".format(train_accuracy))

    return model, avg_loss, train_accuracy


# validation function
def val(model, epoch):
    global best_mota
    correct = 0.
    total = 0.
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
        y_pred, feats, node_adj, edge_adj, labels, t_init, t_end = initialize_graph(X, y, mode='train', cuda=args.cuda)
        if y_pred is None:
            continue
        # compute the classification scores
        scores = model.forward(feats, node_adj, edge_adj)
        scores = torch.cat((1-scores, scores), dim=1)
        if not args.tp_classifier:
            ids = torch.nonzero(y_pred[:, 0] != -1)[:, 0]
            scores[ids, 0] = 0
            scores[ids, 1] = 1
        # compute the accuracy
        pred = scores.data.max(1)[1]  # get the index of the max log-probability
        correct += float(pred.eq(labels.data).cpu().sum())
        total += float(labels.size()[0])
        # intialize graph and run first forward pass
        for t in range(t_init, t_end):
            # update graph for next timestep and run forward pass
            y_pred, feats, node_adj, edge_adj, labels = update_graph(feats, node_adj, labels, scores, y_pred, X, y, t, use_hungraian=args.hungarian, mode='train', cuda=args.cuda)
            scores = model.forward(feats, node_adj, edge_adj)
            scores = torch.cat((1-scores, scores), dim=1)
            if not args.tp_classifier:
                ids = torch.nonzero(y_pred[:, 0] != -1)[:, 0]
                scores[ids, 0] = 0
                scores[ids, 1] = 1
            # compute the accuracy
            pred = scores.data.max(1)[1]  # get the index of the max log-probability
            correct += float(pred.eq(labels.data).cpu().sum())
            total += float(labels.size()[0])
            # prune graph to reduce memory requirements
            if not args.tp_classifier:
                y_pred, feats, node_adj, labels, scores = prune_graph(feats, node_adj, labels, scores, y_pred, 0, t - 1, threshold=0.5, cuda=args.cuda)
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

    print("------------------------\nPredicted {} out of {}".format(correct, total))
    val_accuracy = 100.0 * correct / total
    val_mota = 100.0 * mota
    print("Validation accuracy = {:.2f}%".format(val_accuracy))
    print("Validation MOTA = {:.2f}%\n------------------------".format(val_mota))
    with open(os.path.join(args.output_dir, "logs.txt"), "a") as f:
        f.write("\n------------------------\nPredicted {} out of {}\n".format(int(correct), int(total)))
        f.write("Validation accuracy = {:.2f}%\n".format(val_accuracy))
        f.write("Validation MOTA = {:.2f}%\n------------------------\n".format(val_mota))

    # now save the model if it has better accuracy than the best model seen so forward
    if val_mota > best_mota:
        best_mota = val_mota
        # save the model
        torch.save(model.state_dict(), os.path.join(args.output_dir, 'track-mpnn_' + '%.4d' % (epoch,) + '.pth'))

    return val_accuracy, val_mota


if __name__ == '__main__':
    # get the model, load pretrained weights, and convert it into cuda for if necessary
    model = TrackMPNN(nfeat=1 + 4 + 64 + 10 - 10 + 64, nhid=args.hidden)

    if args.snapshot is not None:
        model.load_state_dict(torch.load(args.snapshot), strict=False)
    if args.cuda:
        model.cuda()
    print(model)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    focal_loss = FocalLoss(gamma=2, alpha=0.25, size_average=True)

    fig1, ax1 = plt.subplots()
    plt.grid(True)
    train_loss = list()

    fig2, ax2 = plt.subplots()
    plt.grid(True)
    ax2.plot([], 'g', label='Train accuracy')
    ax2.plot([], 'b', label='Validation accuracy')
    ax2.legend()

    fig3, ax3 = plt.subplots()
    plt.grid(True)
    ax3.plot([], 'b', label='Validation MOTA')
    ax3.legend()

    train_acc, val_acc, val_mota = list(), list(), list()

    for i in range(1, args.epochs + 1):
        model, avg_loss, acc = train(model, i)
        train_acc.append(acc)

        # plot the loss
        train_loss.append(avg_loss)
        ax1.plot(train_loss, 'k')
        fig1.savefig(os.path.join(args.output_dir, "train_loss.jpg"))

        # plot the train and val accuracies and MOTAs
        acc, mota = val(model, i)
        val_acc.append(acc)
        val_mota.append(mota)

        ax2.plot(train_acc, 'g', label='Train accuracy')
        ax2.plot(val_acc, 'b', label='Validation accuracy')
        fig2.savefig(os.path.join(args.output_dir, 'train_val_accuracy.jpg'))

        ax3.plot(val_mota, 'b', label='Validation MOTA')
        fig3.savefig(os.path.join(args.output_dir, 'val_mota.jpg'))
    plt.close('all')
