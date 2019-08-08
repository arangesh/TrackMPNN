import argparse
import os
import datetime
import matplotlib.pyplot as plt
import statistics

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F

from models.track_mpnn import TrackMPNN
from dataset.kitti_mots import KittiMOTSDataset
from utils.graph import initialize_graph, update_graph, decode_tracks
from utils.metrics import create_mot_accumulator, calc_mot_metrics

parser = argparse.ArgumentParser('Options for training Track-MPNN models in PyTorch...')

parser.add_argument('--dataset-root-path', type=str, default='/home/akshay/data/kitti-mots', help='path to dataset')
parser.add_argument('--output-dir', type=str, default=None, help='output directory for model and logs')
parser.add_argument('--snapshot', type=str, default=None, help='use a pre-trained model snapshot')
parser.add_argument('--timesteps', type=int, default=10, metavar='TS', help='number of timesteps to train on')
parser.add_argument('--hidden', type=int, default=64, metavar='NH', help='number of hidden layer nodes')
parser.add_argument('--epochs', type=int, default=50, metavar='EP', help='number of epochs to train for')
parser.add_argument('--learning-rate', type=float, default=0.0001, metavar='LR', help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='momentum for gradient step')
parser.add_argument('--weight-decay', type=float, default=0.0005, metavar='WD', help='weight decay')
parser.add_argument('--log-schedule', type=int, default=10, metavar='N', help='number of iterations to print/save log after')
parser.add_argument('--seed', type=int, default=1, help='set seed to some constant value to reproduce experiments')
parser.add_argument('--no-cuda', action='store_true', default=False, help='do not use cuda for training')


args = parser.parse_args()

# setup args
args.cuda = not args.no_cuda and torch.cuda.is_available()
if args.output_dir is None:
    args.output_dir = datetime.datetime.now().strftime("%I:%M%p-%B-%d-%Y")
    if args.snapshot is not None:
        args.output_dir = args.output_dir + '-w/snapshot-' + os.path.basename(args.snapshot)[:-4]
    args.output_dir = os.path.join('..', 'experiments', args.output_dir)

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)
else:
    assert False, 'Output directory already exists!'

# This will set both cpu and gpu: https://pytorch.org/docs/stable/notes/randomness.html
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


kwargs = {'batch_size': 1, 'shuffle': True, 'num_workers': 1}
train_loader = DataLoader(KittiMOTSDataset(args.dataset_root_path, 'train', args.timesteps), **kwargs)
val_loader = DataLoader(KittiMOTSDataset(args.dataset_root_path, 'val', args.timesteps), **kwargs)

# global var to store best validation accuracy across all epochs
best_mota = 0.0


# training function
def train(model, epoch):
    epoch_loss = list()
    correct = 0.
    total = 0.
    #motas = []
    model.train()
    for b_idx, (X, y) in enumerate(train_loader):
        if type(X) == type([]) or type(y) == type([]):
            continue
        if args.cuda:
            X, y = X.cuda(), y.cuda()
        # convert the data and targets into Variable and cuda form
        X, y = Variable(X, requires_grad=True), Variable(y)

        # train the network
        optimizer.zero_grad()

        # intialize graph and run first forward pass
        y_pred, feats, node_adj, edge_adj, labels, t_init, t_end = initialize_graph(X, y, mode='train', cuda=args.cuda)
        if y_pred is None:
            continue
        # compute the loss
        scores = model.forward(feats, node_adj, edge_adj)
        loss = F.nll_loss(scores, labels)
        scores = torch.exp(scores)
        # compute the accuracy
        pred = scores.data.max(1)[1]  # get the index of the max log-probability
        correct += float(pred.eq(labels.data).cpu().sum())
        total += float(labels.size()[0])
        # intialize graph and run first forward pass
        for t in range(t_init, t_end):
            # update graph for next timestep and run forward pass
            y_pred, feats, node_adj, edge_adj, labels = update_graph(feats, node_adj, labels, scores, y_pred, X, y, t, mode='train', cuda=args.cuda)
            scores = model.forward(feats, node_adj, edge_adj)
            # compute the loss
            loss += F.nll_loss(scores, labels)
            scores = torch.exp(scores)
            # compute the accuracy
            pred = scores.data.max(1)[1]  # get the index of the max log-probability
            correct += float(pred.eq(labels.data).cpu().sum())
            total += float(labels.size()[0])
            #y_pred, feats, node_adj, scores = prune_graph(feats, node_adj, scores, y_pred, t1, t2, threshold=0.5)
        #tracks = decode_tracks(node_adj, scores, y_pred)
        #acc = create_mot_accumulator(tracks, y)
        #if acc is not None:
        #    motas.append(calc_mot_metrics([acc])['mota'])

        epoch_loss.append(loss.item())
        loss.backward()
        optimizer.step()

        if b_idx % args.log_schedule == 0:
            print('Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.5f}'.format(
                epoch, (b_idx+1), len(train_loader.dataset),
                100. * (b_idx+1) / len(train_loader.dataset), loss.item()))
            with open(os.path.join(args.output_dir, "logs.txt"), "a") as f:
                f.write('Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.5f}\n'.format(
                epoch, (b_idx+1), len(train_loader.dataset),
                100. * (b_idx+1) / len(train_loader.dataset), loss.item()))

    # now that the epoch is completed calculate statistics and store logs
    avg_loss = statistics.mean(epoch_loss)
    print("------------------------\nAverage loss for epoch = {:.2f}".format(avg_loss))
    with open(os.path.join(args.output_dir, "logs.txt"), "a") as f:
        f.write("\n------------------------\nAverage loss for epoch = {:.2f}\n".format(avg_loss))

    
    train_accuracy = 100.0*correct/total
    #train_mota = 100.0*statistics.mean(motas)
    print("Accuracy for epoch = {:.2f}%\n------------------------".format(train_accuracy))
    #print("MOTA for epoch = {:.2f}%\n------------------------".format(train_mota))
    with open(os.path.join(args.output_dir, "logs.txt"), "a") as f:
        f.write("Accuracy for epoch = {:.2f}%\n------------------------\n".format(train_accuracy))
        #f.write("MOTA for epoch = {:.2f}%\n------------------------\n".format(train_mota))

    return model, avg_loss, train_accuracy, None


# validation function
def val(model, epoch):
    global best_mota
    correct = 0.
    total = 0.
    motas = []
    model.eval()
    
    for b_idx, (X, y) in enumerate(val_loader):
        if type(X) == type([]) or type(y) == type([]):
            continue
        if args.cuda:
            X, y = X.cuda(), y.cuda()
        X, y = Variable(X), Variable(y)

        # intialize graph and run first forward pass
        y_pred, feats, node_adj, edge_adj, labels, t_init, t_end = initialize_graph(X, y, mode='train', cuda=args.cuda)
        if y_pred is None:
            continue
        # compute the classification scores
        scores = model.forward(feats, node_adj, edge_adj)
        scores = torch.exp(scores)
        # compute the accuracy
        pred = scores.data.max(1)[1]  # get the index of the max log-probability
        correct += float(pred.eq(labels.data).cpu().sum())
        total += float(labels.size()[0])
        # intialize graph and run first forward pass
        for t in range(t_init, t_end):
            # update graph for next timestep and run forward pass
            y_pred, feats, node_adj, edge_adj, labels = update_graph(feats, node_adj, labels, scores, y_pred, X, y, t, mode='train', cuda=args.cuda)
            scores = model.forward(feats, node_adj, edge_adj)
            scores = torch.exp(scores)
            # compute the accuracy
            pred = scores.data.max(1)[1]  # get the index of the max log-probability
            correct += float(pred.eq(labels.data).cpu().sum())
            total += float(labels.size()[0])
            #y_pred, feats, node_adj, scores = prune_graph(feats, node_adj, scores, y_pred, min(t-10, 0), t-1, threshold=0.5)
        tracks = decode_tracks(node_adj, scores, y_pred)
        acc = create_mot_accumulator(tracks, y)
        if acc is not None:
            motas.append(calc_mot_metrics([acc])['mota'])

        print('Done with sequence {} out {}...'.format(min(b_idx+1, len(val_loader.dataset)), len(val_loader.dataset)))

    print("------------------------\nPredicted {} out of {}".format(correct, total))
    val_accuracy = 100.0*correct/total
    val_mota = 100.0*statistics.mean(motas)
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
        torch.save(model.state_dict(), os.path.join(args.output_dir, 'fgcn_' + '%.4d' % (epoch,) + '.pth'))

    return val_accuracy, val_mota


if __name__ == '__main__':
    # get the model, load pretrained weights, and convert it into cuda for if necessary
    model = TrackMPNN(nfeat=1+4+64+10-10+64,
            nhid=args.hidden,
            nclass=2,
            dropout=False)
    if args.snapshot is not None:
        model.load_state_dict(torch.load(args.snapshot), strict=False)
    if args.cuda:
        model.cuda()
    print(model)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

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
    ax3.plot([], 'g', label='Train MOTA')
    ax3.plot([], 'b', label='Validation MOTA')
    ax3.legend()

    train_acc, val_acc, train_mota, val_mota = list(), list(), list(), list()

    for i in range(1, args.epochs+1):
        model, avg_loss, acc, mota = train(model, i)
        train_acc.append(acc)
        train_mota.append(mota)

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
        
        ax3.plot(train_mota, 'g', label='Train MOTA')
        ax3.plot(val_mota, 'b', label='Validation MOTA')
        fig3.savefig(os.path.join(args.output_dir, 'train_val_mota.jpg'))
    plt.close('all')
