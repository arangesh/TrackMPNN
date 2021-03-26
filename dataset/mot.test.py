import sys
import argparse

from torch.utils.data import DataLoader

sys.path.append('/home/mez/research_code/TrackMPNN')
sys.path.append('/home/mez/.local/share/virtualenvs/TrackMPNN-kwlXr-Yc/lib/python3.8/site-packages')

from dataset.mot import MOTDataset 

kwargs_train = {'batch_size': 1, 'shuffle': True}

def test_loading_mot20_data(dataset_root_path):
    dataset = MOTDataset(dataset_root_path=dataset_root_path,
                              split='train', cat='Pedestrian', detections='mot20')

    train_loader = DataLoader(dataset, **kwargs_train)

    dataset = DataLoader(MOTDataset(dataset_root_path=dataset_root_path,
                              split='val', cat='Pedestrian', detections='mot20'), **kwargs_train)
        

    for b_idx, (X_seq, bbox_pred, _, loss_d) in enumerate(train_loader):
        print(X_seq)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_root_path')
    args = parser.parse_args()
    test_loading_mot20_data(args.dataset_root_path)
    
    