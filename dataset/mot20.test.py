import sys
import argparse

sys.path.append('/home/mez/research_code/TrackMPNN')
sys.path.append('/home/mez/.local/share/virtualenvs/TrackMPNN-kwlXr-Yc/lib/python3.8/site-packages')

from dataset.mot20 import KittiMOTDataset


def test_loading_mot20_data(dataset_root_path):
    dataset = KittiMOTDataset(dataset_root_path=dataset_root_path,
                              split='train', cat='Pedestrian', detections='mot20')
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_root_path')
    args = parser.parse_args()
    test_loading_mot20_data(args.dataset_root_path)
    
    