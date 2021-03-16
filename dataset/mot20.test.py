import sys

sys.path.append('/home/mez/research_code/TrackMPNN')
sys.path.append('/home/mez/.local/share/virtualenvs/TrackMPNN-kwlXr-Yc/lib/python3.8/site-packages')

from dataset.mot20 import KittiMOTDataset


def test_loading_mot20_data():
    dataset = KittiMOTDataset(dataset_root_path='/media/ssd01/dataset/MOT20_Kittie',
                              split='train', cat='Pedestrian', detections='mot20')
    


if __name__ == '__main__':
    test_loading_mot20_data()
    
    