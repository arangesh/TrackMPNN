import os
import glob
import sys
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import cv2
import colorsys


# adapted from https://github.com/xingyizhou/CenterTrack/blob/master/src/tools/vis_tracking_kitti.py
TRK_PATH = sys.argv[1]
DATASET_PATH = sys.argv[2]
SAVE_VIDEO = True
VIZ_VIDEO = False

cats = ['Pedestrian', 'Car', 'Cyclist']
cat_ids = {cat: i for i, cat in enumerate(cats)}


# adapted from https://github.com/VisualComputingInstitute/mots_tools/visualize_mots.py
def generate_colors(N=30):
  """
  Generate random colors.
  To get visually distinct colors, generate them in HSV space then
  convert to RGB.
  """
  brightness = 0.7
  hsv = [(i / N, 1, brightness) for i in range(N)]
  colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
  colors = list(map(lambda color: (int(255*color[0]), int(255*color[1]), int(255*color[2])), colors))
  random.shuffle(colors)
  return colors


def draw_bbox(img, bboxes, colors):
    for bbox in bboxes:
        cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), 
            colors[int(bbox[4])], 2, lineType=cv2.LINE_AA)
        ct = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
        txt = '{}'.format(int(bbox[4]))
        cv2.putText(img, txt, (int(ct[0]), int(ct[1])), cv2.FONT_HERSHEY_SIMPLEX, 
            0.5, colors[int(bbox[4])], thickness=1, lineType=cv2.LINE_AA)


if __name__ == '__main__':
    if SAVE_VIDEO:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video = cv2.VideoWriter('{}.avi'.format(TRK_PATH[:-4],), fourcc, 10.0, (1024, 375))

    preds = {}
    pred_file = open(TRK_PATH, 'r')
    preds[0] = defaultdict(list)
    track_id_list = []
    for line in pred_file:
        tmp = line[:-1].split(' ')
        frame_id = int(tmp[0])
        track_id = int(tmp[1])
        track_id_list.append(track_id)
        cat_id = cat_ids[tmp[2]]
        bbox = [float(tmp[6]), float(tmp[7]), float(tmp[8]), float(tmp[9])]
        score = float(tmp[17])
        preds[0][frame_id].append(bbox + [track_id, cat_id, score])

    images_path = os.path.join(DATASET_PATH, TRK_PATH[-8:-4])
    images = os.listdir(images_path)
    num_images = len([image for image in images if 'png' in image])
    colors = generate_colors(max(track_id_list) + 1)
    
    for frame_id in range(num_images):
        file_path = '{}/{:06d}.png'.format(images_path, frame_id)
        img = cv2.imread(file_path)
        img_pred = img.copy()
        draw_bbox(img_pred, preds[0][frame_id], colors)
        if VIZ_VIDEO:
            cv2.imshow('pred{}'.format(0), img_pred)
            cv2.waitKey()
        if SAVE_VIDEO:
            video.write(cv2.resize(img_pred, (1024, 375)))
        print('Frame %.4d/%.4d...' % (frame_id, num_images-1))
    if SAVE_VIDEO:
        video.release()
