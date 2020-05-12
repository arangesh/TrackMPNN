import os
import numpy as np


def get_tracking_data(dataset_path, split, timesteps):
    seqs = sorted(os.listdir(dataset_path))
    # seqs 13, 16 and 17 have very few or no cars at all
    if split == 'train':
        seqs = seqs[:-1]
        print(seqs)
    elif split == 'val':
        seqs = seqs[-1:]
        print(seqs)
    else:
        pass

    num_frames = [len(os.listdir(os.path.join(dataset_path, x))) for x in seqs]

    # Load tracking dataset; each row is [seq_no, st_fr, ed_fr]
    dataset = []
    if split == 'train':
        for i, seq in enumerate(seqs):
            for st_fr in range(0, num_frames[i], int(timesteps / 2)):
                dataset.append([seq, st_fr, min(st_fr + timesteps, num_frames[i])])
    else:
        for i, seq in enumerate(seqs):
            dataset.append([seq, 0, num_frames[i]])

    return dataset


def store_results_kitti(y_out, X, output_path):
    """
    This is a function that writes the result for the given sequence in KITTI format
    
    y_out [NUM_DETS, 2]: Array of outputs where each row is [ts, track_id]
    X [B, NUM_DETS, NUM_FEATS]: Features for all detections in a sequence
    output_path: Output file to write results in
    """
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    X = X.squeeze(0).detach().cpu().numpy().astype('float32') # (NUM_DETS, NUM_FEATS)
    times = np.sort(y_out[:, 0])
    t_st = times[0]
    t_ed = times[-1]

    with open(output_path, "w") as f:
        for t in range(t_st, t_ed+1):
            hids = np.where(np.logical_and(y_out[:, 0] == t, y_out[:, 1] != -1))[0]
            htracks = y_out[hids, 1]
            htracks = htracks.astype('int64')
            assert (htracks.size == np.unique(htracks).size), "Same track ID assigned to two detections from same timestep!"

            scores = X[hids, 0] * 0.2042707115 + 0.919026958912
            bboxs = X[hids, 1:5] * np.array([1242, 375, 1242, 375]) + np.array([1242, 375, 1242, 375]) / 2

            for i in range(scores.size):
                f.write("%d %d Car -1 -1 -10 %.2f %.2f %.2f %.2f -1 -1 -1 -1000 -1000 -1000 -10 %.2f\n" % (t, htracks[i], bboxs[i, 0], bboxs[i, 1], bboxs[i, 2], bboxs[i, 3], scores[i]))
