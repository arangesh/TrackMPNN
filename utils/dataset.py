import os


def get_tracking_data(dataset_path, split, timesteps):
    seqs = os.listdir(dataset_path)
    if split == 'train':
        seqs = seqs[:-2]
    elif split == 'val':
        seqs = seqs[-2:]
    elif split == 'trainval':
        pass
    else:
        assert False, 'Invalid dataset split!'
    num_frames = [len(os.listdir(os.path.join(dataset_path, x))) for x in seqs]

    # Load tracking dataset; each row is [seq_no, st_fr, ed_fr]
    dataset = []
    for i, seq in enumerate(seqs):
        for st_fr in range(0, num_frames[i], int(timesteps / 2)):
            dataset.append([seq, st_fr, min(st_fr + timesteps, num_frames[i])])

    return dataset