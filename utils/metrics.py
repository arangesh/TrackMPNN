import numpy as np
import motmetrics as mm


def create_mot_accumulator(y_out, X, y):
    """
    This is a function returns an accumulator with tracking predictions and GT stored in it
    
    y_out [NUM_DETS, 2]: Array of outputs where each row is [ts, track_id]
    X [B, NUM_DETS, NUM_FEATS]: Features for all detections in a sequence
    y [B, NUM_DETS, 2]: Array where each row is [ts, track_id]

    Returns: 
    A MOT accumulator object
    """
    X = X.squeeze(0).detach().cpu().numpy().astype('float32') # (NUM_DETS, 2)
    y = y.squeeze(0).detach().cpu().numpy().astype('int64') # (NUM_DETS, 2)
    if np.all(y_out[:, 1] == -1) or np.all(y[:, 1] == -1):
        return None
    times = np.sort(y_out[:, 0])
    t_st = times[0]
    t_ed = times[-1]

    # initialize and load tracking results into MOT accumulator
    acc = mm.MOTAccumulator()

    for t in range(t_st, t_ed+1):
        oids = np.where(np.logical_and(y[:, 0] == t, y[:, 1] != -1))[0]
        otracks = y[oids, 1]
        otracks = otracks.astype('float32')

        hids = np.where(np.logical_and(y_out[:, 0] == t, y_out[:, 1] != -1))[0]
        htracks = y_out[hids, 1]
        htracks = htracks.astype('float32')

        bboxo = X[oids, 1:5]*np.array([1242, 375, 1242, 375]) + np.array([1242, 375, 1242, 375])/2
        bboxo[:, 2:] = bboxo[:, 2:] - bboxo[:, :2]
        bboxh = X[hids, 1:5]*np.array([1242, 375, 1242, 375]) + np.array([1242, 375, 1242, 375])/2
        bboxh[:, 2:] = bboxh[:, 2:] - bboxh[:, :2]
        dists = mm.distances.iou_matrix(bboxo, bboxh, max_iou=1.)

        acc.update(otracks, htracks, dists, frameid=t)

    return acc


def calc_mot_metrics(accs):
    """
    This is a function for computing MOT metrics over many accumulators
    
    accs: A list of MOT accumulators

    Returns: (formatted string presenting MOT challenge metrics)
    [idf1 idp idr recall precision num_unique_objects mostly_tracked partially_tracked 
    mostly_lost num_false_positives num_misses num_switches num_fragmentations mota motp]
    """
    # compute and display MOT metrics
    mh = mm.metrics.create()
    summary = mh.compute_many(accs, metrics=mm.metrics.motchallenge_metrics, names=[str(x) for x in range(len(accs))], generate_overall=True)

    return summary.to_dict('records')[-1]
