import numpy as np
from scipy.spatial.distance import cdist
import torch
import motmetrics as mm


def create_mot_accumulator(tracks, y):
    """
    This is a function returns an accumulator with tracking predictions stored in it
    
    tracks list([N',]): A list of 1D arrays of detection IDs for each track.
    y [B, NUM_DETS, 2]: Array where each row is [ts, track_id]

    Returns: 
    A MOT accumulator object
    """
    y = y.squeeze(0).detach().cpu().numpy().astype('int64') # (NUM_DETS, 2)
    if np.all(y[:, 1] == -1):
        return None
    times = np.sort(y[:, 0])
    t_st = times[0]
    t_ed = times[-1]

    # initialize and load tracking results into MOT accumulator
    acc = mm.MOTAccumulator()

    for t in range(t_st, t_ed+1):
        oids = np.where(np.logical_and(y[:, 0] == t, y[:, 1] != -1))[0]
        otracks = y[oids, 1]
        oids = oids.astype('float32')
        otracks = otracks.astype('float32')

        hids = np.array([])
        htracks = np.array([])

        for i, track in enumerate(tracks):
            idx = np.where(y[track, 0] == t)[0]
            if idx.size == 0:
                continue
            elif idx.size == 1:
                htracks = np.append(htracks, i)
                hids = np.append(hids, track[int(idx)])
            else:
                assert False, "Multiple detections from same frame assigned to same track"

        dists = cdist(np.stack((oids, oids), 1), np.stack((hids, hids), 1))
        dists[dists > 0] = 1
        acc.update(otracks, htracks, dists, frameid=t)
        
    # make sure that desired metrics can be calculated
    try:
        mh = mm.metrics.create()
        summary = mh.compute(acc, metrics=mm.metrics.motchallenge_metrics, name='acc')
    except:
        return None

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
    # strsummary = mm.io.render_summary(summary, formatters=mh.formatters, namemap=mm.io.motchallenge_metric_names)
    return summary.to_dict('records')[-1]
