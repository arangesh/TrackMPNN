import numpy as np
import motmetrics as mm


def create_mot_accumulator(bbox_pred, bbox_gt, y_out, y_gt):
    """
    This is a function returns an accumulator with tracking predictions and GT stored in it
    
    bbox_pred [NUM_DETS_PRED, (x1, y1, x2, y2, score)]: Predicted bboxes in a sequence
    bbox_gt [NUM_DETS_GT, (x1, y1, x2, y2, 1)]: GT bboxes in a sequence
    y_out [NUM_DETS_PRED, (frame, track_id)]: Predicted tracks where each row is [ts, track_id]
    y_gt [NUM_DETS_PRED, (frame, track_id)]: GT tracks where each row is [ts, track_id]

    Returns: 
    A MOT accumulator object
    """
    if np.all(y_gt[:, 1] == -1) or np.all(y_out[:, 1] == -1):
        return None
    times = np.sort(y_gt[:, 0])
    t_st = times[0]
    t_ed = times[-1]

    # initialize and load tracking results into MOT accumulator
    acc = mm.MOTAccumulator()

    for t in range(t_st, t_ed+1):
        oids = np.where(np.logical_and(y_gt[:, 0] == t, y_gt[:, 1] >= 0))[0]
        otracks = y_gt[oids, 1]
        otracks = otracks.astype('float32')

        hids = np.where(np.logical_and(y_out[:, 0] == t, y_out[:, 1] >= 0))[0]
        htracks = y_out[hids, 1]
        htracks = htracks.astype('float32')

        bboxo = bbox_gt[oids, :4]
        bboxo[:, 2:] = bboxo[:, 2:] - bboxo[:, :2]
        bboxh = bbox_pred[hids, :4]
        bboxh[:, 2:] = bboxh[:, 2:] - bboxh[:, :2]
        dists = mm.distances.iou_matrix(bboxo, bboxh, max_iou=0.5)

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
