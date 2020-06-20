import numpy as np
import motmetrics as mm

from utils.misc import vectorized_iou


def create_mot_accumulator(bbox_pred, bbox_gt, y_out, y_gt):
    """
    This is a function returns an accumulator with tracking predictions and GT stored in it

    bbox_pred [NUM_DETS_PRED, (cat_id, alpha, x1, y1, x2, y2, h, w, l, x, y, z, rotation_y, score)]: Predicted bboxes in a sequence
    bbox_gt [NUM_DETS_GT, (cat_id, alpha, x1, y1, x2, y2, h, w, l, x, y, z, rotation_y, score)]: GT bboxes in a sequence
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

        bboxo = bbox_gt[oids, 2:6]
        bboxo[:, 2:] = bboxo[:, 2:] - bboxo[:, :2]
        bboxh = bbox_pred[hids, 2:6]
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


def _compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.

    Code originally from https://github.com/rbgirshick/py-faster-rcnn.

    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def preprocess_bboxes_for_map(bbox_dict):
    """
    bbox_dict: {seq: (y_pred, bbox_pred)}
    y_pred: (frame, track_id)
    bbox_pred: (cat_id, alpha, x1, y1, x2, y2, h, w, l, x, y, z, rotation_y, score)
    """
    res = dict()
    unique_ids = np.array([], dtype=np.str)
    unique_classes = np.array([], dtype=np.str)
    for seq, data in bbox_dict.items():
        y_pred, bbox_pred = data
        ids = np.array([seq + '_' + str(fr) for fr in y_pred[:, 0]], dtype=np.str)
        unique_ids = np.unique(np.concatenate((unique_ids, ids), 0))
        labels = bbox_pred[:, 0].astype(np.str)
        unique_classes = np.unique(np.concatenate((unique_classes, labels), 0))
        scores = bbox_pred[:, 13]
        xmin = bbox_pred[:, 2]
        ymin = bbox_pred[:, 3]
        xmax = bbox_pred[:, 4]
        ymax = bbox_pred[:, 5]

        for i, idx in enumerate(ids):
            idx = ids[i]
            label = labels[i]
            if idx not in res:
                res[idx] = dict()
            if label not in res[idx]:
                res[idx][label] = []
            box = [xmin[i], ymin[i], xmax[i], ymax[i], scores[i]]
            res[idx][label].append(box)
    return res, unique_ids, unique_classes


def compute_map(bbox_pred_dict, bbox_gt_dict, iou_threshold=0.5, verbose=False):
    """
    bbox_pred_dict: {seq: (y_pred, bbox_pred)}
    bbox_pred_dict: {seq: (y_gt, bbox_gt)}
    param iou_threshold: IoU between boxes which count as 'match'. Default: 0.5
    param verbose: print detailed run info. Default: True

    return: mean Average Precision
    """
    all_detections, _, _ = preprocess_bboxes_for_map(bbox_pred_dict)
    all_annotations, unique_ids, unique_classes = preprocess_bboxes_for_map(bbox_gt_dict)
    if verbose:
        print('Detections length: {}'.format(len(all_detections)))
        print('Annotations length: {}'.format(len(all_annotations)))
        print('Unique classes: {}'.format(len(unique_classes)))

    average_precisions = {}
    for zz, label in enumerate(sorted(unique_classes)):

        # Negative class
        if str(label) == 'nan':
            continue

        false_positives = []
        true_positives = []
        scores = []
        num_annotations = 0.0

        for i in range(len(unique_ids)):
            detections = []
            annotations = []
            idx = unique_ids[i]
            if idx in all_detections:
                if label in all_detections[idx]:
                    detections = all_detections[idx][label]
            if idx in all_annotations:
                if label in all_annotations[idx]:
                    annotations = all_annotations[idx][label]

            if len(detections) == 0 and len(annotations) == 0:
                continue

            num_annotations += len(annotations)
            detected_annotations = []

            annotations = np.array(annotations, dtype=np.float64)
            for d in detections:
                scores.append(d[4])

                if len(annotations) == 0:
                    false_positives.append(1)
                    true_positives.append(0)
                    continue

                overlaps = vectorized_iou(np.expand_dims(np.array(d, dtype=np.float64), axis=0)[:, :4], annotations[:, :4])
                assigned_annotation = np.argmax(overlaps, axis=1)
                max_overlap = overlaps[0, assigned_annotation]

                if max_overlap >= iou_threshold and assigned_annotation not in detected_annotations:
                    false_positives.append(0)
                    true_positives.append(1)
                    detected_annotations.append(assigned_annotation)
                else:
                    false_positives.append(1)
                    true_positives.append(0)

        if num_annotations == 0:
            average_precisions[label] = 0, 0
            continue

        false_positives = np.array(false_positives)
        true_positives = np.array(true_positives)
        scores = np.array(scores)

        # sort by score
        indices = np.argsort(-scores)
        false_positives = false_positives[indices]
        true_positives = true_positives[indices]

        # compute false positives and true positives
        false_positives = np.cumsum(false_positives)
        true_positives = np.cumsum(true_positives)

        # compute recall and precision
        recall = true_positives / num_annotations
        precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)

        # compute average precision
        average_precision = _compute_ap(recall, precision)
        average_precisions[label] = average_precision, num_annotations
        if verbose:
            s1 = "{:30s} | {:.6f} | {:7d}".format(label, average_precision, int(num_annotations))
            print(s1)

    present_classes = 0
    precision = 0
    for label, (average_precision, num_annotations) in average_precisions.items():
        if num_annotations > 0:
            present_classes += 1
            precision += average_precision
    mean_ap = precision / present_classes
    if verbose:
        print('mAP: {:.6f}'.format(mean_ap))
    return mean_ap