function [track_ids, ignore] = getGTTrackIDs(detections, gt_boxes)

iou_thresh = 0.5;
ignore_thresh = 0.7;
track_ids = -1*ones(size(detections, 1), 1);
ignore = zeros(size(detections, 1), 1);
detections = [detections(:, 1), detections(:, 2), detections(:, 3)-detections(:, 1), detections(:, 4)-detections(:, 2)];

if isempty(gt_boxes)
    return
end
class_boxes = gt_boxes(gt_boxes(:, 1) == 0, :);
class_overlaps = bboxOverlapRatio(detections, class_boxes(:, 3:end), 'Union');
[max_class_overlaps, idx] = max(class_overlaps, [], 1);

% assert(length(idx) == length(unique(idx)));
for i = 1:length(idx)
    if max_class_overlaps(i) >= iou_thresh
        track_ids(idx(i)) = class_boxes(i, 2);
    end
end

ignore_boxes = gt_boxes(gt_boxes(:, 1) == -1, :);
ignore_overlaps = bboxOverlapRatio(detections, ignore_boxes(:, 3:end), 'Min');
max_ignore_overlaps = max(ignore_overlaps, [], 2);

for i = 1:length(max_ignore_overlaps)
    if (max_ignore_overlaps(i) >= ignore_thresh && track_ids(i) == -1)
        ignore(i) = 1;
    end
end

end