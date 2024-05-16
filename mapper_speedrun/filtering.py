import numpy as np
from .types import bbox_t
from typing import List

def nms_iou(boxes: List[float], scores: List[float], num_boxes: int, num_classes: int, iou_threshold: float, score_threshold: float) -> List[bbox_t]:

    # create a list of bbox_t objects
    bboxes = []
    for i in range(num_boxes):
        bbox = bbox_t(boxes[0, i, 0], boxes[0, i, 1], boxes[0, i, 2], boxes[0, i, 3], 0.0, 0)
        # find the class with the highest score
        max_score = 0.0
        class_id = 0
        for j in range(num_classes):
            if scores[0, i, j] > max_score:
                max_score = scores[0, i, j]
                class_id = j
        bbox = bbox._replace(score=max_score, class_id=class_id)
        bboxes.append(bbox)

    # sort the bboxes by score
    bboxes.sort(key=lambda x: x.score, reverse=True)

    # apply non-maximum suppression
    nms_bboxes = []
    for i in range(len(bboxes)):
        keep = True
        # check negative coordinates
        if bboxes[i].w < 0 or bboxes[i].h < 0 or bboxes[i].x < 0 or bboxes[i].y < 0:
            continue
        # check score threshold
        if bboxes[i].score < score_threshold:
            continue
        for j in range(len(nms_bboxes)):
            if keep:
                overlap = min(bboxes[i].x + bboxes[i].w, nms_bboxes[j].x + nms_bboxes[j].w) - max(bboxes[i].x, nms_bboxes[j].x)
                overlap *= min(bboxes[i].y + bboxes[i].h, nms_bboxes[j].y + nms_bboxes[j].h) - max(bboxes[i].y, nms_bboxes[j].y)
                iou = overlap / (bboxes[i].w * bboxes[i].h + nms_bboxes[j].w * nms_bboxes[j].h - overlap)
                if iou > iou_threshold:
                    keep = False
        if keep:
            nms_bboxes.append(bboxes[i])

    return nms_bboxes
