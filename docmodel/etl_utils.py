import numpy as np
import random


def normalize_bbox(bbox, width, height):
    coords = [
        int(1000 * (bbox[0] / width)),
        int(1000 * (bbox[1] / height)),
        int(1000 * (bbox[2] / width)),
        int(1000 * (bbox[3] / height)),
    ]
    return [max(0, min(i, 1000)) for i in coords]


def unnormalize_bbox(bbox, width, height):
    return [
        int(bbox[0] * width / 1000),
        int(bbox[1] * height / 1000),
        int(bbox[2] * width / 1000),
        int(bbox[3] * height / 1000),
    ]


def iou(A, B):
    """
    A: [left, top, right, bottom]
    B: [left, top, right, bottom]
       ┌───────────────┐
       │               │
       │               │
       │       A       │
       │        ┌──────┼──────┐
       │        │      │      │
       │        │ Int  │      │
       └────────┼──────┘      │
                │      B      │
                │             │
                └─────────────┘
    """
    # determine the (x, y)-coordinates of the intersection rectangle
    max_left = max(A[0], B[0])
    max_top = max(A[1], B[1])
    min_right = min(A[2], B[2])
    min_bottom = min(A[3], B[3])
    # compute the area of intersection rectangle
    intersection_width = max(0, min_right - max_left)
    intersection_height = max(0, min_bottom - max_top)
    intersection_area = intersection_width * intersection_height
    # compute the area of both the prediction and ground-truth
    # rectangles
    a_width = A[2] - A[0]
    a_height = A[3] - A[1]
    a_area = a_width * a_height
    b_width = B[2] - B[0]
    b_height = B[3] - B[1]
    b_area = b_width * b_height
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    union_area = float(a_area + b_area - intersection_area)
    iou = intersection_area / union_area
    return iou


def align_annotations(
    *,
    orig_words,
    orig_boxes,
    orig_labels,
    orig_images,
    new_words,
    new_boxes,
    iou_threshold=0.1,
):
    """
    Aligns annotations from two different OCR processes.
    """
    converted_labels = []
    for word, box in zip(new_words, new_boxes):
        # For each new box, we want to find the original box that is the best match
        # We define "best" as the orig box that has highest IoU with the new box
        match_quality = [iou(A=box, B=gt_box) for gt_box in orig_boxes]
        # Find the index of the highest IoU match
        best_match_idx = np.argmax(match_quality)
        best_match_iou = match_quality[best_match_idx]
        if best_match_iou <= iou_threshold:
            label = "O"
        else:
            label = orig_labels[best_match_idx]
        converted_labels.append(label)
    return new_words, new_boxes, converted_labels, orig_images


def use_reading_order(words, bboxes, labels=None, order="default"):
    if order == "default":
        result = [words, bboxes]
        if labels is not None:
            result.append(labels)
        return result, np.asarray(list(range(len(words))))
    elif order == "single_column":
        arr_bboxes = np.asarray(bboxes)
        # Use y coordinate as primary, x coordinate as secondary
        # We first discretize to prevent small variations in the y coordinate
        # from changing the line a token is assigned to.
        priority = (arr_bboxes[:, 1] // 10) * 1000 + arr_bboxes[:, 0]
        resorted_idxs = np.argsort(priority)
        result = [
            np.asarray(words)[resorted_idxs].tolist(),
            np.asarray(bboxes)[resorted_idxs].tolist(),
        ]
        if labels is not None:
            result.append(np.asarray(labels)[resorted_idxs].tolist())
        return result, resorted_idxs
    elif order == "random":
        idxs = list(range(len(words)))
        # [0, 1, 2, 3, ..., 512]
        random.shuffle(idxs)
        # [124, 15, 501, 5, ..., 176]
        # Get random sort order
        resorted_idxs = np.asarray(idxs)
        # ["this", "is", "a", ..., "end"]
        result = [
            np.asarray(words)[resorted_idxs].tolist(),
            np.asarray(bboxes)[resorted_idxs].tolist(),
        ]
        if labels is not None:
            result.append(np.asarray(labels)[resorted_idxs].tolist())
        return result, resorted_idxs
    elif order == "random_position":
        idxs = list(range(len(words)))
        random.shuffle(idxs)
        resorted_idxs = np.asarray(idxs)
        result = [
            words,
            np.asarray(bboxes)[resorted_idxs].tolist(),
        ]
        if labels is not None:
            result.append(labels)
        return result, resorted_idxs
