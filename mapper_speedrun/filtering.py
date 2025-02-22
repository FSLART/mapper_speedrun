import numpy as np
from .types import bbox_t
from typing import List
import ctypes

# C structure for the bbox_t object
class c_bbox_t(ctypes.Structure):
    _fields_ = [("x1", ctypes.c_float),
                ("y1", ctypes.c_float),
                ("x2", ctypes.c_float),
                ("y2", ctypes.c_float),
                ("score", ctypes.c_float),
                ("class_id", ctypes.c_uint8)]
    
# import the shared library
core = ctypes.CDLL("/usr/lib/libnms.so")

# set the argument types
core.nms.argtypes = [
    ctypes.POINTER(ctypes.c_float), # boxes
    ctypes.POINTER(ctypes.c_float), # scores
    ctypes.c_uint32, # num_boxes
    ctypes.c_uint8, # num_classes
    ctypes.c_float, # confidence_threshold
    ctypes.c_float, # iou_threshold
    ctypes.POINTER(ctypes.POINTER(c_bbox_t)), # keep
    ctypes.POINTER(ctypes.c_uint32) # num_keep
]

# WARNING: C_BBOX_T USES X1, Y1, X2, Y2. BBOX_T USES X, Y, W, H
def nms_iou(boxes: List[float], scores: List[float], num_boxes: int, num_classes: int, iou_threshold: float, score_threshold: float) -> List[bbox_t]:

    # create the boxes pointer
    boxes = np.ascontiguousarray(boxes, dtype=np.float32)
    boxes_ptr = boxes.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    # create the scores pointer
    scores = np.ascontiguousarray(scores, dtype=np.float32)
    scores_ptr = scores.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    # create the keep array
    keep_ptr = ctypes.POINTER(c_bbox_t)()
    keep_ptr_ref = ctypes.pointer(keep_ptr)
    num_keep = ctypes.c_uint32(0)

    # run the nms
    core.nms(boxes_ptr, scores_ptr, num_boxes, num_classes, score_threshold, iou_threshold, keep_ptr_ref, ctypes.byref(num_keep))

    # convert the output to bbox_t
    output = []
    for i in range(num_keep.value):
        bbox = keep_ptr[i]
        output.append(bbox_t(bbox.x1, bbox.y1, bbox.x2 - bbox.x1, bbox.y2 - bbox.y1, bbox.score, bbox.class_id))

    return output
 
