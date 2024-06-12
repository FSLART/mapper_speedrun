from onnxruntime import InferenceSession
import numpy as np
import cv2
from typing import List
from .types import bbox_t
from .filtering import nms_iou

class ConeDetector:
    def __init__(self, model_path: str, confidence_thres: float = 0.6, iou_thres: float = 0.5, infer_size: int = 640):
        self.model: InferenceSession = InferenceSession(model_path)
        self.confidence_thres = confidence_thres
        self.iou_thres = iou_thres
        self.infer_size = infer_size
        self.original_size = None
        
    def infer_pixel_to_original(self, bbox: bbox_t) -> bbox_t:
        """
        Convert the bbox from the infer size to the original size.
        
        Args:
            bbox (bbox_t): The bbox to convert.
        
        Returns:
            bbox_t: The converted bbox.
        """
        
        # get the scale factor
        scale_x = self.original_size[0] / self.infer_size
        scale_y = self.original_size[1] / self.infer_size
        
        # convert the bbox
        bbox = bbox_t(bbox.x * scale_x, bbox.y * scale_y, bbox.w * scale_x, bbox.h * scale_y, bbox.score, bbox.class_id)
        
        return bbox

    def predict(self, image) -> List[bbox_t]:

        # convert image from bgr to rgb
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # get the original size
        self.original_size = (image.shape[1], image.shape[0])

        # resize image to 640x640
        image = cv2.resize(image, (self.infer_size, self.infer_size), interpolation=cv2.INTER_LINEAR)

        # transpose the image channels
        img_tensor = image.transpose((2, 0, 1))

        # add the batch size channel
        img_tensor = img_tensor[None]

        # run inference
        output = self.model.run(None, {self.model.get_inputs()[0].name: img_tensor.astype(np.float32)})

        probs = output[0]
        boxes = output[1]

        # filter with nms-iou
        bboxes = nms_iou(boxes, probs, boxes.shape[1], probs.shape[2], self.iou_thres, self.confidence_thres)

        # convert the bboxes to the original size and remove the ones outside
        bboxes = [self.infer_pixel_to_original(bbox) for bbox in bboxes if bbox.x + (bbox.w / 2) >= 0 and bbox.y + (bbox.h / 2) >= 0 and bbox.x + (bbox.w / 2) <= self.infer_size and bbox.y + (bbox.h / 2) <= self.infer_size]

        return bboxes
