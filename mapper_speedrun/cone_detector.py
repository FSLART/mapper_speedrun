import onnxruntime as ort
from onnxruntime import InferenceSession
import numpy as np
import cv2
from typing import List
import time
from .types import bbox_t
from .filtering import nms_iou

class ConeDetector:
    def __init__(self, model_path: str, confidence_thres: float = 0.65, iou_thres: float = 0.1, infer_size: int = 640):
        self.confidence_thres = confidence_thres
        self.iou_thres = iou_thres
        self.infer_size = infer_size
        self.original_size = None

        """
        self.providers = [
            ('CUDAExecutionProvider', {
                'device_id': 0,
                'arena_extend_strategy': 'kNextPowerOfTwo',
                'gpu_mem_limit': 2 * 1024 * 1024 * 1024,
                'cudnn_conv_algo_search': 'EXHAUSTIVE',
                'do_copy_in_default_stream': True,
            }),
            'CPUExecutionProvider',
        ]
        """
        ort.set_default_logger_severity(3)
        self.providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] # TODO: in Jetson, add TensorrtExecutionProvider  'CUDAExecutionProvider', 'CPUExecutionProvider'
        self.model: InferenceSession = InferenceSession(model_path, providers=self.providers)
        print(f"Model running on {self.model.get_providers()}")

        # do some warmup
        print("Warming up inference...", end="")
        warmup_image = np.zeros((self.infer_size, self.infer_size, 3), dtype=np.uint8)
        for i in range(10):
            self.predict(warmup_image)
        print("done")
        
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

        yolov8 = True

        prep_start = time.time()
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
        if yolov8:
            img_tensor = img_tensor.astype(np.float32) / 255.0
        else:
            img_tensor = img_tensor.astype(np.float32)
        prep_end = time.time()
        print(f"Pre-proc. time: {prep_end - prep_start}s {1.0 / (prep_end - prep_start)} Hz")

        # run inference
        try:
            inference_start = time.time()
            output = self.model.run(None, {'images': img_tensor})
            inference_end = time.time()
            print(f"Inference time: {inference_end - inference_start}s {1.0 / (inference_end - inference_start)} Hz")
        except Exception as e:
            print(f"Inference failed: {e}")

        if yolov8:  # Get the boxes and probabilities
            output = output[0]
            probs = output[:, 4:, :].transpose((0, 2, 1))
            boxes = output[:, :4, :].transpose((0, 2, 1))

            # Vectorized computation for corners
            x_center, y_center, w_box, h_box = boxes[0,:, 0], boxes[0,:, 1], boxes[0,:, 2], boxes[0,:, 3]
            x_min = x_center - w_box / 2
            y_min = y_center - h_box / 2
            x_max = x_center + w_box / 2
            y_max = y_center + h_box / 2

            boxes_corner = np.stack((x_min, y_min, x_max, y_max), axis=-1)[None].astype(np.float32)
            #test(probs, boxes_corner)
            end_for = time.time()

        else: #DAMO
            # get the boxes and probs
            probs = output[0]
            boxes_corner = output[1]
    

        # filter with nms-iou
        nms_start = time.time()
        bboxes = nms_iou(boxes_corner, probs, boxes_corner.shape[1], probs.shape[2],self.iou_thres,self.confidence_thres)
        nms_end = time.time()
        print(f"NMS time: {nms_end-nms_start}s {1 / (nms_end - nms_start)} Hz")

        # convert the bboxes to the original size and remove the ones outside
        bboxes = [self.infer_pixel_to_original(bbox) for bbox in bboxes if bbox.x + (bbox.w / 2) >= 0 and bbox.y + (bbox.h / 2) >= 0 and bbox.x + (bbox.w / 2) <= self.infer_size and bbox.y + (bbox.h / 2) <= self.infer_size]

        return bboxes

def test(probs, boxes):
    detections = []
    for i in range(boxes.shape[1]):  # Loop through all 8400 detections
        class_confidences = probs[0, i, :]  # Confidence scores for 5 classes
        max_confidence = np.max(class_confidences)  # Highest confidence value
        class_id = np.argmax(class_confidences)  # Class with highest confidence
        
        if max_confidence > 0.7:
            # Extract bounding box
            x_min, y_min, x_max, y_max = boxes[0, i, :]

            detections.append((x_min, y_min, x_max, y_max, max_confidence, class_id))

    # Create a black image (height=480, width=640, 3 color channels)
    height, width = 640, 640
    black_image = np.zeros((height, width, 3), dtype=np.uint8)

    # Draw rectangle on the black image
    # Color is green (B, G, R) and thickness is 2 pixels
    for bbox in detections:
        x1, y1, x2, y2 = bbox[:4]
        cv2.rectangle(black_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

    # Save the image
    output_path = "black_with_rectangle.jpg"
    cv2.imwrite(output_path, black_image)
    print(f"Output image saved to: {output_path}")
