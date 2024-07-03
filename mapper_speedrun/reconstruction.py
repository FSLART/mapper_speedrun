import numpy as np
import math
from .camera import Camera
from .types import bbox_t

class Reconstruction:

    """
    Class to reconstruct 3D points from pixels.
    """
    def __init__(self, camera: Camera):
        """
        Initialize the Reconstruction class.

        Args:
            camera (Camera): The camera used for reconstruction.
        """
        self.camera = camera

    def deprojectPixelToPoint(self, pixel: np.ndarray) -> np.ndarray:
        """
        Deproject a pixel to a 3D point in the car frame.

        Args:
            pixel (np.ndarray): The pixel to deproject in format [x, y, d].
        
        Returns:
            np.ndarray: The 3D point in the car frame.
        """

        # pixel must be in format [x, y, d]
        if pixel.shape[0] != 3:
            raise ValueError("Pixel must be in format [x, y, d]")
        
        # get the point coordinates
        point_x: float = pixel[2]
        point_y: float = -(pixel[0] - self.camera.intrinsic[0, 2]) * point_x / self.camera.intrinsic[0, 0]
        point_z: float = -(pixel[1] - self.camera.intrinsic[1, 2]) * point_x / self.camera.intrinsic[1, 1]

        # assign the values
        point: np.ndarray = np.array([[point_x], [point_y], [point_z]])

        # add the homogeneous coordinate
        point = np.vstack((point, np.array([[1.0]])))

        # transform the point to the car frame
        point = self.camera.extrinsic @ point

        # remove the homogeneous coordinate
        point = point[:-1]

        # transpose
        point = point.T[0]

        return point
    
    def pixelForBBox(self, bbox: bbox_t, depth_img: np.ndarray) -> np.ndarray:
        """
        Get the midpoint pixel and depth for a given bounding box (x, y, d).

        Args:
            bbox (bbox_t): Bounding box
            depth_img (np.ndarray): Depth image
        
        Returns:
            np.ndarray: Point in format (x, y, d)
        """

        if bbox is None:
            raise ValueError("Bounding box cannot be None")
        
        if depth_img is None:
            raise RuntimeError("Depth image is None")

        # get the pixel coordinates
        x = int(bbox.x + float(bbox.w / 2))
        y = int(bbox.y + float(bbox.h / 2))

        # get the depth from the last depth image received
        # indexes are inverted because Python OpenCV is row-major
        d = depth_img[y][x]

        # verify if infinite depth
        if math.isinf(d) or math.isnan(d):
            raise ValueError(f"Infinite/invalid depth {d} detected at pixel ({x}, {y})")

        return np.array([x, y, d])
