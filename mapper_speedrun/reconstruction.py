import numpy as np
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
        point_x: float = (pixel[0] - self.camera.intrinsic[0, 2]) / self.camera.intrinsic[0, 0]
        point_y: float = (pixel[1] - self.camera.intrinsic[1, 2]) / self.camera.intrinsic[1, 1]
        point_z: float = 1.0

        # assign the values
        point: np.ndarray = np.array([[point_x], [point_y], [point_z]])

        # normalize the point to the distance d (euclidean norm)
        point = point * pixel[2] / np.linalg.norm(point)

        # transform the point to the car frame
        point = np.dot(self.camera.extrinsic, point)

        return point
    
    def pixelForBBox(self, bbox: bbox_t) -> np.ndarray:
        """
        Get the midpoint pixel and depth for a given bounding box (x, y, d).

        Args:
            bbox (bbox_t): Bounding box
        
        Returns:
            np.ndarray: Point in format (x, y, d)
        """

        if bbox is None:
            raise ValueError("Bounding box cannot be None")
        
        if self.camera.last_depth is None:
            raise ValueError("No depth image received yet")

        # get the pixel coordinates
        x = int(bbox.x + float(bbox.h / 2))
        y = int(bbox.y + float(bbox.w / 2))

        # get the depth from the last depth image received
        d = self.camera.last_depth[x][y]

        return np.array([x, y, d])
