import numpy as np
from threading import Condition

class Camera:
    """
    Class to store information of the camera, such as the intrinsic and extrinsic parameters.
    """
    
    def __init__(self, intrinsic: np.ndarray, extrinsic: np.ndarray):
        """
        Initialize the Camera class.

        Args:
            intrinsic (np.ndarray): The intrinsic matrix of the camera.
            extrinsic (np.ndarray): The extrinsic matrix of the camera. In this case, the transform from the camera frame to the car frame.
        """

        # validate intrinsic matrix shape
        if intrinsic.shape != (3, 3):
            raise ValueError("Intrinsic matrix must be of shape (3, 3)")
        
        # validate extrinsic matrix shape
        if extrinsic.shape != (4, 4):
            raise ValueError("Extrinsic matrix must be of shape (4, 4)")

        # set the intrinsic and extrinsic parameters
        self.intrinsic = intrinsic
        self.extrinsic = extrinsic

        self.last_color = None
        self.new_color_available = False
        self.last_depth = None
        self.new_depth_available = False

        self.last_color_stamp = None
        self.last_depth_stamp = None

        # create a condition variable to notify when a new image is available
        self.color_cv = Condition()
        self.depth_cv = Condition()


    def capture_color(self, img: np.ndarray, timestamp: float):
        """
        Capture the color image.

        Args:
            img (np.ndarray): The color image captured by the camera.
        """
        # remove the alpha channel if it exists
        if img.shape[2] == 4:
            img = img[:, :, :3]

        # wait for the condition
        with self.color_cv:
            self.last_color = img
            self.last_color_stamp = timestamp
            self.new_color_available = True
            self.color_cv.notify() # notify the waiting thread

    def capture_depth(self, img: np.ndarray, timestamp: float):
        """
        Capture the depth image.

        Args:
            img (np.ndarray): The depth image captured by the camera.
        """

        # wait for the condition
        with self.depth_cv:
            self.last_depth = img
            self.last_depth_stamp = timestamp
            self.new_depth_available = True
            self.depth_cv.notify() # notify the waiting thread

    def get_last_color(self) -> np.ndarray:
        """
        Get the last color image captured by the camera.

        Returns:
            np.ndarray: The last color image captured by the camera.
        """
        with self.color_cv:
            self.color_cv.wait_for(lambda: self.last_color is not None and self.new_color_available)
            return self.last_color
    
    def get_last_depth(self) -> np.ndarray:
        """
        Get the last depth image captured by the camera.

        Returns:
            np.ndarray: The last depth image captured by the camera.
        """
        with self.depth_cv:
            self.depth_cv.wait_for(lambda: self.last_depth is not None and self.new_depth_available)
            return self.last_depth
