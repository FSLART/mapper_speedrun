import numpy as np

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
        self.last_depth = None

    def capture_color(self, img: np.ndarray):
        """
        Capture the color image.

        Args:
            img (np.ndarray): The color image captured by the camera.
        """
        # remove the alpha channel if it exists
        if img.shape[2] == 4:
            img = img[:, :, :3]
        self.last_color = img

    def capture_depth(self, img: np.ndarray):
        """
        Capture the depth image.

        Args:
            img (np.ndarray): The depth image captured by the camera.
        """
        self.last_depth = img        
