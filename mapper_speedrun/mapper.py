import rclpy
from rclpy.node import Node
import tf2_ros
from tf2_ros import transformations
from sensor_msgs.msg import Image, CameraInfo
import numpy as np
from geometry_msgs.msg import TransformStamped
from lart_msgs.msg import Cone, ConeArray
from .camera import Camera
from .cone_detector import ConeDetector
from .reconstruction import Reconstruction
from .types import bbox_t
from typing import List

class Mapper(Node):
    def __init__(self):
        super().__init__('mapper')
        self.create_timer(1.0, self.tf_callback)
        
        self.intrinsic = None
        self.extrinsic = None

        # TODO: create the parameters

        # create the cone detector
        self.detector = ConeDetector(model_path='damo_yolo.onnx')

        # the camera will only be created when the camera_info topic is received and the transform
        self.camera = None
        
        # the reconstruction will only be created after the camera is created
        self.reconstruction = None

        # create the tf listener (camera_link to base_link)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # create the color subscriber
        self.color_sub = self.create_subscription(Image, '/color', self.color_callback, 10)

        # create the depth subscriber
        self.depth_sub = self.create_subscription(Image, '/depth', self.depth_callback, 10)

        # create the camera info subscriber
        self.camera_info_sub = self.create_subscription(CameraInfo, '/camera_info', self.camera_info_callback, 10)

        # create the cone publisher
        self.cone_pub = self.create_publisher(ConeArray, '/cones', 10)

    def tf_callback(self):
        # lookup the transform from camera_link to base_link
        trans = self.tf_buffer.lookup_transform('base_link', 'camera_link', rclpy.time.Time())
        # get the extrinsic parameters
        translation = trans.transform.translation
        rotation = trans.transform.rotation
        # convert to a 4x4 matrix
        matrix = transformations.quaternion_matrix([rotation.x, rotation.y, rotation.z, rotation.w])
        matrix[0][3] = translation.x
        matrix[1][3] = translation.y
        matrix[2][3] = translation.z
        self.extrinsic = matrix
        # verify if the camera is already instantiated
        if self.camera is None and self.intrinsic is not None:
            self.camera = Camera(self.intrinsic, self.extrinsic)
            self.reconstruction = Reconstruction(self.camera)


    def color_callback(self, msg: Image):
        # convert the image to numpy array
        img = np.array(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
        # capture the color image
        self.camera.capture_color(img)
        # detect cones using the detector
        cones: List[bbox_t] = self.detector.predict(self.camera.last_color)
        # reconstruct the cones
        cone_array = ConeArray()
        for cone in cones:
            # get the xyd values
            xyd = self.reconstruction.pixelForBBox(cone)
            # get the 3D position of the cone
            pos = self.reconstruction.deprojectPixelToPoint(xyd)
            # convert to Cone message
            cone_msg = Cone()
            cone_msg.position.x = pos[0]
            cone_msg.position.y = pos[1]
            cone_msg.position.z = pos[2]
            cone_msg.class_type = cone.class_id
            cone_array.cones.append(cone_msg)
        # publish the cone array
        self.cone_pub.publish(cone_array)


    def depth_callback(self, msg: Image):
        # convert the image to numpy array
        img = np.array(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
        # capture the depth image
        self.camera.capture_depth(img)

    def camera_info_callback(self, msg: CameraInfo):
        # assign the intrinsic parameters
        self.intrinsic = msg.K
        self.intrinsic = np.reshape(self.intrinsic, (3, 3))
        # instantiate the camera and reconstruction
        if self.camera is None and self.extrinsic is not None:
            self.camera = Camera(self.intrinsic, self.extrinsic)
            self.reconstruction = Reconstruction(self.camera)

def main(args=None):
    rclpy.init(args=args)
    mapper = Mapper()
    rclpy.spin(mapper)
    mapper.destroy_node()
    rclpy.shutdown()
