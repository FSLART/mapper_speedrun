import rclpy
from rclpy.node import Node
import tf2_ros
from sensor_msgs.msg import Image, CameraInfo
import numpy as np
from geometry_msgs.msg import TransformStamped
from scipy.spatial.transform import Rotation as R
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

        # create the parameters
        self.declare_parameter('model_path', 'damo_yolo.onnx')
        self.declare_parameter('rgb_topic', '/color')
        self.declare_parameter('depth_topic', '/depth')
        self.declare_parameter('info_topic', '/camera_info')
        self.declare_parameter('cones_topic', '/cones')
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('camera_frame', 'camera_link')

        # create the cone detector
        model_path = self.get_parameter('model_path').get_parameter_value().string_value
        self.detector = ConeDetector(model_path=model_path)

        # the camera will only be created when the camera_info topic is received and the transform
        self.camera = None
        
        # the reconstruction will only be created after the camera is created
        self.reconstruction = None

        # create the tf listener (camera_link to base_link)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # create the color subscriber
        rgb_topic = self.get_parameter('rgb_topic').get_parameter_value().string_value
        self.color_sub = self.create_subscription(Image, rgb_topic, self.color_callback, 10)

        # create the depth subscriber
        depth_topic = self.get_parameter('depth_topic').get_parameter_value().string_value
        self.depth_sub = self.create_subscription(Image, depth_topic, self.depth_callback, 10)

        # create the camera info subscriber
        info_topic = self.get_parameter('info_topic').get_parameter_value().string_value
        self.camera_info_sub = self.create_subscription(CameraInfo, info_topic, self.camera_info_callback, 10)

        # create the cone publisher
        cones_topic = self.get_parameter('cones_topic').get_parameter_value().string_value
        self.cone_pub = self.create_publisher(ConeArray, cones_topic, 10)

    def tf_callback(self):
        # lookup the transform from camera_link to base_link
        base_frame = self.get_parameter('base_frame').get_parameter_value().string_value
        camera_frame = self.get_parameter('camera_frame').get_parameter_value().string_value
        trans = self.tf_buffer.lookup_transform(base_frame, camera_frame, rclpy.time.Time())
        # get the extrinsic parameters
        translation = trans.transform.translation
        rotation = trans.transform.rotation
        extrinsic = R.from_quat(rotation) # convert quaternion to transform matrix
        extrinsic = np.append(extrinsic, translation, axis=0) # append the translation

        # verify if the camera is already instantiated
        if self.camera is None and self.intrinsic is not None:
            self.camera = Camera(self.intrinsic, self.extrinsic)
            self.reconstruction = Reconstruction(self.camera)


    def color_callback(self, msg: Image):
        # convert the image to numpy array
        img = np.array(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 4)
        # capture the color image
        if self.camera is None:
            return
        self.camera.capture_color(img)
        # detect cones using the detector
        cones: List[bbox_t] = self.detector.predict(self.camera.last_color)
        print("I'M HERE")
        # reconstruct the cones
        cone_array = ConeArray()
        cone_marker_array = MarkerArray()
        for i, cone in enumerate(cones):

            try:
                # get the xyd values
                xyd = self.reconstruction.pixelForBBox(cone)
                # get the 3D position of the cone
                pos = self.reconstruction.deprojectPixelToPoint(xyd)
            except ValueError:
                continue
            
            # convert to Cone message
            cone_msg = Cone()
            cone_msg.position.x = pos[0]
            cone_msg.position.y = pos[1]
            cone_msg.position.z = pos[2]
            cone_msg.class_type = cone.class_id
            cone_array.cones.append(cone_msg)

            # create the marker
            marker = Marker()
            marker.header.frame_id = self.get_parameter('base_frame').get_parameter_value().string_value
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.id = cone.class_id + i
            marker.type = Marker.CYLINDER
            marker.action = Marker.ADD
            marker.pose.position.x = pos[0]
            marker.pose.position.y = pos[1]
            marker.pose.position.z = pos[2]
            marker.pose.orientation.w = 1
            marker.scale.x = 0.1
            marker.scale.y = 0.1
            marker.scale.z = 0.3
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            cone_marker_array.markers.append(marker)

        # publish the cone array
        self.cone_pub.publish(cone_array)

        # publish the cone markers
        self.cone_markers_pub.publish(cone_marker_array)


    def depth_callback(self, msg: Image):
        # convert the image to numpy array
        img = np.array(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 4)
        # capture the depth image
        self.camera.capture_depth(img)

    def camera_info_callback(self, msg: CameraInfo):
        # assign the intrinsic parameters
        self.intrinsic = msg.k
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
