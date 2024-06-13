import rclpy
import rclpy.duration
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSDurabilityPolicy, QoSReliabilityPolicy
import tf2_ros
from sensor_msgs.msg import Image, CameraInfo
import numpy as np
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Int16
from scipy.spatial.transform import Rotation as R
from lart_msgs.msg import Cone, ConeArray
from .camera import Camera
from .cone_detector import ConeDetector
from .reconstruction import Reconstruction
from .types import bbox_t
from typing import List
from threading import Thread
import os
import copy

class Mapper(Node):
    def __init__(self):
        super().__init__('mapper')

        self.get_logger().info(f"Mapper node started in {os.getcwd()}")

        self.create_timer(1.0, self.tf_callback)
        
        self.intrinsic = None
        self.extrinsic = None

        # create the parameters
        self.declare_parameter('model_path', 'model/damo_yolo.onnx')
        self.declare_parameter('rgb_topic', '/zed/image_raw')
        self.declare_parameter('depth_topic', '/zed/depth/image_raw')
        self.declare_parameter('info_topic', '/zed/depth/camera_info')
        self.declare_parameter('cones_topic', '/cones')
        self.declare_parameter('cone_markers_topic', '/cone_markers')
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('camera_frame', 'zed_camera')

        # create the QoS profile
        qos_profile = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE
        )

        # create the cone detector
        model_path = self.get_parameter('model_path').get_parameter_value().string_value
        self.detector = ConeDetector(model_path=model_path)

        # the camera will only be created when the camera_info topic is received and the transform
        self.camera = None
        
        # the reconstruction will only be created after the camera is created
        self.reconstruction = None

        # create the tf listener (camera_link to base_link)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # create the color subscriber
        rgb_topic = self.get_parameter('rgb_topic').get_parameter_value().string_value
        self.color_sub = self.create_subscription(Image, rgb_topic, self.color_callback, qos_profile)

        # create the depth subscriber
        depth_topic = self.get_parameter('depth_topic').get_parameter_value().string_value
        self.depth_sub = self.create_subscription(Image, depth_topic, self.depth_callback, qos_profile)

        # create the camera info subscriber
        info_topic = self.get_parameter('info_topic').get_parameter_value().string_value
        self.camera_info_sub = self.create_subscription(CameraInfo, info_topic, self.camera_info_callback, qos_profile)

        # create the cone publisher
        cones_topic = self.get_parameter('cones_topic').get_parameter_value().string_value
        self.cone_pub = self.create_publisher(ConeArray, cones_topic, 10)

        # create the cone markers publisher
        cone_markers_topic = self.get_parameter('cone_markers_topic').get_parameter_value().string_value
        self.cone_markers_pub = self.create_publisher(MarkerArray, cone_markers_topic, 10)

        # flag to mark the worker as busy
        self.worker_busy: bool = False

        # create a list for markers currently in scene. used for removing markers
        self.marker_ids = []

        # counter of frames
        self.frame_counter = 0

    def tf_callback(self):
        # lookup the transform from camera_link to base_link
        base_frame = self.get_parameter('base_frame').get_parameter_value().string_value
        camera_frame = self.get_parameter('camera_frame').get_parameter_value().string_value
        try:
            trans = self.tf_buffer.lookup_transform(base_frame, camera_frame, rclpy.time.Time(), rclpy.duration.Duration(seconds=0.5))

            # get the extrinsic parameters
            translation = trans.transform.translation
            translation = np.array([translation.x, translation.y, translation.z])
            rotation = trans.transform.rotation
            rotation = np.array([rotation.x, rotation.y, rotation.z, rotation.w])
            extrinsic = R.from_quat(rotation) # convert quaternion to transform matrix
            extrinsic = extrinsic.as_matrix()
            translation = translation.reshape(-1, 1)
            extrinsic = np.append(extrinsic, translation, axis=1) # append the translation
            self.extrinsic = np.append(extrinsic, [[0, 0, 0, 1]], axis=0) # append the last row
            # invert the extrinsic matrix
            self.extrinsic = np.linalg.inv(self.extrinsic)

        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            # use the identity matrix as extrinsic parameters
            self.extrinsic = np.eye(4)
            self.get_logger().warning("Could not get the transform from base link to camera link. Using identity matrix!")


        if self.camera is None:
            if self.intrinsic is not None:
                self.camera = Camera(self.intrinsic, self.extrinsic)
                self.reconstruction = Reconstruction(self.camera)
            else:
                self.get_logger().warning("Received tf without known intrinsic parameters!")
        else:
            self.camera.extrinsic = self.extrinsic


    def color_callback(self, msg: Image):
        # convert the image to numpy array
        img = np.array(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
        # capture the color image
        if self.camera is None:
            return
        self.camera.capture_color(img, msg.header.stamp.sec + (msg.header.stamp.nanosec / 1e9))


    def depth_callback(self, msg: Image):
        img = msg.data.tobytes()
        # convert the image to numpy array
        img = np.frombuffer(img, dtype=np.float32).reshape(msg.height, msg.width)
        if self.camera is None:
            return
        # capture the depth image
        self.camera.capture_depth(img, msg.header.stamp.sec + (msg.header.stamp.nanosec / 1e9))

        def inference_task(last_color_img: np.ndarray, last_depth_img: np.ndarray, color_time: float, depth_time: float):

            # mark the worker as busy
            self.worker_busy = True
            
            # detect cones using the detector
            cones: List[bbox_t] = self.detector.predict(last_color_img)
            # reconstruct the cones
            cone_array = ConeArray()
            cone_marker_array = MarkerArray()

            # create an image to draw bounding boxes
            img_bb = copy.deepcopy(last_color_img)

            if last_depth_img is not None:
                d_img_bb = copy.deepcopy(last_depth_img)
                d_img_bb = (d_img_bb / 10.0) * 255 # 25 is the max depth range of the camera

            num_failed_cones = 0

            for i, cone in enumerate(cones):

                corner1 = (int(cone.x), int(cone.y))
                corner2 = (int(cone.x + cone.w), int(cone.y + cone.h))
                midpoint = (int(cone.x + (cone.w / 2)), int(cone.y + (cone.h / 2)))

                try:
                    # get the xyd values
                    xyd = self.reconstruction.pixelForBBox(cone, last_depth_img)
                except RuntimeError as e:
                    self.get_logger().warning(f"Pixel for bounding box runtime error: {str(e)}")
                    continue
                except ValueError as e:
                    self.get_logger().error(f"Pixel for bounding box value error: {str(e)}")
                    num_failed_cones += 1
                    continue

                try:
                    # get the 3D position of the cone
                    pos = self.reconstruction.deprojectPixelToPoint(xyd)
                except ValueError as e:
                    self.get_logger().error(f"Pixel to point value error: {str(e)}")
                    continue
                
                # convert to Cone message
                cone_msg = Cone()
                cone_msg.position.x = pos[0]
                cone_msg.position.y = pos[1]
                cone_msg.position.z = pos[2]
                cone_msg.class_type = Int16()
                cone_msg.class_type.data = cone.class_id
                cone_array.cones.append(cone_msg)

                # create the marker
                marker = Marker()
                marker.header.frame_id = self.get_parameter('base_frame').get_parameter_value().string_value
                marker.header.stamp = self.get_clock().now().to_msg()
                marker.id = self.frame_counter + cone.class_id + i
                marker.type = Marker.CYLINDER
                marker.action = Marker.ADD
                marker.pose.position.x = pos[0]
                marker.pose.position.y = pos[1]
                marker.pose.position.z = pos[2]
                marker.pose.orientation.w = 1.0
                marker.scale.x = 0.1
                marker.scale.y = 0.1
                marker.scale.z = 0.3
                marker.color.r = 1.0
                marker.color.g = 0.0
                marker.color.b = 0.0
                marker.color.a = 1.0
                cone_marker_array.markers.append(marker)

            # publish the cone array
            self.cone_pub.publish(cone_array)

            # publish the cone markers
            self.cone_markers_pub.publish(cone_marker_array)

            # remove old markers
            cone_marker_array = MarkerArray()
            for marker_id in self.marker_ids:
                marker = Marker()
                marker.header.frame_id = self.get_parameter('base_frame').get_parameter_value().string_value
                marker.header.stamp = self.get_clock().now().to_msg()
                marker.id = marker_id
                marker.action = Marker.DELETE
                cone_marker_array.markers.append(marker)
            self.cone_markers_pub.publish(cone_marker_array)

            # update the frame counter
            self.frame_counter += 1

            # mark the worker as free
            self.worker_busy = False

        # start the inference task if the worker is not busy
        if not self.worker_busy and self.camera.last_color is not None and self.camera.last_depth is not None:
            Thread(target=inference_task, args=(self.camera.last_color.copy(), self.camera.last_depth.copy(), self.camera.last_color_stamp, self.camera.last_depth_stamp)).start()

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
