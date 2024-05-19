import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import TransformStamped
from .camera import Camera
from .cone_detector import ConeDetector

class Mapper(Node):
    def __init__(self):
        super().__init__('mapper')
        self.create_timer(1.0, self.timer_callback)
        
        self.intrinsic = None
        self.extrinsic = None

        # TODO: create the parameters

        # TODO: create the cone detector

        # the camera will only be created when the camera_info topic is received
        self.camera = None
        
        # the reconstruction will only be created after the camera is created
        self.reconstruction = None

        # create the tf subscriber (camera_link to base_link)
        self.tf_sub = self.create_subscription(TransformStamped, '/tf', self.tf_callback, 10)

        # create the color subscriber
        self.color_sub = self.create_subscription(Image, '/color', self.color_callback, 10)

        # create the depth subscriber
        self.depth_sub = self.create_subscription(Image, '/depth', self.depth_callback, 10)

        # create the camera info subscriber
        self.camera_info_sub = self.create_subscription(CameraInfo, '/camera_info', self.camera_info_callback, 10)

    def tf_callback(self, msg: TransformStamped):
        # assign the extrinsic parameters
        self.extrinsic = msg.transform
        # instantiate the camera and reconstruction
        if self.camera is None and self.intrinsic is not None:
            self.camera = Camera(self.intrinsic, self.extrinsic)
            self.reconstruction = Reconstruction(self.camera)


    def color_callback(self, msg: Image):
        # TODO
        pass

    def depth_callback(self, msg: Image):
        # TODO
        pass

    def camera_info_callback(self, msg: CameraInfo):
        # TODO
        pass        


    def timer_callback(self):
        self.get_logger().info('Hello World')

def main(args=None):
    rclpy.init(args=args)
    mapper = Mapper()
    rclpy.spin(mapper)
    mapper.destroy_node()
    rclpy.shutdown()
