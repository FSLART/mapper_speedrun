import rclpy
from rclpy.node import Node

class Mapper(Node):
    def __init__(self):
        super().__init__('mapper')
        self.create_timer(1.0, self.timer_callback)

    def timer_callback(self):
        self.get_logger().info('Hello World')

def main(args=None):
    rclpy.init(args=args)
    mapper = Mapper()
    rclpy.spin(mapper)
    mapper.destroy_node()
    rclpy.shutdown()
