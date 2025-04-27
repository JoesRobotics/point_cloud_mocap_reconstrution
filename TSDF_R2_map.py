import rclpy
from rclpy.node import Node
import open3d as o3d
import numpy as np
from sensor_msgs.msg import Image, CameraInfo
from mocap4r2_msgs.msg import RigidBodies
import tf_transformations
import cv2
from cv_bridge import CvBridge

class TSDFFusionNode(Node):
    def __init__(self):
        super().__init__('tsdf_fusion_node')

        self.bridge = CvBridge()
        self.camera_intrinsics = None
        self.latest_pose = None

        self.tsdf_volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=0.01, # 1 cm voxels
            sdf_trunc=0.04,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
        )

        self.create_subscription(CameraInfo, '/camera/depth/camera_info', self.camera_info_callback, 10)
        self.create_subscription(Image, '/camera/depth/image_raw', self.depth_image_callback, 10)
        self.create_subscription(RigidBodies, '/rigid_bodies', self.pose_callback, 10)

    def camera_info_callback(self, msg):
        if self.camera_intrinsics is None:
            self.camera_intrinsics = o3d.camera.PinholeCameraIntrinsic(
                width=msg.width,
                height=msg.height,
                fx=msg.k[0],
                fy=msg.k[4],
                cx=msg.k[2],
                cy=msg.k[5]
            )
            self.get_logger().info("Camera intrinsics set.")

    def pose_callback(self, msg):
        # Assume one rigid body tracked
        if msg.bodies:
            body = msg.bodies[0]
            self.latest_pose = self.rigid_body_to_matrix(body)

    def depth_image_callback(self, msg):
        if self.camera_intrinsics is None or self.latest_pose is None:
            return  # Wait until both camera info and pose are ready

        depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        depth_o3d = o3d.geometry.Image(depth_image)

        rgb_dummy = o3d.geometry.Image(np.zeros((depth_image.shape[0], depth_image.shape[1], 3), dtype=np.uint8))

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb_dummy, depth_o3d, convert_rgb_to_intensity=False
        )

        self.tsdf_volume.integrate(
            rgbd,
            self.camera_intrinsics,
            np.linalg.inv(self.latest_pose)  # Extrinsic: camera-to-world
        )

    def rigid_body_to_matrix(self, body):
        translation = np.array([body.pose.position.x, body.pose.position.y, body.pose.position.z])
        rotation = np.array([
            body.pose.orientation.x,
            body.pose.orientation.y,
            body.pose.orientation.z,
            body.pose.orientation.w
        ])
        matrix = tf_transformations.quaternion_matrix(rotation)
        matrix[0:3, 3] = translation
        return matrix

def main(args=None):
    rclpy.init(args=args)
    node = TSDFFusionNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print('Saving TSDF map...')
        mesh = node.tsdf_volume.extract_triangle_mesh()
        mesh.compute_vertex_normals()
        o3d.io.write_triangle_mesh("output_map.ply", mesh)
        print('Saved to output_map.ply')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
