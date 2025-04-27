import rclpy
from rclpy.node import Node
import open3d as o3d
import numpy as np
from sensor_msgs.msg import PointCloud2
from mocap4r2_msgs.msg import RigidBodies
import sensor_msgs_py.point_cloud2 as pc2
import tf_transformations

class SLAMOptimizer(Node):
    def __init__(self):
        super().__init__('slam_optimizer_node')

        self.pcd_buffer = []     # List of Open3D PointClouds
        self.pose_buffer = []    # List of 4x4 numpy poses

        self.sub_pcd = self.create_subscription(PointCloud2, '/camera/depth/color/points', self.pcd_callback, 10)
        self.sub_pose = self.create_subscription(RigidBodies, '/rigid_bodies', self.pose_callback, 10)

        self.current_pose = None  # Latest mocap pose

    def pose_callback(self, msg):
        if msg.bodies:
            body = msg.bodies[0]  # Assume first body
            self.current_pose = self.rigid_body_to_matrix(body)

    def pcd_callback(self, msg):
        if self.current_pose is None:
            return  # Wait for pose

        cloud = self.convert_ros_to_o3d(msg)

        self.pcd_buffer.append(cloud)
        self.pose_buffer.append(self.current_pose.copy())

        self.get_logger().info(f"Buffered {len(self.pcd_buffer)} frames.")

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

    def convert_ros_to_o3d(self, ros_cloud):
        points = np.array(list(pc2.read_points(ros_cloud, field_names=("x", "y", "z"), skip_nans=True)))

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        return pcd

    def optimize(self):
        if len(self.pcd_buffer) < 2:
            self.get_logger().warning("Not enough frames to optimize.")
            return

        self.get_logger().info("Starting optimization...")

        pose_graph = o3d.pipelines.registration.PoseGraph()
        pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(np.linalg.inv(self.pose_buffer[0])))

        for source_id in range(len(self.pcd_buffer) - 1):
            target_id = source_id + 1

            source = self.pcd_buffer[source_id]
            target = self.pcd_buffer[target_id]

            init_transform = np.linalg.inv(self.pose_buffer[source_id]) @ self.pose_buffer[target_id]

            reg = o3d.pipelines.registration.registration_icp(
                source, target, 0.05, init_transform,
                o3d.pipelines.registration.TransformationEstimationPointToPlane()
            )

            information = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
                source, target, 0.05, reg.transformation
            )

            pose_graph.edges.append(
                o3d.pipelines.registration.PoseGraphEdge(
                    source_id, target_id, reg.transformation, information, uncertain=False
                )
            )

        # Optimize
        o3d.pipelines.registration.global_optimization(
            pose_graph,
            o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
            o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
            o3d.pipelines.registration.GlobalOptimizationOption(
                max_correspondence_distance=0.05,
                edge_prune_threshold=0.25,
                reference_node=0
            )
        )

        self.get_logger().info("Optimization done! Integrating map...")

        pcd_combined = o3d.geometry.PointCloud()

        for pcd, node in zip(self.pcd_buffer, pose_graph.nodes):
            pcd_temp = pcd.transform(np.linalg.inv(node.pose))
            pcd_combined += pcd_temp

        pcd_combined_down = pcd_combined.voxel_down_sample(voxel_size=0.01)
        o3d.io.write_point_cloud("optimized_map.ply", pcd_combined_down)

        self.get_logger().info("Saved optimized_map.ply!")

def main(args=None):
    rclpy.init(args=args)
    node = SLAMOptimizer()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("Keyboard interrupt. Optimizing...")
        node.optimize()
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
