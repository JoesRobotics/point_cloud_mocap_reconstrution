import open3d as o3d
import numpy as np

def load_point_clouds_and_poses():
    """
    Assume you already have saved:
    - List of Open3D point clouds
    - Corresponding mocap poses (4x4 matrices)
    """
    point_clouds = []
    poses = []

    for i in range(0, 100):  # Replace with actual frame numbers
        pcd = o3d.io.read_point_cloud(f"frame_{i:04d}.ply")
        pose = np.loadtxt(f"pose_{i:04d}.txt")  # Assuming 4x4 pose matrices saved
        point_clouds.append(pcd)
        poses.append(pose)

    return point_clouds, poses

def pairwise_registration(source, target, init_transform):
    threshold = 0.02  # Max correspondence distance (adjust)
    reg = o3d.pipelines.registration.registration_icp(
        source, target, threshold, init_transform,
        o3d.pipelines.registration.TransformationEstimationPointToPlane()
    )
    return reg.transformation, reg.fitness

def full_registration(pcds, init_poses):
    pose_graph = o3d.pipelines.registration.PoseGraph()

    # Add first node with identity (fixed)
    pose_graph.nodes.append(
        o3d.pipelines.registration.PoseGraphNode(np.linalg.inv(init_poses[0]))
    )

    for source_id in range(len(pcds) - 1):
        target_id = source_id + 1

        source = pcds[source_id]
        target = pcds[target_id]

        init_transform = np.linalg.inv(init_poses[source_id]) @ init_poses[target_id]

        transformation_icp, fitness = pairwise_registration(source, target, init_transform)

        information = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
            source, target, 0.02, transformation_icp
        )

        pose_graph.edges.append(
            o3d.pipelines.registration.PoseGraphEdge(
                source_id, target_id, transformation_icp, information, uncertain=False
            )
        )

        # Optional: Add loop closures (for now, we skip)

    return pose_graph

def optimize_pose_graph(pose_graph):
    option = o3d.pipelines.registration.GlobalOptimizationOption(
        max_correspondence_distance=0.02,
        edge_prune_threshold=0.25,
        reference_node=0
    )

    o3d.pipelines.registration.global_optimization(
        pose_graph,
        o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
        o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
        option
    )

def integrate_and_save(pcds, pose_graph):
    pcd_combined = o3d.geometry.PointCloud()

    for pcd, node in zip(pcds, pose_graph.nodes):
        pcd_temp = pcd.transform(np.linalg.inv(node.pose))
        pcd_combined += pcd_temp

    pcd_combined_down = pcd_combined.voxel_down_sample(voxel_size=0.01)
    o3d.io.write_point_cloud("optimized_map.ply", pcd_combined_down)

def main():
    pcds, init_poses = load_point_clouds_and_poses()
    pose_graph = full_registration(pcds, init_poses)
    optimize_pose_graph(pose_graph)
    integrate_and_save(pcds, pose_graph)

if __name__ == '__main__':
    main()
