import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
from message_filters import Subscriber, ApproximateTimeSynchronizer
from ember_detection_interfaces.msg import EmberClusterArray

import numpy as np
import open3d as o3d

def load_pointcloud_from_ros2_msg(msg):
    pc2_points = pc2.read_points_numpy(msg, field_names=("x", "y", "z"), skip_nans=True)
    pc2_points_64 = pc2_points.astype(np.float64)
    valid_idx = ~np.isinf(pc2_points_64).any(axis=1)
    return pc2_points_64[valid_idx]

class ClusterBboxPointCloudFusionViz(Node):
    def __init__(self):
        super().__init__('cluster_bbox_pc_fusion_viz')
        self.cluster_bbox_sub = Subscriber(self, EmberClusterArray, '/ember_detection/ember_cluster_array')
        self.pc_sub = Subscriber(self, PointCloud2, '/zed/zed_node/point_cloud/cloud_registered')
        
        self.ts = ApproximateTimeSynchronizer([self.cluster_bbox_sub, self.pc_sub], 10, 0.1)
        self.ts.registerCallback(self.callback)

        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window('Open3D', width=640, height=480)
        self.setup_visualizer()

    def setup_visualizer(self):
        # Add 8 points to initiate the visualizer's bounding box
        points = np.array([
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [0, 1, 1],
            [10, 0, 0],
            [10, 0, 1],
            [10, 1, 0],
            [10, 1, 1]
        ])

        points *= 4

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        self.vis.add_geometry(pcd, reset_bounding_box=True)

        view_control = self.vis.get_view_control()
        view_control.rotate(0, -525)
        view_control.rotate(500, 0)

        # points thinner and lines thicker
        self.vis.get_render_option().point_size = 2.0
        self.vis.get_render_option().line_width = 10.0

    def callback(self, cluster_bboxes_msg, pc):
        self.vis.clear_geometries()
        self.vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5), reset_bounding_box=False)

        pc_points = load_pointcloud_from_ros2_msg(pc)

        cluster_bboxes = cluster_bboxes_msg.clusters

        for ember_cluster in cluster_bboxes:
            ember_bbox = ember_cluster.bounding_box

            if ember_bbox.points_count.data != 8:
                print('Invalid bounding box with {} points'.format(ember_bbox.points_count))
                continue
            points_np = np.array([[p.x, p.y, p.z] for p in ember_bbox.points])
            box_pc = o3d.geometry.PointCloud()
            box_pc.points = o3d.utility.Vector3dVector(points_np)
            self.vis.add_geometry(box_pc, reset_bounding_box=False)

            bbox = box_pc.get_axis_aligned_bounding_box()
            bbox.color = (1, 0, 0)
            self.vis.add_geometry(bbox, reset_bounding_box=False)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc_points)
        self.vis.add_geometry(pcd, reset_bounding_box=False)
        
        self.vis.poll_events()
        self.vis.update_renderer()


def main(args=None):
    rclpy.init(args=args)
    node = ClusterBboxPointCloudFusionViz()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
