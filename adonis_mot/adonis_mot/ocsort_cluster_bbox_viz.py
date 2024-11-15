import rclpy
import rclpy
from rclpy.node import Node
import sensor_msgs_py.point_cloud2 as pc2
from ember_detection_interfaces.msg import EmberClusterArray
import numpy as np
import open3d as o3d
from .ocsort_tracker.ocsort import OCSort

def load_pointcloud_from_ros2_msg(msg):
    pc2_points = pc2.read_points_numpy(msg, field_names=("x", "y", "z"), skip_nans=True)
    pc2_points_64 = pc2_points.astype(np.float64)
    valid_idx = ~np.isinf(pc2_points_64).any(axis=1)
    return pc2_points_64[valid_idx]

def get_corners_from__8point_bbox(points):
    # returns the min and max corners of the bounding box
    corners = np.zeros((2, 3))
    corners[0] = np.min(points, axis=0)
    corners[1] = np.max(points, axis=0)
    return corners

def get_2d_bbox_from_3d_bbox(points):
    # returns the min and max corners of the bounding box
    corners = np.zeros((2, 2))
    corners[0] = np.min(points[:, :2], axis=0)
    corners[1] = np.max(points[:, :2], axis=0)
    return corners

def get_track_struct_from_2d_bbox(bbox):
    # returns the min and max corners of the bounding box
    track = np.zeros(5)
    track[0] = bbox[0][0]
    track[1] = bbox[0][1]
    track[2] = bbox[1][0]
    track[3] = bbox[1][1]
    track[4] = 0.9 # dummy confidence score
    return track

class ClusterBoundingBoxViz(Node):
    def __init__(self):
        super().__init__('cluster_bbox_viz')
        self.sub = self.create_subscription(EmberClusterArray, '/ember_detection/ember_cluster_array', self.callback, 10)
        
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window('Open3D', width=640, height=480)
        self.setup_visualizer()

        self.id_to_color = {
            0: (0.3, 0.3, 0.3),
        }

        self.ocsort = OCSort(
            det_thresh=0.5, 
            iou_threshold=0.1,
            delta_t=15,
            max_age=150,
            inertia=0.2,
            )

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

    def callback(self, msg):
        self.vis.clear_geometries()
        self.vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5), reset_bounding_box=False)

        ember_cluster_array = msg.clusters

        bboxes_array = np.array([get_2d_bbox_from_3d_bbox(np.array([[p.x, p.y, p.z] for p in ember_cluster.bounding_box.points])) for ember_cluster in ember_cluster_array])
        bboxes_to_track = np.array([get_track_struct_from_2d_bbox(bbox) for bbox in bboxes_array])

        mock_img_info = np.array([640, 480])
        mock_img_size = np.array([640, 480])

        tracking_res = self.ocsort.update(bboxes_to_track, mock_img_info, mock_img_size)



        for i in range(len(ember_cluster_array)):
            ember_cluster = ember_cluster_array[i]
            ember_bbox = ember_cluster.bounding_box
            ember_pc2 = ember_cluster.point_cloud

            # track id is last element in the tracking res, find using the bbox
            # find track result where 4 first elements match the bbox
            track_id = 0

            for track in tracking_res:
                if np.allclose(track[:4], bboxes_to_track[i][:4]):
                    track_id = track[4]
                    break

            if track_id not in self.id_to_color:
                self.id_to_color[track_id] = np.random.rand(3)
            
            color = self.id_to_color[track_id]

            bbox_points = np.array([[p.x, p.y, p.z] for p in ember_bbox.points])

            box_pc = o3d.geometry.PointCloud()
            box_pc.points = o3d.utility.Vector3dVector(bbox_points)
            self.vis.add_geometry(box_pc, reset_bounding_box=False)

            bbox = box_pc.get_axis_aligned_bounding_box()
            bbox.color = color
            self.vis.add_geometry(bbox, reset_bounding_box=False)

            cluster_points = load_pointcloud_from_ros2_msg(ember_pc2)

            cluster_point_cloud = o3d.geometry.PointCloud()
            cluster_point_cloud.points = o3d.utility.Vector3dVector(cluster_points)
            cluster_point_cloud.paint_uniform_color(color)
            self.vis.add_geometry(cluster_point_cloud, reset_bounding_box=False)
            
            

        self.vis.poll_events()
        self.vis.update_renderer()


def main(args=None):
    rclpy.init(args=args)
    node = ClusterBoundingBoxViz()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
