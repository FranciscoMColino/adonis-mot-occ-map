import rclpy
from rclpy.node import Node
import sensor_msgs_py.point_cloud2 as pc2
from ember_detection_interfaces.msg import EmberClusterArray
import numpy as np
import open3d as o3d
import cv2

from .ocsort_tracker.ocsort import OCSort
from .ocsort_tracker.giocsort import GIOCSort
from .ocsort_tracker.utils import convert_x_to_bbox

def load_pointcloud_from_ros2_msg(msg):
    pc2_points = pc2.read_points_numpy(msg, field_names=("x", "y", "z"), skip_nans=True)
    pc2_points_64 = pc2_points.astype(np.float64)
    valid_idx = ~np.isinf(pc2_points_64).any(axis=1)
    return pc2_points_64[valid_idx]

def get_corners_from_8point_bbox(points):
    # Returns the min and max corners of the bounding box
    corners = np.zeros((2, 3))
    corners[0] = np.min(points, axis=0)
    corners[1] = np.max(points, axis=0)
    return corners

def get_2d_bbox_from_3d_bbox(points):
    # Returns the min and max corners of the bounding box
    corners = np.zeros((2, 2))
    corners[0] = np.min(points[:, :2], axis=0)
    corners[1] = np.max(points[:, :2], axis=0)
    return corners

def get_track_struct_from_2d_bbox(bbox):
    # Returns the min and max corners of the bounding box
    track = np.zeros(5)
    track[0] = bbox[0][0]
    track[1] = bbox[0][1]
    track[2] = bbox[1][0]
    track[3] = bbox[1][1]
    track[4] = 1.0 # Dummy confidence score
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

        self.ocsort = GIOCSort(
            #det_thresh=0.5,
            inertia_iou_threshold=0.1,
            growth_iou_threshold=0.005,
            default_iou_threshold=0.02,
            ignore_t=30,
            delta_t=90,          
            min_hits=5,
            max_age=60,
            inertia=0.5,        # 0.8
            intertia_age_weight=0.3,
            growth_rate=0.1,
            growth_age_weight=0.9,
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

    def draw_kf_predict(self, trackers):

        MAX_TIME_SINCE_UPDATE = 60
        MIN_NUM_OBSERVATIONS = 5

        for trk in trackers:

            track_id = int(trk.id) + 1

            if trk.time_since_update > MAX_TIME_SINCE_UPDATE or len(trk.observations) < MIN_NUM_OBSERVATIONS:
                continue

            if track_id not in self.id_to_color:
                self.id_to_color[track_id] = np.random.rand(3)

            bbox = trk.get_growth_bbox()

            if bbox is None:
                continue

            x1, y1, x2, y2 = bbox
            z1, z2 = 0, 4   # TODO - Get the z values from the point cloud?

            color = self.id_to_color[track_id]

            # Draw the bounding box
            points = np.array([
                [x1, y1, z1],
                [x1, y1, z2],
                [x1, y2, z1],
                [x1, y2, z2],
                [x2, y1, z1],
                [x2, y1, z2],
                [x2, y2, z1],
                [x2, y2, z2],
            ])
            point_cloud = o3d.geometry.PointCloud()
            point_cloud.points = o3d.utility.Vector3dVector(points)
            point_cloud.paint_uniform_color(color)
            self.vis.add_geometry(point_cloud, reset_bounding_box=False)

            bbox_o3d = point_cloud.get_axis_aligned_bounding_box()
            bbox_o3d.color = color
            self.vis.add_geometry(bbox_o3d, reset_bounding_box=False)

    def draw_mean_bbox(self, trackers, track_ids):
        for trk in trackers:
            track_id = int(trk.id) + 1

            if track_ids is not None and track_id not in track_ids:
                continue

            if track_id not in self.id_to_color:
                self.id_to_color[track_id] = np.random.rand(3)

            bbox = trk.get_mean_bbox()

            if bbox is None:
                continue

            x1, y1, x2, y2 = bbox
            z1, z2 = 0, 3

            color = self.id_to_color[track_id]
            points = np.array([
                [x1, y1, z1],
                [x1, y1, z2],
                [x1, y2, z1],
                [x1, y2, z2],
                [x2, y1, z1],
                [x2, y1, z2],
                [x2, y2, z1],
                [x2, y2, z2],
            ])
            point_cloud = o3d.geometry.PointCloud()
            point_cloud.points = o3d.utility.Vector3dVector(points)
            point_cloud.paint_uniform_color(color)
            self.vis.add_geometry(point_cloud, reset_bounding_box=False)

            bbox_o3d = point_cloud.get_axis_aligned_bounding_box()
            bbox_o3d.color = color
            self.vis.add_geometry(bbox_o3d, reset_bounding_box=False)

    def draw_trk_velocity_direction(self, trackers, track_ids):
        for trk in trackers:
            track_id = int(trk.id) + 1

            if track_ids is not None and track_id not in track_ids:
                continue

            if track_id not in self.id_to_color:
                self.id_to_color[track_id] = np.random.rand(3)

            color = self.id_to_color[track_id]

            velocity = trk.velocity

            if velocity is None:
                continue

            # draw the velocity vector as a line

            x_center = (trk.last_observation[2] + trk.last_observation[0]) / 2
            y_center = (trk.last_observation[3] + trk.last_observation[1]) / 2
            z_center = 0

            x_end = x_center + velocity[1]
            y_end = y_center + velocity[0]
            z_end = 0

            points = np.array([
                [x_center, y_center, z_center],
                [x_end, y_end, z_end]
            ])

            
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(points)
            line_set.lines = o3d.utility.Vector2iVector([[0, 1]])
            line_set.colors = o3d.utility.Vector3dVector([(1, 0, 0)])
            self.vis.add_geometry(line_set, reset_bounding_box=False)

    def callback(self, msg):
        self.vis.clear_geometries()
        self.vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5), reset_bounding_box=False)

        ember_cluster_array = msg.clusters

        bboxes_array = np.array([get_2d_bbox_from_3d_bbox(np.array([[p.x, p.y, p.z] for p in ember_cluster.bounding_box.points])) for ember_cluster in ember_cluster_array])
        bboxes_to_track = np.array([get_track_struct_from_2d_bbox(bbox) for bbox in bboxes_array])
        centroids2d_array = np.array([[cluster.centroid.x, cluster.centroid.y] for cluster in ember_cluster_array])

        tracking_res = self.ocsort.update_v2(bboxes_to_track, centroids2d_array)
        tracking_ids = tracking_res[:, 4]
    
        self.draw_kf_predict(self.ocsort.get_trackers())
        self.draw_mean_bbox(self.ocsort.get_trackers(), tracking_ids)
        self.draw_trk_velocity_direction(self.ocsort.get_trackers(), tracking_ids)

        for i in range(len(ember_cluster_array)):
            ember_cluster = ember_cluster_array[i]
            ember_bbox = ember_cluster.bounding_box
            ember_pc2 = ember_cluster.point_cloud
            ember_centroid = ember_cluster.centroid

            # Track ID is the last element in the tracking result, find using the bbox
            track_id = 0

            for track in tracking_res:
                if np.allclose(track[:4], bboxes_to_track[i][:4]):
                    if track_id == 0:
                        track_id = int(track[4])
                    elif track_id != int(track[4]):
                        print(f"Found multiple tracks for the same bbox {bboxes_to_track[i]}")
                        print(f"Existing track {track_id}")
                        print(f"New track {int(track[4])}")
                        if int(track[4]) < track_id:
                            track_id = int(track[4])

            if track_id not in self.id_to_color:
                self.id_to_color[track_id] = np.random.rand(3)
            
            color = self.id_to_color[track_id]

            bbox_points = np.array([[p.x, p.y, p.z] for p in ember_bbox.points])

            box_pc = o3d.geometry.PointCloud()
            box_pc.points = o3d.utility.Vector3dVector(bbox_points)
            box_pc.paint_uniform_color(color)
            self.vis.add_geometry(box_pc, reset_bounding_box=False)

            bbox = box_pc.get_axis_aligned_bounding_box()
            bbox.color = color
            self.vis.add_geometry(bbox, reset_bounding_box=False)

            cluster_points = load_pointcloud_from_ros2_msg(ember_pc2)

            cluster_point_cloud = o3d.geometry.PointCloud()
            cluster_point_cloud.points = o3d.utility.Vector3dVector(cluster_points)
            cluster_point_cloud.paint_uniform_color(color)
            self.vis.add_geometry(cluster_point_cloud, reset_bounding_box=False)

            # Draw the centroid
            centroid = np.array([ember_centroid.x, ember_centroid.y, ember_centroid.z])
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
            sphere.translate(centroid)
            sphere.paint_uniform_color([0.7, 0, 1])
            self.vis.add_geometry(sphere, reset_bounding_box=False)
            
        self.vis.poll_events()
        self.vis.update_renderer()

        # Capture the current render
        image = self.capture_image()
        # Add text overlay using OpenCV
        image = self.add_text_overlay(image, tracking_res)
        # Display the image with text overlay
        cv2.imshow("Open3D with Text Overlay", image)
        cv2.waitKey(1)

    def capture_image(self):
        """
        Capture the current image from the Open3D visualizer.
        """
        image = self.vis.capture_screen_float_buffer(do_render=True)
        image = np.asarray(image) * 255
        image = image.astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image

    def add_text_overlay(self, image, tracking_res):
        """
        Add text overlay to the image using OpenCV.
        """
        # Get the view and projection matrices
        view_control = self.vis.get_view_control()
        camera_parameters = view_control.convert_to_pinhole_camera_parameters()
        intrinsic = camera_parameters.intrinsic.intrinsic_matrix
        extrinsic = camera_parameters.extrinsic

        for track in tracking_res:
            bbox = track[:4]
            track_id = int(track[4])
            text = f"ID: {track_id}"

            # Calculate the farthest corner of the bounding box
            farthest_corner_3d = np.array([bbox[0], bbox[1], 0])  # Assuming z=0 for 2D bbox corners
            farthest_corner_2d = self.project_to_2d(farthest_corner_3d, intrinsic, extrinsic)

            # Overlay text on the image
            cv2.putText(image, text, (int(farthest_corner_2d[0]), int(farthest_corner_2d[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_AA)
        
        return image


    def project_to_2d(self, point_3d, intrinsic, extrinsic):
        """
        Project a 3D point to 2D screen coordinates using the intrinsic and extrinsic camera parameters.
        """
        # Convert point to homogeneous coordinates
        point_3d_homo = np.append(point_3d, 1)
        
        # Apply extrinsic matrix (transform point to camera coordinates)
        point_cam = extrinsic @ point_3d_homo
        
        # Apply intrinsic matrix (project point to 2D)
        point_2d_homo = intrinsic @ point_cam[:3]
        
        # Normalize the coordinates
        point_2d = point_2d_homo[:2] / point_2d_homo[2]
        
        return point_2d


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
