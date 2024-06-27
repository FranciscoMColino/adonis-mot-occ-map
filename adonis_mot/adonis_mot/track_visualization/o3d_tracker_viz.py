import open3d as o3d
import numpy as np

from adonis_mot.ocsort_tracker.kalmantracker import ObjectTypes as KFTrackerObjectTypes
from adonis_mot.ocsort_tracker.utils import convert_x_to_bbox
from adonis_mot.utils import *

class Open3DTrackerVisualizer:
    def __init__(self, name="Open3D Tracker Visualizer"):

        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(name, width=640, height=480)
        self.setup_visualizer()

        self.id_to_color = {
            0: (0.3, 0.3, 0.3),
        }
    
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

    def reset(self):
        self.vis.clear_geometries()
        self.vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5), reset_bounding_box=False)

    def render(self):
        self.vis.poll_events()
        self.vis.update_renderer()

    def draw_occ_grid_bounds(self, occupancy_grid):
        # draw the bounds of the occupancy grid in o3d
        x1, y1, x2, y2 = occupancy_grid.x_o, occupancy_grid.y_o, occupancy_grid.x_o + occupancy_grid.width, occupancy_grid.y_o + occupancy_grid.height
        z1, z2 = 0, 10

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
        point_cloud.paint_uniform_color([0, 0, 1])
        self.vis.add_geometry(point_cloud, reset_bounding_box=False)

        bbox = point_cloud.get_axis_aligned_bounding_box()
        bbox.color = [0, 0, 1]
        self.vis.add_geometry(bbox, reset_bounding_box=False)

    def draw_bbox_from_tracker(self, bbox, color):
        x1, y1, x2, y2 = bbox
        z1, z2 = 0, 2

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

    def draw_growth_bboxes(self, trackers):
        for trk in trackers:

            track_id = int(trk.id) + 1

            if track_id not in self.id_to_color:
                self.id_to_color[track_id] = np.random.rand(3)

            bbox = trk.get_growth_bbox()

            if bbox is None:
                continue

            self.draw_bbox_from_tracker(bbox, self.id_to_color[track_id])

    def draw_mean_bbox(self, trackers):
        for trk in trackers:
            track_id = int(trk.id) + 1

            if track_id not in self.id_to_color:
                self.id_to_color[track_id] = np.random.rand(3)

            bbox = trk.get_mean_bbox()

            if bbox is None:
                continue

            self.draw_bbox_from_tracker(bbox, self.id_to_color[track_id])

    def draw_trk_velocity_direction(self, trackers):
        for trk in trackers:
            track_id = int(trk.id) + 1

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

    def draw_future_predictions(self, trackers):

        for trk in trackers:
            track_id = int(trk.id) + 1

            if trk.object_type == KFTrackerObjectTypes.STATIC:
                continue

            if track_id not in self.id_to_color:
                self.id_to_color[track_id] = np.random.rand(3)

            color = self.id_to_color[track_id]
            # make color darker by 0.1 but positive
            color = np.clip(color - 0.3, 0, 1)
                
            x_next = trk.get_k_away_prediction(30)
            bbox = convert_x_to_bbox(x_next)[0]
            
            if bbox is None:
                continue

            self.draw_bbox_from_tracker(bbox, color)

    def draw_ember_cluster_array(self, ember_cluster_array, tracking_res, bboxes_to_track):
        """
            Draw the bounding boxes, point clouds and centroids
        """

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


