import rclpy
from rclpy.node import Node
import sensor_msgs_py.point_cloud2 as pc2
from ember_detection_interfaces.msg import EmberClusterArray
import numpy as np
import open3d as o3d
import cv2
import math

from .ocsort_tracker.ocsort import OCSort
from .ocsort_tracker.giocsort import GIOCSort
from .ocsort_tracker.utils import *
from .ocsort_tracker.kalmantracker import ObjectTypes as KFTrackerObjectTypes

from .ocsort_tracker.kalmanfilter import predict as kf_predict

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

class OccupancyGrid2DAdonis:
    def __init__(self, x_o, y_o, width, height, resolution):
        self.x_o = x_o
        self.y_o = y_o
        self.width = width
        self.height = height
        self.resolution = resolution
        self.grid = np.zeros((int(height / resolution), int(width / resolution)))

class ClusterBoundingBoxViz(Node):
    def __init__(self):
        super().__init__('cluster_bbox_viz')
        self.sub = self.create_subscription(EmberClusterArray, '/ember_detection/ember_cluster_array', self.callback, 10)
        
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window('Open3D', width=640, height=480)
        self.setup_visualizer()

        self.cv2_track_window_name = "Track ID and Type"
        self.cv2_occ_grid_window_name = "Occupancy Grid"

        self.id_to_color = {
            0: (0.3, 0.3, 0.3),
        }

        self.ocsort = GIOCSort(
            #det_thresh=0.5,
            inertia_iou_threshold=0.05,
            growth_iou_threshold=0.002,
            default_iou_threshold=0.02,
            ignore_t=30,
            delta_t=90,          
            min_hits=5,
            max_age=60,
            inertia=0.5,        # 0.8
            intertia_age_weight=0.3,
            growth_rate=0.15,
            growth_age_weight=1.2,
        )

        self.occupancy_grid = OccupancyGrid2DAdonis(
            x_o = -5,
            y_o = -20,
            width=30,
            height=40,
            resolution=0.2
        )

        self.future_pred_occ_weight = 0.4

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

    def draw_occ_grid_bounds(self):
        # draw the bounds of the occupancy grid in o3d
        x1, y1, x2, y2 = self.occupancy_grid.x_o, self.occupancy_grid.y_o, self.occupancy_grid.x_o + self.occupancy_grid.width, self.occupancy_grid.y_o + self.occupancy_grid.height
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

    def capture_occ_grid_image(self):
        # Draw the occupancy grid in opencv2 new window
        
        max_size = 640

        if self.occupancy_grid.width > self.occupancy_grid.height:
            occ_grid_width = max_size
            occ_grid_height = int(max_size * (self.occupancy_grid.height / self.occupancy_grid.width))
        else:
            occ_grid_height = max_size
            occ_grid_width = int(max_size * (self.occupancy_grid.width / self.occupancy_grid.height))

        occ_grid = self.occupancy_grid.grid
        occ_grid = (1-occ_grid) * 255
        occ_grid = occ_grid.astype(np.uint8)

        occ_grid = cv2.resize(occ_grid, (occ_grid_width, occ_grid_height), interpolation=cv2.INTER_NEAREST)
        
        # mirror on the x-axis
        occ_grid = cv2.flip(occ_grid, 0)

        occ_grid = cv2.cvtColor(occ_grid, cv2.COLOR_GRAY2BGR)

        # draw the grid lines
        #for i in range(0, occ_grid_width, int(occ_grid_width / (self.occupancy_grid.width / self.occupancy_grid.resolution))):
        #    cv2.line(occ_grid, (i, 0), (i, occ_grid_height), (255, 255, 255), 1)
        #for i in range(0, occ_grid_height, int(occ_grid_height / (self.occupancy_grid.height / self.occupancy_grid.resolution))):
        #    cv2.line(occ_grid, (0, i), (occ_grid_width, i), (255, 255, 255), 1)

        return occ_grid

            
    def clear_occ_grid(self):
        self.occupancy_grid.grid = np.zeros((int(self.occupancy_grid.height / self.occupancy_grid.resolution), int(self.occupancy_grid.width / self.occupancy_grid.resolution)))

    def convert_bbox_to_grid_coords(self, bbox, safe_margin=0):

        x1, y1, x2, y2 = bbox[:4]

        # add a safe margin
        x1 -= safe_margin
        y1 -= safe_margin
        x2 += safe_margin
        y2 += safe_margin

        x1 = max(x1, self.occupancy_grid.x_o)
        y1 = max(y1, self.occupancy_grid.y_o)
        x2 = min(x2, self.occupancy_grid.x_o + self.occupancy_grid.width)
        y2 = min(y2, self.occupancy_grid.y_o + self.occupancy_grid.height)

        x1 = math.floor((x1 - self.occupancy_grid.x_o) / self.occupancy_grid.resolution)
        y1 = math.floor((y1 - self.occupancy_grid.y_o) / self.occupancy_grid.resolution)
        x2 = math.ceil((x2 - self.occupancy_grid.x_o) / self.occupancy_grid.resolution)
        y2 = math.ceil((y2 - self.occupancy_grid.y_o) / self.occupancy_grid.resolution)

        return x1, y1, x2, y2
    
    def update_occ_grid_fast(self, trackers, safe_margin=0.1, k_ahead=30):

        MAX_TIME_SINCE_UPDATE = 60
        MIN_NUM_OBSERVATIONS = 10

        display_lines = True

        for trk in trackers:

            if trk.time_since_update > MAX_TIME_SINCE_UPDATE or len(trk.observations) < MIN_NUM_OBSERVATIONS:
                continue

            bbox = convert_x_to_bbox(trk.kf.x)[0]

            if bbox is None or np.any(np.isnan(bbox)):
                continue

            x1, y1, x2, y2 = self.convert_bbox_to_grid_coords(bbox, safe_margin=safe_margin)
            self.occupancy_grid.grid[y1:y2, x1:x2] = 1

            if trk.object_type == KFTrackerObjectTypes.DYNAMIC:
                future_bbox = convert_x_to_bbox(trk.get_k_away_prediction(k_ahead))[0]

                if future_bbox is not None and not np.any(np.isnan(future_bbox)):

                    future_x1, future_y1, future_x2, future_y2 = self.convert_bbox_to_grid_coords(future_bbox, safe_margin=safe_margin)

                    self.occupancy_grid.grid[future_y1:future_y2, future_x1:future_x2] = 0.5

                    center_cur = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
                    center_future = np.array([(future_x1 + future_x2) / 2, (future_y1 + future_y2) / 2])

                    radius_cur = np.sqrt((center_cur[0] - x1) ** 2 + (center_cur[1] - y1) ** 2)
                    radius_future = np.sqrt((center_future[0] - future_x1) ** 2 + (center_future[1] - future_y1) ** 2)

                    vector_cur_to_future = center_future - center_cur
                    vector_cur_to_future /= np.linalg.norm(vector_cur_to_future)

                    vector_perpendicular = np.array([vector_cur_to_future[1], -vector_cur_to_future[0]])

                    corner_1 = center_cur + vector_perpendicular * radius_cur - vector_cur_to_future * radius_cur
                    corner_2 = center_cur - vector_perpendicular * radius_cur - vector_cur_to_future * radius_cur
                    corner_3 = center_future + vector_perpendicular * radius_future + vector_cur_to_future * radius_future
                    corner_4 = center_future - vector_perpendicular * radius_future + vector_cur_to_future * radius_future

                    if np.any(np.isnan(corner_1)) or np.any(np.isnan(corner_2)) or np.any(np.isnan(corner_3)) or np.any(np.isnan(corner_4)):
                        continue

                    if display_lines:
                        # Draw the line between the centers
                        for i in range(0, 100):
                            t = i / 100
                            x = int(center_cur[0] + t * (center_future[0] - center_cur[0]))
                            y = int(center_cur[1] + t * (center_future[1] - center_cur[1]))
                            self.occupancy_grid.grid[y, x] = 1

                        # Draw the line between the corners
                        for i in range(0, 100):
                            t = i / 100
                            x = int(corner_1[0] + t * (corner_3[0] - corner_1[0]))
                            y = int(corner_1[1] + t * (corner_3[1] - corner_1[1]))
                            self.occupancy_grid.grid[y, x] = 1

                            x = int(corner_2[0] + t * (corner_4[0] - corner_2[0]))
                            y = int(corner_2[1] + t * (corner_4[1] - corner_2[1]))
                            self.occupancy_grid.grid[y, x] = 1


    def update_occ_grid(self, trackers, safe_margin=0.1, k_ahead=30, radial_margin=2):

        MAX_TIME_SINCE_UPDATE = 60
        MIN_NUM_OBSERVATIONS = 10

        for trk in trackers:

            if trk.time_since_update > MAX_TIME_SINCE_UPDATE or len(trk.observations) < MIN_NUM_OBSERVATIONS:
                continue
                
            bbox = convert_x_to_bbox(trk.kf.x)[0]

            # check if the bbox is valid
            if bbox is None or np.any(np.isnan(bbox)):
                continue

            # Update current bbox
            x1, y1, x2, y2 = self.convert_bbox_to_grid_coords(bbox, safe_margin=safe_margin)
            self.occupancy_grid.grid[y1:y2, x1:x2] = 1

            if trk.object_type == KFTrackerObjectTypes.DYNAMIC:
                future_bbox = convert_x_to_bbox(trk.get_k_away_prediction(k_ahead))[0]

                if future_bbox is not None and not np.any(np.isnan(future_bbox)):
                    future_x1, future_y1, future_x2, future_y2 = self.convert_bbox_to_grid_coords(future_bbox, safe_margin=safe_margin)

                    # Get the centers of the current and future bounding boxes
                    center_x1 = (x1 + x2) / 2
                    center_y1 = (y1 + y2) / 2
                    center_x2 = (future_x1 + future_x2) / 2
                    center_y2 = (future_y1 + future_y2) / 2

                    dist_cur_to_future = np.sqrt((center_x2 - center_x1) ** 2 + (center_y2 - center_y1) ** 2)

                    # Interpolate cells between the current bbox and the future bbox
                    for i in range(min(x1, future_x1), max(x2, future_x2)):
                        for j in range(min(y1, future_y1), max(y2, future_y2)):
                            dist_to_current = np.sqrt((i - center_x1) ** 2 + (j - center_y1) ** 2)
                            dist_to_future = np.sqrt((i - center_x2) ** 2 + (j - center_y2) ** 2)
                            total_dist = (dist_to_current + dist_to_future)
                            if total_dist != 0:
                                value = (dist_to_future / total_dist) * 1 + (dist_to_current / total_dist) * self.future_pred_occ_weight
                            else:
                                value = 0.5 * (1 + self.future_pred_occ_weight)  # Equal weight if total distance is zero

                            # Apply radial margin with quadratic decay
                            if radial_margin > 0:
                                for dx in range(-radial_margin, radial_margin + 1):
                                    for dy in range(-radial_margin, radial_margin + 1):
                                        if 0 <= i + dx < self.occupancy_grid.grid.shape[1] and 0 <= j + dy < self.occupancy_grid.grid.shape[0]:
                                            dist_from_center = np.sqrt(dx ** 2 + dy ** 2)
                                            if dist_from_center <= radial_margin:
                                                decay_factor = (1 - (dist_from_center / radial_margin)) ** 4
                                                dist_diff_factor = (dist_cur_to_future / total_dist) ** 4
                                                self.occupancy_grid.grid[j + dy, i + dx] += value * decay_factor * dist_diff_factor
                                                self.occupancy_grid.grid[j + dy, i + dx] = np.clip(self.occupancy_grid.grid[j + dy, i + dx], 0, 1)
                else:
                    # No future bbox found, apply radial margin around current bbox
                    if radial_margin > 0:
                        for i in range(x1, x2):
                            for j in range(y1, y2):
                                value = 0.5 * (1 + self.future_pred_occ_weight)  # Use the current bbox value

                                # Apply radial margin with quadratic decay
                                for dx in range(-radial_margin, radial_margin + 1):
                                    for dy in range(-radial_margin, radial_margin + 1):
                                        if 0 <= i + dx < self.occupancy_grid.grid.shape[1] and 0 <= j + dy < self.occupancy_grid.grid.shape[0]:
                                            dist_from_center = np.sqrt(dx ** 2 + dy ** 2)
                                            if dist_from_center <= radial_margin:
                                                decay_factor = (1 - (dist_from_center / radial_margin)) ** 2  # Quadratic decay
                                                self.occupancy_grid.grid[j + dy, i + dx] += value * decay_factor
                                                self.occupancy_grid.grid[j + dy, i + dx] = np.clip(self.occupancy_grid.grid[j + dy, i + dx], 0, 1)


    def draw_bbox_from_tracker(self, bbox, color):
        x1, y1, x2, y2 = bbox
        z1, z2 = 0, 3

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

            self.draw_bbox_from_tracker(bbox, self.id_to_color[track_id])

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

            self.draw_bbox_from_tracker(bbox, self.id_to_color[track_id])

    def draw_trk_velocity_direction(self, trackers, track_ids):
        for trk in trackers:
            track_id = int(trk.id) + 1

            if track_ids is not None and track_id not in track_ids:
                continue

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

    def draw_future_predictions(self, trackers, track_ids):

        MAX_TIME_SINCE_UPDATE = 60
        MIN_NUM_OBSERVATIONS = 5

        for trk in trackers:
            track_id = int(trk.id) + 1

            if track_ids is not None and track_id not in track_ids:
                continue

            if trk.time_since_update > MAX_TIME_SINCE_UPDATE or len(trk.observations) < MIN_NUM_OBSERVATIONS:
                continue

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

    def callback(self, msg):
        self.vis.clear_geometries()
        self.vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5), reset_bounding_box=False)

        ember_cluster_array = msg.clusters

        bboxes_array = np.array([get_2d_bbox_from_3d_bbox(np.array([[p.x, p.y, p.z] for p in ember_cluster.bounding_box.points])) for ember_cluster in ember_cluster_array])
        bboxes_to_track = np.array([get_track_struct_from_2d_bbox(bbox) for bbox in bboxes_array])
        centroids2d_array = np.array([[cluster.centroid.x, cluster.centroid.y] for cluster in ember_cluster_array])

        tracking_res = self.ocsort.update_v1(bboxes_to_track, centroids2d_array)
        tracking_ids = tracking_res[:, 4]

        self.clear_occ_grid()
        self.update_occ_grid_fast(self.ocsort.get_trackers(), safe_margin=0.1)

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
            object_type = KFTrackerObjectTypes.STATIC

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

        #self.draw_kf_predict(self.ocsort.get_trackers())
        #self.draw_mean_bbox(self.ocsort.get_trackers(), tracking_ids)
        #self.draw_trk_velocity_direction(self.ocsort.get_trackers(), tracking_ids)
        self.draw_future_predictions(self.ocsort.get_trackers(), None)
        self.draw_occ_grid_bounds()

        self.vis.poll_events()
        self.vis.update_renderer()

        
        objec_tracking_res_types = np.array([KFTrackerObjectTypes.STATIC] * len(tracking_res))
        for i, track in enumerate(tracking_res):
            track_id = int(track[4]-1)
            kalman_tracker = self.ocsort.get_tracker_by_id(track_id)
            if kalman_tracker is not None:
                objec_tracking_res_types[i] = kalman_tracker.object_type
            else:
                objec_tracking_res_types[i] = KFTrackerObjectTypes.INVALID


        image_trk = self.capture_image()# Capture the current render
        image_trk = self.add_text_overlay(image_trk, tracking_res, object_types=objec_tracking_res_types)
        image_occ_grid = self.capture_occ_grid_image()
        # Display the image with text overlay
        cv2.imshow(self.cv2_track_window_name, image_trk)
        cv2.imshow(self.cv2_occ_grid_window_name, image_occ_grid)
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

    def add_text_overlay(self, image, tracking_res, object_types=None):
        """
        Add text overlay to the image using OpenCV.
        """
        # Get the view and projection matrices
        view_control = self.vis.get_view_control()
        camera_parameters = view_control.convert_to_pinhole_camera_parameters()
        intrinsic = camera_parameters.intrinsic.intrinsic_matrix
        extrinsic = camera_parameters.extrinsic

        for i, track in enumerate(tracking_res):
            bbox = track[:4]
            track_id = int(track[4])
            text_id = f"ID: {track_id}"
            text_type = ""

            if object_types is not None:
                if object_types[i] == KFTrackerObjectTypes.STATIC:
                    text_type = "STATIC"
                elif object_types[i] == KFTrackerObjectTypes.DYNAMIC:
                    text_type = "DYNAMIC"
                elif object_types[i] == KFTrackerObjectTypes.INVALID:
                    text_type = "INVALID"
                else:
                    text_type = "UNKNOWN"

            # Calculate the farthest corner of the bounding box
            farthest_corner_3d = np.array([bbox[0], bbox[1], 0])  # Assuming z=0 for 2D bbox corners
            farthest_corner_2d = self.project_to_2d(farthest_corner_3d, intrinsic, extrinsic)

            # Overlay text on the image
            cv2.putText(image, text_id, (int(farthest_corner_2d[0]), int(farthest_corner_2d[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(image, text_type, (int(farthest_corner_2d[0]), int(farthest_corner_2d[1] + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_AA)

        return image


    def project_to_2d(self, point_3d, intrinsic, extrinsic):
        """
        Project a 3D point to 2D screen coordinates using the intrinsic and extrinsic camera parameters.
        """
        point_3d_homo = np.append(point_3d, 1) # Convert point to homogeneous coordinates
        point_cam = extrinsic @ point_3d_homo # Apply extrinsic matrix (transform point to camera coordinates)
        point_2d_homo = intrinsic @ point_cam[:3] # Apply intrinsic matrix (project point to 2D)
        point_2d = point_2d_homo[:2] / point_2d_homo[2] # Normalize the coordinates
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
