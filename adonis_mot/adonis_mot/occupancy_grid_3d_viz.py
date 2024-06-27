import rclpy
from rclpy.node import Node
from ember_detection_interfaces.msg import EmberClusterArray
import numpy as np
import open3d as o3d
import cv2

from adonis_mot.utils import *
from adonis_mot.ocsort_tracker.giocsort import GIOCSort
from adonis_mot.ocsort_tracker.utils import *
from adonis_mot.ocsort_tracker.kalmantracker import ObjectTypes as KFTrackerObjectTypes
from adonis_mot.occupancy_grid.tracker_occ_grid import TrackerOccGrid
from adonis_mot.track_visualization.o3d_tracker_viz import Open3DTrackerVisualizer

class ClusterBoundingBoxViz(Node):
    def __init__(self):
        super().__init__('cluster_bbox_viz')
        self.sub = self.create_subscription(EmberClusterArray, '/ember_detection/ember_cluster_array', self.callback, 10)
        
        self.o3d_viz = Open3DTrackerVisualizer()

        self.cv2_track_window_name = "Track ID and Type"
        self.cv2_occ_grid_window_name = "Occupancy Grid"

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

        self.occupancy_grid = TrackerOccGrid(
            x_o = -50,
            y_o = -50,
            width=100,
            height=100,
            resolution=0.2,
            cur_occ_weight=0.2,
            fut_occ_weight=0.18,
            decay_rate=0.1
        )

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
        occ_grid = cv2.flip(occ_grid, 0)
        occ_grid = cv2.cvtColor(occ_grid, cv2.COLOR_GRAY2BGR)

        return occ_grid

    def callback(self, msg):
        
        self.o3d_viz.reset()

        ember_cluster_array = msg.clusters

        bboxes_array = np.array([get_2d_bbox_from_3d_bbox(np.array([[p.x, p.y, p.z] for p in ember_cluster.bounding_box.points])) for ember_cluster in ember_cluster_array])
        bboxes_to_track = np.array([get_track_struct_from_2d_bbox(bbox) for bbox in bboxes_array])
        centroids2d_array = np.array([[cluster.centroid.x, cluster.centroid.y] for cluster in ember_cluster_array])

        tracking_res = self.ocsort.update_v1(bboxes_to_track, centroids2d_array)
        tracking_ids = tracking_res[:, 4]

        MAX_TIME_SINCE_UPDATE = 30
        MIN_NUM_OBSERVATIONS = 10

        valid_in_scope_trks = np.array([trk for trk in self.ocsort.get_trackers() if trk.time_since_update < MAX_TIME_SINCE_UPDATE and trk.hits > MIN_NUM_OBSERVATIONS])
        valid_by_id_trks = np.array([trk for trk in self.ocsort.get_trackers() if trk.id in tracking_ids])
        valid_in_scope_id_trks = np.array([trk for trk in valid_by_id_trks if trk.time_since_update < MAX_TIME_SINCE_UPDATE and trk.hits > MIN_NUM_OBSERVATIONS])

        #self.clear_occ_grid()
        self.occupancy_grid.decay_occ_grid()
        self.occupancy_grid.update_occ_grid_poly(valid_in_scope_trks)

        self.o3d_viz.draw_ember_cluster_array(ember_cluster_array, tracking_res, bboxes_to_track)

        self.o3d_viz.draw_growth_bboxes(valid_in_scope_trks)
        self.o3d_viz.draw_mean_bbox(valid_by_id_trks)
        self.o3d_viz.draw_trk_velocity_direction(valid_by_id_trks)
        self.o3d_viz.draw_future_predictions(valid_by_id_trks)
        self.o3d_viz.draw_occ_grid_bounds(self.occupancy_grid)
        self.o3d_viz.render()
        
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
        image = self.o3d_viz.vis.capture_screen_float_buffer(do_render=True)
        image = np.asarray(image) * 255
        image = image.astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image

    def add_text_overlay(self, image, tracking_res, object_types=None):
        """
        Add text overlay to the image using OpenCV.
        """
        # Get the view and projection matrices
        view_control = self.o3d_viz.vis.get_view_control()
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
