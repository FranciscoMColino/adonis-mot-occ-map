import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
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
from adonis_mot.track_visualization.cv2_tracker_label_viz import TrackerLabelVisualizer
from adonis_mot.track_visualization.cv2_occ_grid_viz import OccupancyGridVisualizer

class ClusterBoundingBoxViz(Node):
    def __init__(self):
        super().__init__('cluster_bbox_viz')
        self.sub = self.create_subscription(EmberClusterArray, '/ember_detection/ember_cluster_array', self.callback, 10)
        self.pub = self.create_publisher(OccupancyGrid, '/tracking_occupancy/occupancy_grid', 10)
        
        self.o3d_viz = Open3DTrackerVisualizer()
        self.trk_label_viz = TrackerLabelVisualizer()
        self.occ_grid_viz = OccupancyGridVisualizer()

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

    def callback(self, msg):
        
        self.o3d_viz.reset()

        ember_cluster_array = msg.clusters

        bboxes_array = np.array([get_2d_bbox_from_3d_bbox(np.array([[p.x, p.y, p.z] for p in ember_cluster.bounding_box.points])) for ember_cluster in ember_cluster_array])
        bboxes_to_track = np.array([get_track_struct_from_2d_bbox(bbox) for bbox in bboxes_array])
        centroids2d_array = np.array([[cluster.centroid.x, cluster.centroid.y] for cluster in ember_cluster_array])

        tracking_res = self.ocsort.update_v1(bboxes_to_track, centroids2d_array)
        tracking_ids = tracking_res[:, 4]

        MAX_TIME_SINCE_UPDATE = 60
        MIN_NUM_OBSERVATIONS = 10

        valid_in_scope_trks = np.array([trk for trk in self.ocsort.get_trackers() if trk.time_since_update < MAX_TIME_SINCE_UPDATE and trk.hits > MIN_NUM_OBSERVATIONS])
        valid_by_id_trks = np.array([trk for trk in self.ocsort.get_trackers() if trk.id in tracking_ids])
        valid_in_scope_id_trks = np.array([trk for trk in valid_by_id_trks if trk.time_since_update < MAX_TIME_SINCE_UPDATE and trk.hits > MIN_NUM_OBSERVATIONS])

        #self.clear_occ_grid()
        self.occupancy_grid.decay_occ_grid()
        self.occupancy_grid.update_occ_grid_poly(valid_in_scope_trks)

        header = msg.header
        occ_grid_msg = self.occupancy_grid.to_ros2_msg(header)
        self.pub.publish(occ_grid_msg)

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

        # Display the image with text overlay
        cv2.imshow(self.trk_label_viz.window_name, self.trk_label_viz.generate_frame(self.o3d_viz.vis, tracking_res, objec_tracking_res_types))
        cv2.imshow(self.occ_grid_viz.window_name, self.occ_grid_viz.generate_frame(self.occupancy_grid))
        cv2.waitKey(1)

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
