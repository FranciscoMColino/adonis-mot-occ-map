import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from ember_detection_interfaces.msg import EmberBoundingBox3DArray, EmberClusterArray
from sensor_msgs.msg import PointCloud2
import numpy as np
import open3d as o3d
import cv2
import multiprocessing
import signal
import time
import yaml
import argparse

from collections import deque
import threading

from adonis_mot.utils import *
from adonis_mot.ocsort_tracker.giocsort import GIOCSort
from adonis_mot.ocsort_tracker.utils import *
from adonis_mot.ocsort_tracker.kalmantracker import ObjectTypes as KFTrackerObjectTypes
from adonis_mot.occupancy_grid.tracker_occ_grid import TrackerOccGrid
from adonis_mot.track_visualization.o3d_tracker_viz import Open3DTrackerVisualizer
from adonis_mot.track_visualization.cv2_tracker_label_viz import TrackerLabelVisualizer
from adonis_mot.track_visualization.cv2_occ_grid_viz import OccupancyGridVisualizer

from adonis_mot.detection_recording.detection_recorder import Detection2dRecorder

def o3d_vis_worker(o3d_vis_input_queue):

    def signal_handler(sig, frame):
        print("SIGINT received")
        o3d_vis_input_queue.put(None)
        exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    o3d_viz = Open3DTrackerVisualizer()
    trk_label_viz = TrackerLabelVisualizer()
    occ_grid_viz = OccupancyGridVisualizer()

    while True:

        if o3d_vis_input_queue.empty():
            o3d_viz.render()
            time.sleep(1/120)
            continue

        msg = o3d_vis_input_queue.get()
        if msg is None:
            break

        valid_in_scope_trks = msg["valid_in_scope_trks"]
        occ_grid = msg["occ_grid"]
        cluster_array = msg["ember_cluster_array"]
        o3d_viz.reset()
        #o3d_viz.draw_growth_bboxes(valid_in_scope_trks)
        o3d_viz.draw_growth_areas(valid_in_scope_trks)
        o3d_viz.draw_mean_bbox(valid_in_scope_trks)
        o3d_viz.draw_trk_velocity_direction(valid_in_scope_trks)
        o3d_viz.draw_future_predictions(valid_in_scope_trks)
        o3d_viz.draw_occ_grid_bounds(occ_grid)
        if cluster_array is not None:
            o3d_viz.draw_ember_cluster_array_no_track(cluster_array)
        o3d_viz.render()

        tracking_res = msg["tracking_res"]
        objec_tracking_res_types = msg["objec_tracking_res_types"]

        cv2.imshow(trk_label_viz.window_name, trk_label_viz.generate_frame(o3d_viz.vis, tracking_res, objec_tracking_res_types))
        cv2.imshow(occ_grid_viz.window_name, occ_grid_viz.generate_frame(occ_grid))
        cv2.waitKey(1)

class DetectionRecordingConfig:
    def __init__(self, enable=False, save_dir=None):
        self.enable = enable
        self.save_dir = save_dir

class ClusterBoundingBoxViz(Node):
    def __init__(self, vis_input_queue, config_filename=None):
        super().__init__('cluster_bbox_viz')
        # Subscribers
        #self.detection_sub = self.create_subscription(EmberBoundingBox3DArray, '/ember_detection/ember_fusion_bboxes', self.callback, 10)
        self.detection_sub = self.create_subscription(EmberBoundingBox3DArray, '/ember_detection/ermis_bbox_array', self.callback, 10)
        self.pointcloud_sub = self.create_subscription(PointCloud2, '/zed/zed_node/point_cloud/cloud_registered', self.pointcloud_callback, 3)
        self.cluster_sub = self.create_subscription(EmberClusterArray, '/ember_detection/ember_cluster_array', self.cluster_callback, 3)
        self.pub = self.create_publisher(OccupancyGrid, '/tracking_occupancy/occupancy_grid', 10)

        self.ocsort = GIOCSort(
            #det_thresh=0.5,
            inertia_iou_threshold=0.40,
            growth_iou_threshold=0.001,
            default_iou_threshold=0.02,
            ignore_t=5,
            delta_t=25,          
            min_hits=10,
            max_age=60,
            inertia=0.2,        # 0.8
            intertia_age_weight=0.3,
            growth_rate=0.25,#0.175,
            growth_age_weight=0.01,
        )

        # ado1 config
        #self.ocsort = GIOCSort(
        #    #det_thresh=0.5,
        #    inertia_iou_threshold=0.80,
        #    growth_iou_threshold=0.10,
        #    default_iou_threshold=0.20,
        #    ignore_t=30,
        #    delta_t=90,          
        #    min_hits=10,
        #    max_age=60,
        #    inertia=0.5,        # 0.8
        #    intertia_age_weight=0.5,
        #    growth_rate=0.15,
        #    growth_age_weight=0.8,
        #)

        # ado2 config
        # self.ocsort = GIOCSort(
        #     #det_thresh=0.5,
        #     inertia_iou_threshold=0.05,
        #     growth_iou_threshold=0.002,
        #     default_iou_threshold=0.20,
        #     ignore_t=0,
        #     delta_t=25,          
        #     min_hits=10,
        #     max_age=60,
        #     inertia=0.8,        # 0.8
        #     intertia_age_weight=1.2,
        #     growth_rate=0.2,
        #     growth_age_weight=1.2,
        # )
        

        self.occupancy_grid = TrackerOccGrid(
            x_o = -50,
            y_o = -50,
            width=100,
            height=100,
            resolution=0.2,
            cur_occ_weight=0.2,
            fut_occ_weight=0.05,  #0.18,
            decay_rate=0.1
        )

        self.pointcloud_queue = deque(maxlen=30)
        self.cluster_queue = deque(maxlen=30)
        
        # Lock for thread-safe operations
        self.lock = threading.Lock()

        self.vis_input_queue = vis_input_queue

        self.load_config(config_filename)

        if self.detection_recording_config.enable:
            self.detection_recorder = Detection2dRecorder(self.detection_recording_config.save_dir)

    def load_config(self, config_filename):
        with open(config_filename, 'r') as stream:
            try:
                data = yaml.safe_load(stream)
                if data['detection_recording'] is not None and data['detection_recording']['enable']:
                    self.detection_recording_config = DetectionRecordingConfig(
                        enable=data['detection_recording']['enable'],
                        save_dir=data['detection_recording']['save_dir']
                    )
                else:
                    self.detection_recording_config = DetectionRecordingConfig(enable=False)
            except yaml.YAMLError as exc:
                print(exc)
        

    def pointcloud_callback(self, msg):
        with self.lock:
            self.pointcloud_queue.append(msg)

    def cluster_callback(self, msg):
        with self.lock:
            self.cluster_queue.append(msg)

    def find_closest_pointcloud_msg(self, timestamp):
        closest_msg = None
        smallest_diff = float('inf')
        for msg in self.pointcloud_queue:
            time_diff = abs((msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9) - (timestamp.sec + timestamp.nanosec * 1e-9))
            if time_diff < smallest_diff:
                smallest_diff = time_diff
                closest_msg = msg
        #print(f"Smallest diff: {smallest_diff}")
        return closest_msg

    def find_closest_cluster_msg(self, timestamp):
        closest_msg = None
        smallest_diff = float('inf')
        for msg in self.cluster_queue:
            time_diff = abs((msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9) - (timestamp.sec + timestamp.nanosec * 1e-9))
            if time_diff < smallest_diff:
                smallest_diff = time_diff
                closest_msg = msg
        #print(f"Smallest diff: {smallest_diff}")
        return closest_msg

    def callback(self, msg):
        
        start_time = time.time()

        ember_bbox_array = msg.boxes

        seconds = msg.header.stamp.sec
        nanoseconds = msg.header.stamp.nanosec

        print(f"Received message at {seconds}.{nanoseconds}")

        bboxes_array = np.array([get_2d_bbox_from_3d_bbox(np.array([[p.x, p.y, p.z] for p in ember_bbox.points])) for ember_bbox in ember_bbox_array])
        bboxes_to_track = np.array([get_track_struct_from_2d_bbox(bbox) for bbox in bboxes_array])
        bboxes_z_values = np.array([get_z_value_range_from_3d_bbox(np.array([[p.x, p.y, p.z] for p in ember_bbox.points])) for ember_bbox in ember_bbox_array])
        # centroids2d_array = np.array([[cluster.centroid.x, cluster.centroid.y] for cluster in ember_cluster_array]) TODO keep centroids in ember_bbox_array
        centroids2d_array = np.array([get_centroid_from_bbox(bbox) for bbox in bboxes_array])

        tracking_res = self.ocsort.update_v3(bboxes_to_track, bboxes_z_values)
        tracking_ids = tracking_res[:, 4] - 1 # on update, the tracker id is incremented by 1

        MAX_TIME_SINCE_UPDATE = 60
        MIN_NUM_OBSERVATIONS = 15

        valid_in_scope_trks = np.array([trk for trk in self.ocsort.get_trackers() if trk.time_since_update < MAX_TIME_SINCE_UPDATE and trk.hits > MIN_NUM_OBSERVATIONS])
        valid_by_id_trks = np.array([trk for trk in self.ocsort.get_trackers() if trk.id in tracking_ids])
        valid_in_scope_id_trks = np.array([trk for trk in valid_by_id_trks if trk.time_since_update < MAX_TIME_SINCE_UPDATE and trk.hits > MIN_NUM_OBSERVATIONS])

        print(f"Valid in scope trackers: {[trk.id for trk in valid_in_scope_trks]}")

        #self.clear_occ_grid()
        self.occupancy_grid.decay_occ_grid()
        self.occupancy_grid.update_occ_grid_poly(valid_in_scope_trks)

        header = msg.header
        occ_grid_msg = self.occupancy_grid.to_ros2_msg(header)
        self.pub.publish(occ_grid_msg)

        objec_tracking_res_types = np.array([KFTrackerObjectTypes.STATIC] * len(tracking_res))
        for i, track in enumerate(tracking_res):
            track_id = int(track[4]-1)
            kalman_tracker = self.ocsort.get_tracker_by_id(track_id)
            if kalman_tracker is not None:
                objec_tracking_res_types[i] = kalman_tracker.object_type
            else:
                objec_tracking_res_types[i] = KFTrackerObjectTypes.INVALID

        vis_input = {
            "valid_in_scope_trks": valid_in_scope_trks,
            "occ_grid": self.occupancy_grid,
            "tracking_res": tracking_res,
            "objec_tracking_res_types": objec_tracking_res_types
        }

        if self.detection_recording_config.enable:
            # use valid_in_scope_trks to get the bboxes and centroids
            bboxes_points = []
            centroids = []
            for trk in valid_in_scope_trks:
                # get position of element in tracking_ids
                if trk.id not in tracking_ids:
                    p1_x, p1_y, p2_x, p2_y = trk.get_state()[0][:4]
                else:
                    if trk.last_observation.sum() < 0:
                        print(f"ERROR: no last observation for track {trk.id}, using current state")
                        p1_x, p1_y, p2_x, p2_y = trk.get_state()[0][:4]
                    else:
                        p1_x, p1_y, p2_x, p2_y = trk.last_observation[:4]

            
                bbox = np.array([[p1_x, p1_y], [p2_x, p1_y], [p2_x, p2_y], [p1_x, p2_y]])

                centroid = np.mean(bbox, axis=0)
                bboxes_points.append(bbox)
                centroids.append(centroid)
            self.detection_recorder.record(bboxes_points, centroids, header.stamp.sec, header.stamp.nanosec)
                
            

        cluster_msg = self.find_closest_cluster_msg(header.stamp)
        if cluster_msg is not None:
            vis_input["ember_cluster_array"] = cluster_msg.clusters
        else:
            vis_input["ember_cluster_array"] = None

        self.vis_input_queue.put(vis_input)

def signal_handler(sig, frame, node, process, queue):
    print("SIGINT received")
    queue.put(None)
    process.terminate()
    process.join()
    node.destroy_node()
    rclpy.shutdown()
    exit(0)

def main(args=None):

    parser = argparse.ArgumentParser(description='Open3D Point Cloud Visualizer')
    parser.add_argument('config_fp', type=str, help='Filepath for configuration file')
    parsed_args = parser.parse_args(args=args)

    o3d_vis_input_queue = multiprocessing.Queue()
    o3d_vis_worker_process = multiprocessing.Process(target=o3d_vis_worker, args=(o3d_vis_input_queue,))
    o3d_vis_worker_process.start()

    signal.signal(signal.SIGINT, lambda sig, frame: signal_handler(sig, frame, node, o3d_vis_worker_process, o3d_vis_input_queue))

    rclpy.init(args=args)
    node = ClusterBoundingBoxViz(o3d_vis_input_queue, config_filename=parsed_args.config_fp)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        o3d_vis_input_queue.put(None)
        o3d_vis_worker_process.terminate()
        o3d_vis_worker_process.join()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
