import numpy as np
import sensor_msgs_py.point_cloud2 as pc2

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

def get_z_value_range_from_3d_bbox(points):
    # Returns the min and max corners of the bounding box
    z_values = np.zeros(2)
    z_values[0] = np.min(points[:, 2])
    z_values[1] = np.max(points[:, 2])
    return z_values

def get_centroid_from_bbox(bbox):
    return np.mean(bbox, axis=0)