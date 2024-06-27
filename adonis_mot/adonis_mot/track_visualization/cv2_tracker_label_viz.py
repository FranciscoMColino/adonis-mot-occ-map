import cv2
import numpy as np

from adonis_mot.ocsort_tracker.kalmantracker import ObjectTypes as KFTrackerObjectTypes

class TrackerLabelVisualizer:
    def __init__(self, window_name="Track ID and Type"):
        self.window_name = window_name
        self.image = None

    def capture_image(self, vis):
        """
        Capture the current image from the Open3D visualizer.
        """
        image = vis.capture_screen_float_buffer(do_render=True)
        image = np.asarray(image) * 255
        image = image.astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
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

    def add_text_overlay(self, vis, image, tracking_res, object_types=None):
        """
        Add text overlay to the image using OpenCV.
        """
        # Get the view and projection matrices
        view_control = vis.get_view_control()
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

    def generate_frame(self, vis, tracking_res, object_types=None):
        """
        Generate the frame with the tracking results.
        """
        self.image = self.capture_image(vis)
        image = self.add_text_overlay(vis, self.image, tracking_res, object_types)
        return image