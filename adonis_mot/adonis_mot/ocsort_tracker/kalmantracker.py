import numpy as np
from adonis_mot.ocsort_tracker.utils import *
from adonis_mot.ocsort_tracker.association import iou_single
from adonis_mot.ocsort_tracker.kalmanfilter import predict as kf_predict
from enum import Enum


# types can be either static or dynamic
class ObjectTypes(Enum):
    STATIC = 1
    DYNAMIC = 2
    INVALID = 3

class KalmanBoxTracker(object):
    """
    This class represents the internal state of individual tracked objects observed as bbox.
    """
    count = 0

    def __init__(self, bbox, ignore_t=1, delta_t=3, orig=False, growth_rate=0.1):
        """
        Initialises a tracker using initial bounding box.

        """
        # define constant velocity model
        if not orig:
          from adonis_mot.ocsort_tracker.kalmanfilter import KalmanFilterNew as KalmanFilter
          self.kf = KalmanFilter(dim_x=7, dim_z=4)
        else:
          from filterpy.kalman import KalmanFilter
          self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([[1, 0, 0, 0, 1, 0, 0], [0, 1, 0, 0, 0, 1, 0], [0, 0, 1, 0, 0, 0, 1], [
                            0, 0, 0, 1, 0, 0, 0],  [0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 1]])
        self.kf.H = np.array([[1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0],
                            [0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0]])

        self.kf.R[2:, 2:] *= 10.
        self.kf.P[4:, 4:] *= 1000.  # give high uncertainty to the unobservable initial velocities
        self.kf.P *= 10.
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        self.kf.x[:4] = convert_bbox_to_z(bbox)
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        """
        NOTE: [-1,-1,-1,-1,-1] is a compromising placeholder for non-observation status, the same for the return of 
        function k_previous_obs. It is ugly and I do not like it. But to support generate observation array in a 
        fast and unified way, which you would see below k_observations = np.array([k_previous_obs(...]]), let's bear it for now.
        """
        self.last_observation = np.array([-1, -1, -1, -1, -1])  # placeholder
        self.observations = dict()
        self.history_observations = []

        self.last_centroid = np.array([-1, -1])  # placeholder
        self.observed_centroids = dict()
        self.history_centroids = []

        self.velocity = None
        self.ignore_t = ignore_t
        self.delta_t = delta_t

        self.growth_rate = growth_rate
        self.start_growth_t = 0     # TODO make this a parameter
        self.start_growth_boost = 1 # TODO make this a parameter

        self.mean_w = None
        self.mean_h = None
        self.mean_count = 0

        self.box_dims = None
        
        self.object_type = ObjectTypes.DYNAMIC # default object type is dynamic
        self.object_type_threshold = 0.3

    def update_mean(self, bbox):
        self.mean_count += 1
        if self.mean_w is None:
            self.mean_w = np.abs(bbox[2] - bbox[0])
            self.mean_h = np.abs(bbox[3] - bbox[1])
        else:
            self.mean_w = (self.mean_w * (self.mean_count - 1) + np.abs(bbox[2] - bbox[0])) / self.mean_count
            self.mean_h = (self.mean_h * (self.mean_count - 1) + np.abs(bbox[3] - bbox[1])) / self.mean_count

    def get_mean_bbox(self):
        bbox = self.get_state()[0]
        if bbox is None:
            return None
        x_center = (bbox[2] + bbox[0]) / 2
        y_center = (bbox[3] + bbox[1]) / 2

        if self.mean_w is None or self.mean_h is None:
            return bbox

        bbox_w = self.mean_w
        bbox_h = self.mean_h
        bbox[0] = x_center - bbox_w / 2
        bbox[1] = y_center - bbox_h / 2
        bbox[2] = x_center + bbox_w / 2
        bbox[3] = y_center + bbox_h / 2
        return bbox
    
    def get_current_bbox(self):
        return self.get_state()[0]
    
    def interval_centroid_speed_direction(self, centroid, start_delta_t=30, end_delta_t=60):     
        previous_centroid = None
        velocities = np.zeros((end_delta_t - start_delta_t, 2))
        for i in range(start_delta_t, end_delta_t):
            dt = end_delta_t - i
            if self.age - dt in self.observed_centroids:
                previous_centroid = self.observed_centroids[self.age-dt]
                index = i - start_delta_t
                velocities[index] = centroid_speed_direction(previous_centroid, centroid)

        velocities = velocities[~np.all(velocities == 0, axis=1)]

        if len(velocities) > 0:
            velocity = np.mean(velocities, axis=0)
        else:
            velocity = centroid_speed_direction(self.last_centroid, centroid)

        return velocity
    
    def interval_mean_speed_direction(self, bbox, start_delta_t=30, end_delta_t=60):

        previous_box = None
        velocities = np.zeros((end_delta_t - start_delta_t, 2))
        for i in range(start_delta_t, end_delta_t):
            dt = end_delta_t - i
            if self.age - dt in self.observations:
                previous_box = self.observations[self.age-dt]
                index = i - start_delta_t
                velocities[index] = speed_direction(previous_box, bbox)

        velocities = velocities[~np.all(velocities == 0, axis=1)]

        if len(velocities) > 0:
            velocity = np.mean(velocities, axis=0)
        else:
            velocity = speed_direction(self.last_observation, bbox)

        return velocity

    def delta_speed_direction(self, bbox, delta_t=3):
        previous_box = None
        for i in range(delta_t):
            dt = delta_t - i
            if self.age - dt in self.observations:
                previous_box = self.observations[self.age-dt]
                break
        if previous_box is None:
            previous_box = self.last_observation
        """
            Estimate the track speed direction with observations \Delta t steps away
        """
        velocity = speed_direction(previous_box, bbox)
        return velocity
    
    def update_object_type(self, bbox, thresh=0.5):
        k = self.delta_t
        previous_box = k_previous_obs(self.observations, self.age, k)

        if previous_box is None:
            previous_box = self.last_observation

        bbox = bbox[:4]
        previous_box = previous_box[:4]
        iou_result = iou_single(bbox, previous_box)

        if iou_result >= thresh:
            self.object_type = ObjectTypes.STATIC
        else:
            self.object_type = ObjectTypes.DYNAMIC

    def update(self, bbox, centroid=None):
        """
        Updates the state vector with observed bbox.
        """
        if bbox is not None:

            self.update_mean(bbox)

            if self.last_observation.sum() >= 0:  # no previous observation
                """
                  Estimate the track speed direction
                """
                if centroid is not None:
                    self.velocity = self.interval_centroid_speed_direction(centroid, self.ignore_t, self.delta_t)
                else:
                    #self.velocity = self.delta_speed_direction(bbox, self.delta_t)
                    self.velocity = self.interval_mean_speed_direction(bbox, self.ignore_t, self.delta_t)
            
            """
              Insert new observations. This is a ugly way to maintain both self.observations
              and self.history_observations. Bear it for the moment.
            """
            self.last_observation = bbox
            self.observations[self.age] = bbox
            self.history_observations.append(bbox)

            if centroid is not None:
                self.last_centroid = centroid
                self.observed_centroids[self.age] = centroid
                self.history_centroids.append(centroid)

            self.time_since_update = 0
            self.history = []
            self.hits += 1
            self.hit_streak += 1

            self.kf.update(convert_bbox_to_z(bbox))
            self.update_object_type(bbox, self.object_type_threshold)
        else:
            self.kf.update(bbox)

    # TODO make time_since_update and velocity parameters
    def grow_bbox(self, bbox=None):

        bbox = bbox.copy()

        if self.time_since_update <= self.start_growth_t or bbox is None:
            print("No bbox or not enough time since update")
            return bbox
        
        x_center = (bbox[2] + bbox[0]) / 2
        y_center = (bbox[3] + bbox[1]) / 2

        if self.mean_h is not None and self.mean_w is not None:
            bbox_w = self.mean_w
            bbox_h = self.mean_h
        else:
            bbox_w = np.abs(bbox[2] - bbox[0])
            bbox_h = np.abs(bbox[3] - bbox[1])

        bbox_mean_size = (bbox_w + bbox_h) / 2
        
        bbox_mean_increase = bbox_mean_size * self.growth_rate * (self.time_since_update - self.start_growth_t + self.start_growth_boost)

        bbox_w_increase = bbox_mean_increase
        bbox_h_increase = bbox_mean_increase

        bbox_new_w = bbox_w + bbox_w_increase
        bbox_new_h = bbox_h + bbox_h_increase
        
        if bbox[0] < bbox[2]:
            bbox[0] = x_center - bbox_new_w / 2
            bbox[2] = x_center + bbox_new_w / 2
        else:
            bbox[0] = x_center + bbox_new_w / 2
            bbox[2] = x_center - bbox_new_w / 2

        if bbox[1] < bbox[3]:
            bbox[1] = y_center - bbox_new_h / 2
            bbox[3] = y_center + bbox_new_h / 2
        else:
            bbox[1] = y_center + bbox_new_h / 2
            bbox[3] = y_center - bbox_new_h / 2

        if self.velocity is not None:
            pred_direction = self.velocity
            
            bbox[0] += pred_direction[1] * bbox_mean_increase/2
            bbox[1] += pred_direction[0] * bbox_mean_increase/2
            bbox[2] += pred_direction[1] * bbox_mean_increase/2
            bbox[3] += pred_direction[0] * bbox_mean_increase/2

        return bbox

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        if((self.kf.x[6]+self.kf.x[2]) <= 0):
            self.kf.x[6] *= 0.0

        self.kf.predict()
        self.age += 1
        if(self.time_since_update > 0):
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x))

        return self.history[-1]

    def get_growth_bbox(self):
        if len(self.history) == 0:
            return None
        bbox = self.history[-1][0]
        return self.grow_bbox(bbox)
        
    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return convert_x_to_bbox(self.kf.x)
    
    def get_k_away_prediction(self, k):
        """
        Returns the k-th away prediction of the kalman filter.
        """
        x_next, P_next = self.kf.x, self.kf.P
        for _ in range(k):
            x_next, P_next = kf_predict(x_next, P_next, self.kf.F, self.kf.Q)

        return x_next