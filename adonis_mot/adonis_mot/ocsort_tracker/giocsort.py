"""
    This script is adopted from the SORT script by Alex Bewley alex@bewley.ai
"""
from __future__ import print_function

import numpy as np
from adonis_mot.ocsort_tracker.association import *
from adonis_mot.ocsort_tracker.utils import *
from adonis_mot.ocsort_tracker.kalmantracker import KalmanBoxTracker

"""
    We support multiple ways for association cost calculation, by default
    we use IoU. GIoU may have better performance in some situations. We note 
    that we hardly normalize the cost by all methods to (0,1) which may not be 
    the best practice.
"""
ASSO_FUNCS = {  "iou": iou_batch,
                "giou": giou_batch,
                "ciou": ciou_batch,
                "diou": diou_batch,
                "ct_dist": ct_dist}


class GIOCSort(object):
    def __init__(self, max_age=30, min_hits=3, inertia_iou_threshold=0.2, growth_iou_threshold=0.1, default_iou_threshold=0.3,
                 ignore_t=1, delta_t=3, asso_func="iou", inertia=0.2, intertia_age_weight=0.5, growth_rate=0.1, growth_age_weight=0.5):
        """
        Sets key parameters for SORT
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.inertia_iou_threshold = inertia_iou_threshold
        self.growth_iou_threshold = growth_iou_threshold
        self.default_iou_threshold = default_iou_threshold
        self.trackers = []
        self.frame_count = 0
        self.ignore_t = ignore_t # number of frames to ignore the tracker
        self.delta_t = delta_t # number of frames to look back for the tracker
        self.asso_func = ASSO_FUNCS[asso_func]
        self.inertia = inertia
        self.inertia_age_weight = intertia_age_weight
        self.growth_rate = growth_rate
        self.growth_age_weight = growth_age_weight
        KalmanBoxTracker.count = 0

    def get_trackers(self):
        return self.trackers

    # use the inertia association as first association and then growth as second association
    def update_v2(self, dets, centroids=None):
        """
        Params:
        dets - a numpy array of detections in the format [[x1,y1,x2,y2],[x1,y1,x2,y2],...]
        Requires: this method must be called once for each frame even with empty detections (use np.empty((0, 4)) for frames without detections).
        Returns a similar array, where the last column is the object ID.
        NOTE: The number of objects returned may differ from the number of detections provided.
        """

        if dets is None:
            return np.empty((0, 5))
        elif dets.shape[1] == 4:
            dets = np.concatenate((dets, np.ones((dets.shape[0], 1))), axis=1)

        self.frame_count += 1

        trks = np.zeros((len(self.trackers), 5))
        grown_trks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
                grown_trks[t] = trk
            else:
                grown_trks[t] = self.trackers[t].grow_bbox(trk)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        grown_trks = np.ma.compress_rows(np.ma.masked_invalid(grown_trks))
        for t in reversed(to_del):
            self.trackers.pop(t)

        """
            FIRST round of association, based on inertia and tracker ages
        """
            
        last_boxes = np.array([trk.last_observation for trk in self.trackers])
        tracker_ages = np.array([trk.age for trk in self.trackers])
        velocities = np.array([trk.velocity if trk.velocity is not None else np.array((0, 0)) for trk in self.trackers])
        k_observations = np.array([k_previous_obs(trk.observations, trk.age, self.delta_t) for trk in self.trackers])

        matched_first_as, unmatched_dets_first_as, unmatched_trks_first_as = associate_inertia_boxes(
            dets, trks, self.inertia_iou_threshold, velocities, k_observations, self.inertia, tracker_ages, self.inertia_age_weight)
        for m in matched_first_as:
            self.trackers[m[1]].update(dets[m[0], :])
        
        unmatched_dets_first_to_second_map = dict()
        for i in range(len(unmatched_dets_first_as)):
            unmatched_dets_first_to_second_map[i] = unmatched_dets_first_as[i]
        
        unmatched_trks_first_to_second_map = dict()
        for i in range(len(unmatched_trks_first_as)):
            unmatched_trks_first_to_second_map[i] = unmatched_trks_first_as[i]

        """
            SECOND round of association, based on grown boxes and tracker ages
        """
        
        dets_sec_as = np.array([dets[i] for i in unmatched_dets_first_as])
        trks_sec_as = np.array([grown_trks[i] for i in unmatched_trks_first_as])
        tracker_ages_sec_as = np.array([self.trackers[i].age for i in unmatched_trks_first_as])

        matched_sec_as, unmatched_dets_sec_as, unmatched_trks_sec_as = associate_growth_boxes(
            dets_sec_as, trks_sec_as, self.growth_iou_threshold, tracker_ages_sec_as, self.growth_age_weight)
        for m in matched_sec_as:
            tracker_ind = unmatched_trks_first_as[m[1]]
            self.trackers[tracker_ind].update(dets_sec_as[m[0], :])

        unmatched_dets = np.array([unmatched_dets_first_to_second_map[i] for i in unmatched_dets_sec_as])
        unmatched_trks = np.array([unmatched_trks_first_to_second_map[i] for i in unmatched_trks_sec_as])

        matched_ids_first_as = [self.trackers[m[1]].id+1 for m in matched_first_as]
        matched_ids_sec_as = [self.trackers[unmatched_trks_first_as[m[1]]].id+1 for m in matched_sec_as]

        print(f'Mathed ids, first association: {matched_ids_first_as}')
        print(f'Mathed ids, second association: {matched_ids_sec_as}')
        print(f'Unmatched dets: {len(unmatched_dets)}, unmatched trks: {len(unmatched_trks)}')

        """
            THIRD round of association, simple IoU matching by OCR according to oc-sort original code
            TODO use different IoU thresholds for different rounds
        """
        if unmatched_dets.shape[0] > 0 and unmatched_trks.shape[0] > 0:
            left_dets = dets[unmatched_dets]
            left_trks = last_boxes[unmatched_trks]
            iou_left = self.asso_func(left_dets, left_trks)
            iou_left = np.array(iou_left)
            if iou_left.max() > self.default_iou_threshold:
                rematched_indices = linear_assignment(-iou_left)
                to_remove_det_indices = []
                to_remove_trk_indices = []
                for m in rematched_indices:
                    det_ind, trk_ind = unmatched_dets[m[0]], unmatched_trks[m[1]]
                    if iou_left[m[0], m[1]] < self.default_iou_threshold:
                        continue
                    self.trackers[trk_ind].update(dets[det_ind, :], centroids[det_ind])
                    to_remove_det_indices.append(det_ind)
                    to_remove_trk_indices.append(trk_ind)
                unmatched_dets = np.setdiff1d(unmatched_dets, np.array(to_remove_det_indices))
                unmatched_trks = np.setdiff1d(unmatched_trks, np.array(to_remove_trk_indices))

        for m in unmatched_trks:
            self.trackers[m].update(None)

        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i, :], ignore_t=self.ignore_t, delta_t=self.delta_t, growth_rate=self.growth_rate)
            self.trackers.append(trk)
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            if trk.last_observation.sum() < 0:
                d = trk.get_state()[0]
            else:
                d = trk.last_observation[:4]
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(np.concatenate((d, [trk.id+1])).reshape(1, -1))
            i -= 1
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)
        if len(ret) > 0:
            return np.concatenate(ret)
        return np.empty((0, 5))

    def update_v1(self, dets, centroids=None):
        """
        Params:
        dets - a numpy array of detections in the format [[x1,y1,x2,y2],[x1,y1,x2,y2],...]
        Requires: this method must be called once for each frame even with empty detections (use np.empty((0, 4)) for frames without detections).
        Returns a similar array, where the last column is the object ID.
        NOTE: The number of objects returned may differ from the number of detections provided.
        """

        if dets is None:
            return np.empty((0, 5))
        elif dets.shape[1] == 4:
            dets = np.concatenate((dets, np.ones((dets.shape[0], 1))), axis=1)

        self.frame_count += 1

        trks = np.zeros((len(self.trackers), 5))
        grown_trks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
                grown_trks[t] = trk
            else:
                grown_trks[t] = self.trackers[t].grow_bbox(trk)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        grown_trks = np.ma.compress_rows(np.ma.masked_invalid(grown_trks))
        for t in reversed(to_del):
            self.trackers.pop(t)

        """
            FIRST round of association, based on growth and tracker ages
        """
            
        last_boxes = np.array([trk.last_observation for trk in self.trackers])
        tracker_ages = np.array([trk.age for trk in self.trackers])

        matched_first_as, unmatched_dets_first_as, unmatched_trks_first_as = associate_growth_boxes(
            dets, grown_trks, self.growth_iou_threshold, tracker_ages, self.growth_age_weight)
        for m in matched_first_as:
            self.trackers[m[1]].update(dets[m[0], :])

        unmatched_dets_first_to_second_map = dict()
        for i in range(len(unmatched_dets_first_as)):
            unmatched_dets_first_to_second_map[i] = unmatched_dets_first_as[i]
        
        unmatched_trks_first_to_second_map = dict()
        for i in range(len(unmatched_trks_first_as)):
            unmatched_trks_first_to_second_map[i] = unmatched_trks_first_as[i]

        """
            SECOND round of association, based on inertia and tracker ages
        """

        # use the unmathced_trks to get the remaining trackers
        # sec_as = second association
        dets_sec_as = np.array([dets[i] for i in unmatched_dets_first_as])
        trks_sec_as = np.array([trks[i] for i in unmatched_trks_first_as])
        k_observations_sec_as = np.array([k_previous_obs(self.trackers[i].observations, self.trackers[i].age, self.delta_t) for i in unmatched_trks_first_as])
        velocities_sec_as = np.array([self.trackers[i].velocity if self.trackers[i].velocity is not None else np.array((0, 0)) for i in unmatched_trks_first_as])
        tracker_ages_sec_as = np.array([self.trackers[i].age for i in unmatched_trks_first_as])

        matched_sec_as, unmatched_dets_sec_as, unmatched_trks_sec_as = associate_inertia_boxes(
            dets_sec_as, trks_sec_as, self.inertia_iou_threshold, velocities_sec_as, k_observations_sec_as, 
            self.inertia, tracker_ages_sec_as, self.inertia_age_weight)
        for m in matched_sec_as:
            tracker_ind = unmatched_trks_first_as[m[1]]
            self.trackers[tracker_ind].update(dets_sec_as[m[0], :])

        unmatched_dets = np.array([unmatched_dets_first_to_second_map[i] for i in unmatched_dets_sec_as])
        unmatched_trks = np.array([unmatched_trks_first_to_second_map[i] for i in unmatched_trks_sec_as])

        matched_ids_first_as = [self.trackers[m[1]].id+1 for m in matched_first_as]
        matched_ids_sec_as = [self.trackers[unmatched_trks_first_as[m[1]]].id+1 for m in matched_sec_as]

        #print(f'Mathed ids, first association: {matched_ids_first_as}')
        #print(f'Mathed ids, second association: {matched_ids_sec_as}')
        #print(f'Unmatched dets: {len(unmatched_dets)}, unmatched trks: {len(unmatched_trks)}')
        
        """
            THIRD round of association, simple IoU matching by OCR according to oc-sort original code
            TODO use different IoU thresholds for different rounds
        """
        if unmatched_dets.shape[0] > 0 and unmatched_trks.shape[0] > 0:
            left_dets = dets[unmatched_dets]
            left_trks = last_boxes[unmatched_trks]
            iou_left = self.asso_func(left_dets, left_trks)
            iou_left = np.array(iou_left)
            if iou_left.max() > self.default_iou_threshold:
                rematched_indices = linear_assignment(-iou_left)
                to_remove_det_indices = []
                to_remove_trk_indices = []
                for m in rematched_indices:
                    det_ind, trk_ind = unmatched_dets[m[0]], unmatched_trks[m[1]]
                    if iou_left[m[0], m[1]] < self.default_iou_threshold:
                        continue
                    self.trackers[trk_ind].update(dets[det_ind, :], centroids[det_ind])
                    to_remove_det_indices.append(det_ind)
                    to_remove_trk_indices.append(trk_ind)
                unmatched_dets = np.setdiff1d(unmatched_dets, np.array(to_remove_det_indices))
                unmatched_trks = np.setdiff1d(unmatched_trks, np.array(to_remove_trk_indices))

        for m in unmatched_trks:
            self.trackers[m].update(None)

        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i, :], ignore_t=self.ignore_t, delta_t=self.delta_t, growth_rate=self.growth_rate)
            self.trackers.append(trk)
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            if trk.last_observation.sum() < 0:
                d = trk.get_state()[0]
            else:
                d = trk.last_observation[:4]
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(np.concatenate((d, [trk.id+1])).reshape(1, -1))
            i -= 1
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)
        if len(ret) > 0:
            return np.concatenate(ret)
        return np.empty((0, 5))


