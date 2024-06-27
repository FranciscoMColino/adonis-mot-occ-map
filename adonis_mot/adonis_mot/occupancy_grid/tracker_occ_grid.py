import numpy as np
import math

from adonis_mot.ocsort_tracker.kalmantracker import ObjectTypes as KFTrackerObjectTypes
from adonis_mot.ocsort_tracker.utils import convert_x_to_bbox

# TODO should this be in a utils file?
def points_in_polygon(corners):
        # corners is a list of tuples [(x1, y1), (x2, y2), ..., (xn, yn)]
        # Assuming corners are given in counter-clockwise order

        # Find bounding box of the polygon
        xmin = min(x for x, y in corners)
        xmax = max(x for x, y in corners)
        ymin = min(y for x, y in corners)
        ymax = max(y for x, y in corners)

        points_inside = []

        # Scanline algorithm
        for y in range(ymin, ymax + 1):
            intersections = []

            # Find intersections of scanline with polygon edges
            for i in range(len(corners)):
                x1, y1 = corners[i]
                x2, y2 = corners[(i + 1) % len(corners)]

                if y1 <= y < y2 or y2 <= y < y1:
                    # Calculate intersection x-coordinate
                    if y1 == y2:
                        continue  # Skip horizontal edges

                    x_intersect = (y - y1) * (x2 - x1) / (y2 - y1) + x1
                    intersections.append(x_intersect)

            # Sort intersection points by x-coordinate
            intersections.sort()

            # Fill points between intersections
            for j in range(0, len(intersections), 2):
                x_start = int(intersections[j])
                x_end = int(intersections[j + 1]) if j + 1 < len(intersections) else x_start

                # Add all points (x, y) for this scanline segment
                for x in range(x_start, x_end + 1):
                    points_inside.append((x, y))

        return points_inside


class TrackerOccGrid:
    def __init__(self, x_o, y_o, width, height, resolution, cur_occ_weight=0.2, fut_occ_weight=0.18, decay_rate=0.1):

        self.x_o = x_o
        self.y_o = y_o
        self.width = width
        self.height = height
        self.resolution = resolution
        self.grid = np.zeros((int(height / resolution), int(width / resolution)))

        self.cur_occ_weight = cur_occ_weight
        self.fut_occ_weight = fut_occ_weight
        self.decay_rate = decay_rate

    def clear_occ_grid(self):
        self.grid = np.zeros((int(self.height / self.resolution), int(self.width / self.resolution)))

    def decay_occ_grid(self, decay_rate=None):
        if decay_rate is None:
            decay_rate = self.decay_rate
        self.grid -= decay_rate
        self.grid = np.clip(self.grid, 0, 1)

    def convert_bbox_to_grid_coords(self, bbox, safe_margin=0):

        x1, y1, x2, y2 = bbox[:4]

        # add a safe margin
        x1 -= safe_margin
        y1 -= safe_margin
        x2 += safe_margin
        y2 += safe_margin

        x1 = max(x1, self.x_o)
        y1 = max(y1, self.y_o)
        x2 = min(x2, self.x_o + self.width)
        y2 = min(y2, self.y_o + self.height)

        x1 = math.floor((x1 - self.x_o) / self.resolution)
        y1 = math.floor((y1 - self.y_o) / self.resolution)
        x2 = math.ceil((x2 - self.x_o) / self.resolution)
        y2 = math.ceil((y2 - self.y_o) / self.resolution)

        return x1, y1, x2, y2
    
    def update_occ_grid_poly(self, trackers, safe_margin_a=0.1, safe_margin_b=0.4, k_ahead=30):

        # would it be faster to compute for all points in a box whose corners are the bbox cur and future or to compute just for the points in the polygon?

        MAX_TIME_SINCE_UPDATE = 30
        MIN_NUM_OBSERVATIONS = 10

        display_lines = False

        for trk in trackers:

            if trk.time_since_update > MAX_TIME_SINCE_UPDATE or len(trk.observations) < MIN_NUM_OBSERVATIONS:
                continue

            bbox = convert_x_to_bbox(trk.kf.x)[0]

            if bbox is None or np.any(np.isnan(bbox)):
                continue

            x1, y1, x2, y2 = self.convert_bbox_to_grid_coords(bbox, safe_margin=safe_margin_a)
            self.grid[y1:y2, x1:x2] = 1

            if trk.object_type == KFTrackerObjectTypes.DYNAMIC:
                future_bbox = convert_x_to_bbox(trk.get_k_away_prediction(k_ahead))[0]

                if future_bbox is not None and not np.any(np.isnan(future_bbox)):

                    future_x1, future_y1, future_x2, future_y2 = self.convert_bbox_to_grid_coords(future_bbox, safe_margin=safe_margin_a)

                    center_cur = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
                    center_future = np.array([(future_x1 + future_x2) / 2, (future_y1 + future_y2) / 2])

                    radius_cur = np.sqrt((center_cur[0] - x1) ** 2 + (center_cur[1] - y1) ** 2) + safe_margin_b
                    radius_future = np.sqrt((center_future[0] - future_x1) ** 2 + (center_future[1] - future_y1) ** 2) + safe_margin_b

                    vector_cur_to_future = center_future - center_cur
                    vector_cur_to_future /= np.linalg.norm(vector_cur_to_future)

                    vector_perpendicular = np.array([vector_cur_to_future[1], -vector_cur_to_future[0]])

                    corner_1 = center_cur + vector_perpendicular * radius_cur - vector_cur_to_future * radius_cur
                    corner_2 = center_cur - vector_perpendicular * radius_cur - vector_cur_to_future * radius_cur
                    corner_3 = center_future + vector_perpendicular * radius_future + vector_cur_to_future * radius_future
                    corner_4 = center_future - vector_perpendicular * radius_future + vector_cur_to_future * radius_future

                    dist_cur_to_future = np.sqrt((center_cur[0] - center_future[0]) ** 2 + (center_cur[1] - center_future[1]) ** 2)

                    if np.any(np.isnan(corner_1)) or np.any(np.isnan(corner_2)) or np.any(np.isnan(corner_3)) or np.any(np.isnan(corner_4)):
                        continue
                    
                    corner_1 = corner_1.astype(int)
                    corner_2 = corner_2.astype(int)
                    corner_3 = corner_3.astype(int)
                    corner_4 = corner_4.astype(int)

                    points_inside = points_in_polygon([corner_1, corner_3, corner_4, corner_2])
                    for x, y in points_inside:
                        dist_cur = np.sqrt((x - center_cur[0]) ** 2 + (y - center_cur[1]) ** 2)
                        dist_future = np.sqrt((x - center_future[0]) ** 2 + (y - center_future[1]) ** 2)
                        total_dist = dist_cur + dist_future

                        if total_dist != 0:
                            value = (dist_future / total_dist) * self.cur_occ_weight + (dist_cur / total_dist) * self.fut_occ_weight
                        else:
                            value = 0.5 * (1 + self.fut_occ_weight)

                        value *= (dist_cur_to_future / total_dist) ** 2
                        
                        self.grid[y, x] += value
                        self.grid[y, x] = np.clip(self.grid[y, x], 0, 1)

                    if display_lines:
                        # Draw the line between the centers
                        for i in range(0, 100):
                            t = i / 100
                            x = int(center_cur[0] + t * (center_future[0] - center_cur[0]))
                            y = int(center_cur[1] + t * (center_future[1] - center_cur[1]))
                            self.grid[y, x] = 1

                        # Draw the line between the corners
                        for i in range(0, 100):
                            t = i / 100
                            x = int(corner_1[0] + t * (corner_3[0] - corner_1[0]))
                            y = int(corner_1[1] + t * (corner_3[1] - corner_1[1]))
                            self.grid[y, x] = 1

                            x = int(corner_2[0] + t * (corner_4[0] - corner_2[0]))
                            y = int(corner_2[1] + t * (corner_4[1] - corner_2[1]))
                            self.grid[y, x] = 1

    def update_occ_grid_slow(self, trackers, safe_margin=0.1, k_ahead=30, radial_margin=2):

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
            self.grid[y1:y2, x1:x2] = 1

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
                                        if 0 <= i + dx < self.grid.shape[1] and 0 <= j + dy < self.grid.shape[0]:
                                            dist_from_center = np.sqrt(dx ** 2 + dy ** 2)
                                            if dist_from_center <= radial_margin:
                                                decay_factor = (1 - (dist_from_center / radial_margin)) ** 4
                                                dist_diff_factor = (dist_cur_to_future / total_dist) ** 4
                                                self.grid[j + dy, i + dx] += value * decay_factor * dist_diff_factor
                                                self.grid[j + dy, i + dx] = np.clip(self.grid[j + dy, i + dx], 0, 1)
                else:
                    # No future bbox found, apply radial margin around current bbox
                    if radial_margin > 0:
                        for i in range(x1, x2):
                            for j in range(y1, y2):
                                value = 0.5 * (1 + self.future_pred_occ_weight)  # Use the current bbox value

                                # Apply radial margin with quadratic decay
                                for dx in range(-radial_margin, radial_margin + 1):
                                    for dy in range(-radial_margin, radial_margin + 1):
                                        if 0 <= i + dx < self.grid.shape[1] and 0 <= j + dy < self.grid.shape[0]:
                                            dist_from_center = np.sqrt(dx ** 2 + dy ** 2)
                                            if dist_from_center <= radial_margin:
                                                decay_factor = (1 - (dist_from_center / radial_margin)) ** 2  # Quadratic decay
                                                self.grid[j + dy, i + dx] += value * decay_factor
                                                self.grid[j + dy, i + dx] = np.clip(self.grid[j + dy, i + dx], 0, 1)
