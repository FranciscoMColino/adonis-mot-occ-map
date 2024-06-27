import cv2
import numpy as np

class OccupancyGridVisualizer:
    def __init__(self, window_name="Occupancy Grid"):
        self.window_name = window_name
        self.image = None

    def capture_occ_grid_image(self, occupancy_grid):

        # Draw the occupancy grid in opencv2 new window
        
        max_size = 640

        if occupancy_grid.width > occupancy_grid.height:
            occ_grid_width = max_size
            occ_grid_height = int(max_size * (occupancy_grid.height / occupancy_grid.width))
        else:
            occ_grid_height = max_size
            occ_grid_width = int(max_size * (occupancy_grid.width / occupancy_grid.height))

        occ_grid = occupancy_grid.grid
        occ_grid = (1-occ_grid) * 255
        occ_grid = occ_grid.astype(np.uint8)

        occ_grid = cv2.resize(occ_grid, (occ_grid_width, occ_grid_height), interpolation=cv2.INTER_NEAREST)
        occ_grid = cv2.flip(occ_grid, 0)
        occ_grid = cv2.cvtColor(occ_grid, cv2.COLOR_GRAY2BGR)

        return occ_grid
    
    def generate_frame(self, occupancy_grid):
        """
        Generate a frame for the occupancy grid visualization.
        """
        image = self.capture_occ_grid_image(occupancy_grid)
        return image