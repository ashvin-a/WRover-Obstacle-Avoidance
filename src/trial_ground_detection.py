import cv2
import numpy as np


def detect_ground_points(frame_with_depth_values):

    floor_threshold_depth = 0.3 # To be figured out experimentally
    height, width = frame_with_depth_values.shape
    depth = frame_with_depth_values # name sake
    ground_points = []
    

    # To store where ground ends for each column
    ground_mask = np.zeros_like(depth, dtype=bool)
    
    for x in range(width):
        for y in range(height):
            diff = abs(depth[y, x] - depth[y-1, x])

            if diff > floor_threshold_depth:
                ground_points.append((y, x))
                break

    for point in range(len(ground_points) - 2):
        cv2.line(depth, ground_points[point], ground_points[point + 1], color=(0, 0, 255), thickness=2)
    return depth