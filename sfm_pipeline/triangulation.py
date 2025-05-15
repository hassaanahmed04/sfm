# sfm_pipeline/triangulation.py
import cv2
import numpy as np

def triangulate_pts(point_2d_1, point_2d_2, projection_matrix_1, projection_matrix_2) -> tuple:
    '''
    Given corresponding 2D points from two camera views and their respective projection matrices,
    this function estimates the 3D coordinates of those points in space by triangulation.
    
    Parameters:
    - point_2d_1: 2D points from the first camera image (usually in homogeneous coordinates).
    - point_2d_2: 2D points from the second camera image (same format as above).
    - projection_matrix_1: Projection matrix (3x4) of the first camera.
    - projection_matrix_2: Projection matrix (3x4) of the second camera.
    
    Returns:
    - Tuple containing:
      * The transposed projection matrix of the first camera (for consistent orientation).
      * The transposed projection matrix of the second camera.
      * The 3D points reconstructed via triangulation, normalized from homogeneous coordinates 
        to Cartesian coordinates.
    '''
    # Use OpenCV's triangulatePoints function to compute 3D points in homogeneous coordinates.
    # Note that the projection matrices must be transposed to match the expected shape by OpenCV.
    pt_cloud = cv2.triangulatePoints(point_2d_1, point_2d_2, projection_matrix_1.T, projection_matrix_2.T)
    
    # The output is in homogeneous coordinates (x, y, z, w).
    # Normalize each point by dividing by the w component to convert to Cartesian coordinates (x/w, y/w, z/w).
    normalized_pt_cloud = pt_cloud / pt_cloud[3]
    
    # Return the transposed projection matrices along with the normalized 3D point cloud.
    return projection_matrix_1.T, projection_matrix_2.T, normalized_pt_cloud
