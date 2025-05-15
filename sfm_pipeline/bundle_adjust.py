# sfm_pipeline/bundle_adjust.py
import numpy as np
from scipy.optimize import least_squares
import cv2

def calc_opt_reproj_err(obj_pts) -> np.array:
    '''
    Calculates the reprojection error vector used during bundle adjustment optimization.
    
    Parameters:
    - obj_pts: A 1D array containing concatenated parameters:
      * First 12 elements: flattened camera projection matrix (3x4)
      * Next 9 elements: flattened camera intrinsic matrix (3x3)
      * Next portion: 2D image points
      * Remaining: 3D object points
    
    Returns:
    - Flattened vector of squared reprojection errors normalized by the number of points.
    '''
    # Extract and reshape the camera projection matrix (3x4)
    trans_mat = obj_pts[0:12].reshape((3, 4))
    
    # Extract and reshape the intrinsic matrix (3x3)
    K = obj_pts[12:21].reshape((3, 3))
    
    # Calculate how many 2D image points are included based on remaining length (40% assumed)
    rest = int(len(obj_pts[21:]) * 0.4)
    
    # Extract 2D image points and reshape to Nx2 (transpose for correct shape)
    p = obj_pts[21:21 + rest].reshape((2, int(rest / 2))).T
    
    # Extract 3D object points (remaining part of input vector)
    obj_pts = obj_pts[21 + rest:].reshape((int(len(obj_pts[21 + rest:]) / 3), 3))
    
    # Extract rotation matrix (3x3) and translation vector (3x1) from the projection matrix
    R = trans_mat[:3, :3]
    t = trans_mat[:3, 3]
    
    # Convert rotation matrix to rotation vector (required by OpenCV projectPoints)
    rot_vec, _ = cv2.Rodrigues(R)
    
    # Project 3D points into 2D image points using current camera parameters
    proj_pts, _ = cv2.projectPoints(obj_pts, rot_vec, t, K, None)
    
    # Reshape projected points to Nx2 array for error calculation
    proj_pts = proj_pts[:, 0, :]
    
    # Calculate squared reprojection errors for each point (difference between observed and projected)
    err = [ (p[idx] - proj_pts[idx])**2 for idx in range(len(p))]
    
    # Return flattened error vector normalized by the number of points (to scale error)
    return np.array(err).ravel() / len(p)


def bundle_adjust(pts_3d, img_pts, trans_mat, K, err_thresh) -> tuple:
    '''
    Performs bundle adjustment to refine camera parameters and 3D point estimates.
    
    Uses the Trust Region Reflective ('trf') algorithm by default.
    
    Parameters:
    - pts_3d: Initial 3D points (Nx3 array).
    - img_pts: Corresponding 2D image points (Nx2 array).
    - trans_mat: Initial camera projection matrix (3x4).
    - K: Initial camera intrinsic matrix (3x3).
    - err_thresh: Error threshold for optimization convergence.
    
    Returns:
    - Optimized 3D points, 2D image points, and camera projection matrix as a tuple.
    '''
    Levenberg_Marquardt = False  # Flag to toggle between optimization algorithms
    
    if Levenberg_Marquardt:
        # Flatten all variables into a single 1D array for optimization
        variables = np.hstack((trans_mat.ravel(), K.ravel(), img_pts.ravel(), pts_3d.ravel()))
        
        # Run least squares optimization using Trust Region Reflective method with robust loss
        optimized = least_squares(
            calc_opt_reproj_err,
            variables,
            method='trf',          # Other options include 'dogbox'
            loss='cauchy',         # Robust to outliers like Linear, huber, soft_l1 as per the chosen method 
            f_scale=1.0,
            verbose=0,
            ftol=1e-6,
            xtol=1e-6,
            gtol=err_thresh,
            max_nfev=200
        ).x
        
        # Extract optimized intrinsic matrix (3x3)
        K = optimized[12:21].reshape((3, 3))
        
        # Calculate size of 2D points vector after optimization
        rest = int(len(optimized[21:]) * 0.4)
        
        # Extract optimized 2D image points and reshape accordingly
        img_pts_opt = optimized[21:21 + rest].reshape((2, int(rest / 2))).T
        
        # Extract optimized 3D points
        pts_3d_opt = optimized[21 + rest:].reshape((int(len(optimized[21 + rest:]) / 3), 3))
        
        # Extract optimized camera projection matrix (3x4)
        trans_mat_opt = optimized[0:12].reshape((3, 4))
        
        return pts_3d_opt, img_pts_opt, trans_mat_opt
    
    else:
        # Prepare variables vector without specifying loss or method explicitly
        variables = np.hstack((trans_mat.ravel(), K.ravel()))
        variables = np.hstack((variables, img_pts.ravel()))
        variables = np.hstack((variables, pts_3d.ravel()))

        # Run least squares optimization with default settings, stopping when error threshold is reached
        optimized = least_squares(calc_opt_reproj_err, variables, gtol=err_thresh).x
        
        # Extract optimized intrinsic matrix (3x3)
        K = optimized[12:21].reshape((3, 3))
        
        # Calculate number of 2D image points in optimized variables
        rest = int(len(optimized[21:]) * 0.4)
        
        # Return optimized 3D points, 2D points, and camera projection matrix as a tuple
        return (
            optimized[21 + rest:].reshape((int(len(optimized[21 + rest:]) / 3), 3)),
            optimized[21:21 + rest].reshape((2, int(rest / 2))).T,
            optimized[0:12].reshape((3, 4))
        )
