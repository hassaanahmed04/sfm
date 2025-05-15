# sfm_pipeline/reprojection.py
import cv2
import numpy as np

def calc_reproj_err(obj_pts, img_pts, trans_mat, K, homog) -> tuple:
    '''
    Computes the reprojection error between 3D object points projected into the image plane 
    and the corresponding observed 2D image points.
    
    Parameters:
    - obj_pts: 3D coordinates of object points (can be in homogeneous or Cartesian coordinates).
    - img_pts: 2D coordinates of observed image points.
    - trans_mat: Transformation matrix (4x4) that includes rotation and translation from object to camera coordinates.
    - K: Camera intrinsic matrix containing focal lengths and principal point.
    - homog: Integer flag indicating whether obj_pts are in homogeneous coordinates (1) or not (0).
    
    Returns:
    - Tuple containing:
      * The average reprojection error (mean Euclidean distance) between projected and observed points.
      * The possibly converted object points in Cartesian coordinates (Nx3).
    '''
    # Extract the rotation matrix (3x3) from the top-left of the transformation matrix
    R = trans_mat[:3, :3]
    
    # Extract the translation vector (3x1) from the last column of the transformation matrix
    t = trans_mat[:3, 3]
    
    # Convert the rotation matrix to a rotation vector using Rodrigues formula (required by OpenCV)
    rot_vec, _ = cv2.Rodrigues(R)
    
    # If the object points are given in homogeneous coordinates (x, y, z, w),
    # convert them to Cartesian coordinates (x/w, y/w, z/w)
    if homog == 1:
        obj_pts = cv2.convertPointsFromHomogeneous(obj_pts.T)
    
    # Project the 3D object points into the 2D image plane using the camera parameters
    # rot_vec: rotation vector, t: translation vector, K: camera intrinsic matrix
    proj_pts, _ = cv2.projectPoints(obj_pts, rot_vec, t, K, None)
    
    # Reshape projected points to Nx2 float32 array for error calculation
    proj_pts = np.float32(proj_pts[:, 0, :])
    
    # Calculate the L2 (Euclidean) norm between the projected points and the observed image points.
    # If input image points are homogeneous, transpose accordingly.
    img_pts_float = np.float32(img_pts.T) if homog == 1 else np.float32(img_pts)
    err = cv2.norm(proj_pts, img_pts_float, cv2.NORM_L2)
    
    # Compute average reprojection error by dividing the total error by the number of points
    avg_error = err / len(proj_pts)
    
    # Return average error and the object points (now in Cartesian coordinates if converted)
    return avg_error, obj_pts
