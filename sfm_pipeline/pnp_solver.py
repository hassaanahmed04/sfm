# sfm_pipeline/pnp_solver.py

import cv2
import numpy as np

def solve_pnp_prob(obj_pts, img_pts, K, rot_vec, init_flag) -> tuple:
    processed_obj_pts, processed_img_pts, processed_rot_vec = _prepare_inputs(obj_pts, img_pts, rot_vec, init_flag)
    return _compute_pnp(processed_obj_pts, processed_img_pts, K, processed_rot_vec)

def _prepare_inputs(obj_pts: np.ndarray, img_pts: np.ndarray, rot_vec: np.ndarray, init_flag: int) -> tuple:
    if init_flag == 1:
        obj_pts = obj_pts[:, 0, :]
        img_pts = img_pts.T
        rot_vec = rot_vec.T
    return obj_pts, img_pts, rot_vec

def _compute_pnp(obj_pts: np.ndarray, img_pts: np.ndarray, K: np.ndarray, rot_vec: np.ndarray) -> tuple:
    _, rot_vec_calc, t, inliers = cv2.solvePnPRansac(obj_pts, img_pts, K, cv2.SOLVEPNP_ITERATIVE)
    R, _ = cv2.Rodrigues(rot_vec_calc)

    if inliers is not None:
        img_pts = img_pts[inliers[:, 0]]
        obj_pts = obj_pts[inliers[:, 0]]
        rot_vec = rot_vec[inliers[:, 0]]

    return R, t, img_pts, obj_pts, rot_vec
