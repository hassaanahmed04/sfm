# sfm_pipeline/feature_matcher.py
import cv2
import numpy as np

def find_features(img1, img2) -> tuple:
    gray1, gray2 = _convert_to_grayscale(img1, img2)
    kp1, desc1, kp2, desc2 = _detect_features(gray1, gray2)
    return _match_features(kp1, desc1, kp2, desc2)

def _convert_to_grayscale(img1: np.ndarray, img2: np.ndarray) -> tuple:
    return cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY), cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

def _detect_features(gray1: np.ndarray, gray2: np.ndarray) -> tuple:
    sift = cv2.xfeatures2d.SIFT_create()
    kp1, desc1 = sift.detectAndCompute(gray1, None)
    kp2, desc2 = sift.detectAndCompute(gray2, None)
    return kp1, desc1, kp2, desc2

def _match_features(kp1, desc1, kp2, desc2) -> tuple:
    matcher = cv2.BFMatcher()
    matches = matcher.knnMatch(desc1, desc2, k=2)
    good_matches = [m for m, n in matches if m.distance < 0.45 * n.distance]
    pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])
    return pts1, pts2

def find_common_pts(pts1, pts2, pts3) -> tuple:
    common1, common2 = _find_common_indices(pts1, pts2)
    mask1 = _mask_array(pts2, common2)
    mask2 = _mask_array(pts3, common2)
    return np.array(common1), np.array(common2), mask1, mask2

def _find_common_indices(pts1: np.ndarray, pts2: np.ndarray) -> tuple:
    common1, common2 = [], []
    for i in range(pts1.shape[0]):
        idx = np.where(pts2 == pts1[i, :])
        if idx[0].size != 0:
            common1.append(i)
            common2.append(idx[0][0])
    return common1, common2

def _mask_array(pts: np.ndarray, indices: list) -> np.ndarray:
    masked = np.ma.array(pts, mask=False)
    masked.mask[indices] = True
    compressed = masked.compressed().reshape(-1, 2)
    return compressed