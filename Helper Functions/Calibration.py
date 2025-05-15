# import cv2
# import numpy as np
# import glob

# def save_camera_matrix(K, filename="camera_matrix.txt"):
#     with open(filename, 'w') as f:
#         for row in K:
#             f.write(' '.join([f"{val:.2f}" for val in row]) + '\n')
#     print(f"Camera matrix saved to {filename}")

# def calibrate_camera_from_chessboard(image_folder, chessboard_size=(8, 10), square_size=15.0):
#     # Prepare 3D object points based on real square size (15 mm)
#     objp = np.zeros((chessboard_size[1] * chessboard_size[0], 3), np.float32)
#     objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
#     objp *= square_size  # Now in mm

#     objpoints = []  # 3D points in real world space
#     imgpoints = []  # 2D points in image plane

#     images = glob.glob(f'{image_folder}/*.jpg')  # Change if using PNG or HEIC

#     for fname in images:
#         img = cv2.imread(fname)
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#         # Find the chessboard corners
#         ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

#         if ret:
#             objpoints.append(objp)
#             corners2 = cv2.cornerSubPix(
#                 gray, corners, (11, 11), (-1, -1),
#                 criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
#             )
#             imgpoints.append(corners2)

#     # Run calibration
#     ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
#         objpoints, imgpoints, gray.shape[::-1], None, None
#     )

#     # Display and save the intrinsic matrix
#     print("Camera matrix (K):\n", mtx)
#     save_camera_matrix(mtx)

#     return mtx, dist, rvecs, tvecs

# # 🧪 Example usage:
# # Place your checkerboard images inside "calibration_images"
# camera_matrix, dist_coeffs, rvecs, tvecs = calibrate_camera_from_chessboard(
#     image_folder="Datasets/calibration",      # 🔁 Replace with your folder
#     chessboard_size=(7, 9),                # 🟩 Inner corners: cols x rows
#     square_size=15.0                        # 📏 mm (from calib.io)
# )
import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
# Chessboard size (change this based on your checkerboard pattern)
chessboard_size = (7,9)  # Number of internal corners in the chessboard (width, height)
square_size = 15
  # Size of a square in some unit (e.g., mm, cm)

# Prepare object points (3D points in real-world space)
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
objp *= square_size  # Scale by the square size

# Arrays to store object points and image points
objpoints = []  # 3D points in real-world coordinates
imgpoints = []  # 2D points in image plane

# Load calibration images
# image_paths = glob.glob('camera pic/images_left_stereo/*.png')  # Select first 6 images
# import glob

# Get all PNG and JPG images from the directory
image_paths = glob.glob('Datasets/calibration/*.png') + glob.glob('Datasets/calibration/*.jpg')


for fname in image_paths:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # plt.imshow(gray)

    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

    if ret:
        objpoints.append(objp)  # Store 3D points
        imgpoints.append(corners)  # Store 2D points

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
     

cv2.destroyAllWindows()

# Perform camera calibration
ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# # Compute reprojection error
reprojection_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, dist)
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
    reprojection_error += error

reprojection_error /= len(objpoints)

# Display results
print("Camera Matrix1 (K):\n", K)
print("Distortion Coefficients1:\n", dist.ravel())
print("Reprojection Error1:", reprojection_error)