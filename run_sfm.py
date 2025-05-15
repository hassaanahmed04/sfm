# run_sfm.py
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import open3d as o3d
import requests
import argparse
from sfm_pipeline.image_loader import ImgLoader
from sfm_pipeline.triangulation import triangulate_pts
from sfm_pipeline.pnp_solver import solve_pnp_prob
from sfm_pipeline.reprojection import calc_reproj_err
from sfm_pipeline.bundle_adjust import bundle_adjust
from sfm_pipeline.feature_matcher import find_features, find_common_pts
from sfm_pipeline.ply_generator import create_ply, create_ply_with_cameras
from sfm_pipeline.point_utils import *


class SFMVisualizer:
    def __init__(self):
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name='3D Reconstruction', width=800, height=600)
        self.point_cloud = o3d.geometry.PointCloud()
        self.coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.6)
        self.vis.add_geometry(self.coordinate_frame)
        self.vis.add_geometry(self.point_cloud)
        self.initialized_visualization = False
    
    def update_visualization(self, points, colors):
        self.point_cloud.points = o3d.utility.Vector3dVector(points)
        self.point_cloud.colors = o3d.utility.Vector3dVector(colors / 255.0)
        
        if not self.initialized_visualization:
            self.vis.reset_view_point(True)
            self.initialized_visualization = True
        
        self.vis.update_geometry(self.point_cloud)
        self.vis.poll_events()
        self.vis.update_renderer()
    
    def run_visualization(self):
        self.vis.run()
        self.vis.destroy_window()


class BundleAdjustmentHandler:
    @staticmethod
    def bundle_adjust_ceres(observations, camera_params, points_3d, intrinsics):
        url = "http://localhost:8080/bundle_adjust"
        data = {
            "observations": observations,
            "camera_params": camera_params,
            "points_3d": points_3d,
            "intrinsics": intrinsics
        }
        response = requests.post(url, json=data)
        return response.json()


class SFMPipeline:
    def __init__(self, img_path: str, scale_factor: float = 2.0):
        self.img_loader = ImgLoader(img_path, scale_factor)
        self.visualizer = SFMVisualizer()
        self.error_list = []
        self.threshold = 0.5

    def initialize_camera_poses(self):
        poses = self.img_loader.K.ravel()
        trans_mat_0 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
        trans_mat_1 = np.empty((3, 4))
        
        cam_pose_0 = np.matmul(self.img_loader.K, trans_mat_0)
        cam_pose_1 = np.empty((3, 4)) 
        
        return poses, trans_mat_0, trans_mat_1, cam_pose_0, cam_pose_1

    def process_first_two_frames(self):
        img_0 = self.img_loader.scale_down(cv2.imread(self.img_loader.img_list[0]))
        img_1 = self.img_loader.scale_down(cv2.imread(self.img_loader.img_list[1]))
        ft_0, ft_1 = find_features(img_0, img_1)

        E_mat, mask = cv2.findEssentialMat(ft_0, ft_1, self.img_loader.K)
        ft_0 = ft_0[mask.ravel() == 1]
        ft_1 = ft_1[mask.ravel() == 1]

        _, R, t, mask = cv2.recoverPose(E_mat, ft_0, ft_1, self.img_loader.K)
        ft_0 = ft_0[mask.ravel() > 0]
        ft_1 = ft_1[mask.ravel() > 0]
        
        return img_0, img_1, ft_0, ft_1, R, t

    def triangulate_initial_points(self, cam_pose_0, cam_pose_1, ft_0, ft_1):
        ft_0, ft_1, pts_3d = triangulate_pts(cam_pose_0, cam_pose_1, ft_0, ft_1)
        err, pts_3d = calc_reproj_err(pts_3d, ft_1, self.trans_mat_1, self.img_loader.K, homog=1)
        _, _, ft_1, pts_3d, _ = solve_pnp_prob(pts_3d, ft_1, self.img_loader.K, ft_0, init_flag=1)
        return ft_0, ft_1, pts_3d

    def process_frame(self, img_2, i):
        ft_cur, ft_2 = find_features(self.img_1, img_2)

        if i != 0:
            self.ft_0, self.ft_1, self.pts_3d = triangulate_pts(self.cam_pose_0, self.cam_pose_1, self.ft_0, self.ft_1)
            self.ft_1 = self.ft_1.T
            self.pts_3d = cv2.convertPointsFromHomogeneous(self.pts_3d.T)
            self.pts_3d = self.pts_3d[:, 0, :]

        cm_pts_0, cm_pts_1, cm_mask_0, cm_mask_1 = find_common_pts(self.ft_1, ft_cur, ft_2)
        cm_pts_2 = ft_2[cm_pts_1]
        cm_pts_cur = ft_cur[cm_pts_1]

        R, t, cm_pts_2, self.pts_3d, cm_pts_cur = solve_pnp_prob(
            self.pts_3d[cm_pts_0], cm_pts_2, self.img_loader.K, cm_pts_cur, init_flag=0
        )
        self.trans_mat_1 = np.hstack((R, t))
        cam_pose_2 = np.matmul(self.img_loader.K, self.trans_mat_1)
        self.camera_positions.append(self.trans_mat_1[:3, 3])
        self.camera_orientations.append(self.trans_mat_1[:3, :3])
        err, self.pts_3d = calc_reproj_err(self.pts_3d, cm_pts_2, self.trans_mat_1, self.img_loader.K, homog=0)
        cm_mask_0, cm_mask_1, self.pts_3d = triangulate_pts(self.cam_pose_1, cam_pose_2, cm_mask_0, cm_mask_1)
        err, self.pts_3d = calc_reproj_err(self.pts_3d, cm_mask_1, self.trans_mat_1, self.img_loader.K, homog=1)
        return img_2, ft_cur, ft_2, cam_pose_2, err, cm_mask_1

    def handle_bundle_adjustment(self, method, img_2, cm_mask_1):
        if method == "bundle_adjust":
            self.pts_3d, cm_mask_1, self.trans_mat_1 = bundle_adjust(
                self.pts_3d, cm_mask_1, self.trans_mat_1, self.img_loader.K, self.threshold
            )
            cam_pose_2 = np.matmul(self.img_loader.K, self.trans_mat_1)
            err, self.pts_3d = calc_reproj_err(
                self.pts_3d, cm_mask_1, self.trans_mat_1, self.img_loader.K, homog=0
            )
            self.update_point_cloud(img_2, cm_mask_1)
            
        elif method == "ceres":
            try:
                if len(cm_mask_1) == 0 or len(self.pts_3d) == 0:
                    observations = cm_mask_1[:, :num_points].reshape(-1).tolist()    
                    raise ValueError("No points to optimize")
                
                if self.pts_3d.ndim == 3:
                    self.pts_3d = self.pts_3d[:, 0, :]
                
                num_points = min(len(cm_mask_1), len(self.pts_3d))
                if num_points < 2:
                    raise ValueError(f"Not enough points ({num_points}) for optimization")
                
                observations = cm_mask_1[:, :num_points].reshape(-1).tolist()
                points_3d = self.pts_3d[:num_points].reshape(-1).tolist()
                camera_params = self.trans_mat_1.reshape(-1).tolist()
                intrinsics = [
                    float(self.img_loader.K[0, 0]),
                    float(self.img_loader.K[1, 1]),
                    float(self.img_loader.K[0, 2]),
                    float(self.img_loader.K[1, 2])
                ]
                
                result = BundleAdjustmentHandler.bundle_adjust_ceres(
                    observations, camera_params, points_3d, intrinsics
                )
                
                self.trans_mat_1 = np.array(result["optimized_camera"]).reshape(3, 4)
                self.pts_3d[:num_points] = np.array(result["optimized_points"]).reshape(-1, 3)
                cam_pose_2 = np.matmul(self.img_loader.K, self.trans_mat_1)
                
                err, _ = calc_reproj_err(
                    self.pts_3d[:num_points],
                    cm_mask_1[:, :num_points].T,
                    self.trans_mat_1,
                    self.img_loader.K,
                    homog=0
                )
                
                self.update_point_cloud(img_2, cm_mask_1[:num_points])
                
            except Exception as e:
                print(f"Error in Ceres optimization: {str(e)}")
                self.update_point_cloud(img_2, cm_mask_1)
        else:
            self.update_point_cloud(img_2, cm_mask_1.T)

    def update_point_cloud(self, img, points):
        if self.pts_3d.ndim == 3:
            pts_to_add = self.pts_3d[:, 0, :]
        else:
            pts_to_add = self.pts_3d

        self.all_pts = np.vstack((self.all_pts, pts_to_add))
        
        left_pts = np.array(points, dtype=np.int32)
        colors = np.array([img[p[1], p[0]] for p in left_pts])
        self.all_colors = np.vstack((self.all_colors, colors))

    def update_error_plot(self, err):
        self.error_list.append(err)
        plt.clf()
        plt.plot(range(len(self.error_list)), self.error_list, 'b-')
        plt.xlabel('Frame')
        plt.ylabel('Reprojection Error')
        plt.title('Reprojection Error Over Frames')
        plt.grid(True)
        plt.draw()
        plt.pause(0.01)

    def run(self, method="none"):
        cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        
        # Initialize data structures
        self.poses, self.trans_mat_0, self.trans_mat_1, self.cam_pose_0, self.cam_pose_1 = self.initialize_camera_poses()
        self.all_pts = np.zeros((1, 3))
        self.all_colors = np.zeros((1, 3))
        self.camera_positions = []
        self.camera_orientations = []

        # Process first two frames
        self.img_0, self.img_1, self.ft_0, self.ft_1, R, t = self.process_first_two_frames()
        
        self.trans_mat_1[:3, :3] = np.matmul(R, self.trans_mat_0[:3, :3])
        self.trans_mat_1[:3, 3] = self.trans_mat_0[:3, 3] + np.matmul(self.trans_mat_0[:3, :3], t.ravel())
        self.cam_pose_1 = np.matmul(self.img_loader.K, self.trans_mat_1)
        
        # Store first two camera poses
        self.camera_positions.append(self.trans_mat_0[:3, 3])
        self.camera_orientations.append(self.trans_mat_0[:3, :3])
        self.camera_positions.append(self.trans_mat_1[:3, 3])
        self.camera_orientations.append(self.trans_mat_1[:3, :3])

        # Triangulate initial points
        self.ft_0, self.ft_1, self.pts_3d = self.triangulate_initial_points(
            self.cam_pose_0, self.cam_pose_1, self.ft_0, self.ft_1
        )

        self.poses = np.hstack((np.hstack((self.poses, self.cam_pose_0.ravel())), self.cam_pose_1.ravel()))
        
        # Initialize matplotlib figure for error plot
        plt.figure(figsize=(10, 5))
        plt.ion()

        total_imgs = len(self.img_loader.img_list) - 2
        
        for i in tqdm(range(total_imgs)):
            img_2 = self.img_loader.scale_down(cv2.imread(self.img_loader.img_list[i + 2]))
            img_2, ft_cur, ft_2, cam_pose_2, err, cm_mask_1 = self.process_frame(img_2, i)
            
            # Handle bundle adjustment based on method
            self.handle_bundle_adjustment(method, img_2, cm_mask_1)
            
            # Update visualization periodically
            if i % 2 == 0:
                self.visualizer.update_visualization(self.all_pts, self.all_colors)
            
            # Update error plot
            self.update_error_plot(err)

            # Show current image
            cv2.imshow('image', img_2)
            if cv2.waitKey(1) & 0xff == ord('q'):
                break

            # Update variables for next iteration
            self.trans_mat_0 = np.copy(self.trans_mat_1)
            self.cam_pose_0 = np.copy(self.cam_pose_1)
            self.img_0 = np.copy(self.img_1)
            self.img_1 = np.copy(img_2)
            self.ft_0 = np.copy(ft_cur)
            self.ft_1 = np.copy(ft_2)
            self.cam_pose_1 = np.copy(cam_pose_2)

        cv2.destroyAllWindows()
        plt.ioff()
        
        # Final processing
        print("Creating .ply file")
        create_ply_with_cameras(
            self.img_loader.path, 
            self.all_pts, 
            self.all_colors, 
            self.camera_positions, 
            self.camera_orientations, 
            self.img_loader.img_list
        )
        create_ply(self.img_loader.path, self.all_pts, self.all_colors, self.img_loader.img_list)
    
        print("Process completed - Close the 3D visualization window to exit")
        self.visualizer.run_visualization()


def main():
    parser = argparse.ArgumentParser(description="Structure from Motion Pipeline")
    parser.add_argument(
        "--method", 
        choices=["none", "bundle_adjust", "ceres"], 
        default="none",
        help="Optimization method to use (none, bundle_adjust, or ceres)"
    )
    parser.add_argument(
        "--dataset", 
        default="Datasets/GustavIIAdolf",
        help="Path to the image dataset"
    )
    args = parser.parse_args()

    sfm = SFMPipeline(args.dataset)
    sfm.run(method=args.method)


if __name__ == '__main__':
    main()