# sfm_pipeline/ply_generator.py
import numpy as np
import os
from typing import  Optional, List
import open3d as o3d

def create_ply(path: str, points: np.ndarray, 
                     color_data: np.ndarray, image_paths: List[str]) -> None:
    """
    Creates a PLY file from 3D point cloud data with associated colors.
    
    Args:
        path: Directory to save the PLY file
        points: 3D point coordinates (Nx3 array)
        color_data: RGB color values for each point (Nx3 array)
        image_paths: List of image paths used for naming the output file
    """
    # Reshape and scale point data
    point_cloud = points.reshape(-1, 3) * 200
    rgb_values = color_data.reshape(-1, 3)
    
    # Combine coordinates and colors
    vertex_data = np.column_stack([point_cloud, rgb_values])
    
    # Filter distant points
    centroid = np.mean(vertex_data[:, :3], axis=0)
    centered = vertex_data[:, :3] - centroid
    distances = np.linalg.norm(centered, axis=1)
    valid_indices = distances < np.mean(distances) + 300
    filtered_vertices = vertex_data[valid_indices]
    
    # Create PLY header
    ply_header = f"""ply
format ascii 1.0
element vertex {len(filtered_vertices)}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
"""
    
    # Determine output filename from first image path
    folder_name = image_paths[0].split('/')[-2]
    output_path = os.path.join(path, 'results', f'{folder_name}.ply')
    
    # Write to file
    with open(output_path, 'w') as output_file:
        output_file.write(ply_header)
        np.savetxt(output_file, filtered_vertices, 
                  fmt='%.6f %.6f %.6f %d %d %d')
def create_ply_with_cameras(path: str, points: np.ndarray, 
                                color_data: np.ndarray, 
                                cam_positions: List[np.ndarray],
                                cam_orientations: List[np.ndarray],
                                image_paths: List[str],
                                target_position: Optional[np.ndarray] = None) -> None:
    """
    Creates a PLY file with point cloud, camera positions, orientations, and optional target lines.
    
    Args:
        path: Output directory path
        points: 3D point coordinates
        color_data: RGB color values for points
        cam_positions: List of camera positions
        cam_orientations: List of camera orientation matrices
        image_paths: List of source image paths
        target_position: Optional 3D position to draw lines from cameras
    """
    # Process point cloud data
    scaled_points = points.reshape(-1, 3) * 200
    point_colors = color_data.reshape(-1, 3)
    
    # Process camera data
    scaled_cam_positions = np.array(cam_positions) * 200
    camera_markers = np.tile([255, 0, 0], (len(scaled_cam_positions), 1))
    
    # Create camera direction indicators
    direction_segments = []
    direction_colors = []
    for pos, rot in zip(scaled_cam_positions, cam_orientations):
        view_direction = rot[:, 2] * 50
        segment_end = pos + view_direction
        direction_segments.extend([pos, segment_end])
        direction_colors.extend([[0, 255, 0], [0, 255, 0]])
    
    # Create target lines if specified
    target_lines = []
    if target_position is not None:
        scaled_target = np.array(target_position) * 200
        for cam_pos in scaled_cam_positions:
            target_lines.extend([cam_pos, scaled_target])
            direction_colors.extend([[255, 255, 0], [255, 255, 0]])
    
    # Combine all geometric elements
    all_vertices = [scaled_points, scaled_cam_positions, np.array(direction_segments)]
    if target_position:
        all_vertices.append(np.array(target_lines))
    
    combined_vertices = np.vstack(all_vertices)
    combined_colors = np.vstack([point_colors, camera_markers, np.array(direction_colors)])
    
    # Filter distant points
    cloud_center = np.mean(combined_vertices[:, :3], axis=0)
    centered_cloud = combined_vertices[:, :3] - cloud_center
    point_distances = np.linalg.norm(centered_cloud, axis=1)
    close_points = point_distances < np.mean(point_distances) + 300
    
    # Create pyramid-shaped camera indicators
    pyramid_vertices = []
    pyramid_lines = []
    pyramid_size = 30
    pyramid_depth = 40
    
    for pos, rot in zip(scaled_cam_positions, cam_orientations):
        # Camera's view direction (z-axis of rotation matrix)
        view_dir = rot[:, 2]
        
        # Base center is slightly in front of camera position
        base_center = pos + view_dir * pyramid_depth
        
        # Calculate base vectors (using camera's x and y axes)
        right = rot[:, 0] * pyramid_size
        up = rot[:, 1] * pyramid_size
        
        # Base vertices
        base1 = base_center + right + up    # top-right
        base2 = base_center + right - up    # bottom-right
        base3 = base_center - right - up    # bottom-left
        base4 = base_center - right + up    # top-left
        
        # Add pyramid vertices (camera position is the apex)
        pyramid_vertices.extend([pos, base1, base2, base3, base4])
        
        # Add lines for pyramid edges (4 base edges + 4 lines to apex)
        pyramid_lines.extend([pos, base1])  # apex to base1
        pyramid_lines.extend([pos, base2])  # apex to base2
        pyramid_lines.extend([pos, base3])  # apex to base3
        pyramid_lines.extend([pos, base4])  # apex to base4
        pyramid_lines.extend([base1, base2])  # base edges
        pyramid_lines.extend([base2, base3])
        pyramid_lines.extend([base3, base4])
        pyramid_lines.extend([base4, base1])
    
    # Final vertex assembly
    if pyramid_vertices:
        pyramid_vertex_array = np.array(pyramid_vertices)
        pyramid_line_array = np.array(pyramid_lines)
        pyramid_vertex_colors = np.tile([255, 0, 0], (len(pyramid_vertices), 1))
        pyramid_line_colors = np.tile([255, 165, 0], (len(pyramid_lines), 1))  # Orange color for pyramid lines
        
        final_vertices = np.vstack([
            combined_vertices[close_points],
            pyramid_vertex_array,
            pyramid_line_array
        ])
        final_colors = np.vstack([
            combined_colors[close_points],
            pyramid_vertex_colors,
            pyramid_line_colors
        ])
    else:
        final_vertices = combined_vertices[close_points]
        final_colors = combined_colors[close_points]
    
    # Prepare PLY data
    ply_data = np.column_stack([final_vertices, final_colors])
    
    # Create PLY header
    ply_header = f"""ply
format ascii 1.0
element vertex {len(ply_data)}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
"""
    
    # Determine output filename
    folder_name = image_paths[0].split('/')[-2]
    output_path = os.path.join(path, 'results', f'{folder_name}_cams.ply')
    
    # Write to file
    with open(output_path, 'w') as output_file:
        output_file.write(ply_header)
        np.savetxt(output_file, ply_data, fmt='%.6f %.6f %.6f %d %d %d')


