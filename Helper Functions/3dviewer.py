# import open3d as o3d

# ply_path = "results/GustavIIAdolf_cams.ply"  
# pcd = o3d.io.read_point_cloud(ply_path)

# print(pcd)
# print("Number of points:", len(pcd.points))

# o3d.visualization.draw_geometries([pcd])
import open3d as o3d
import numpy as np

def visualize_ply_with_cameras(ply_path, background_color=(0, 0, 0)):
    # Load the PLY file
    mesh = o3d.io.read_triangle_mesh(ply_path)
    
    if not mesh.has_vertices():
        raise ValueError("PLY file contains no vertices")
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = mesh.vertices
    
    if mesh.has_vertex_colors():
        pcd.colors = mesh.vertex_colors
    
    geometries = [pcd]
    
    if mesh.has_triangles():
        mesh.compute_vertex_normals()
        geometries.append(mesh)
    
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=100.0)
    geometries.append(coord_frame)
    
    # Visualization with custom background
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    opt = vis.get_render_option()
    opt.background_color = np.asarray(background_color)
    
    for geom in geometries:
        vis.add_geometry(geom)
    
    vis.run()
    vis.destroy_window()

# Usage examples:
# Black background (default)
visualize_ply_with_cameras("results/GustavIIAdolf_cams.ply", background_color=(0, 0, 0))  

# White background
# visualize_ply_with_cameras("results/GustavIIAdolf.ply", background_color=(1, 1, 1))
