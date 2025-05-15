import numpy as np
def create_camera_frustum(position, rotation, scale=15):
    """
    Create a camera frustum (pyramid) at the given position and orientation.
    """
    # Frustum in local camera coordinates
    frustum = np.array([
        [0, 0, 0],     # camera center
        [1, 1, 1.5],   # top right
        [-1, 1, 1.5],  # top left
        [-1, -1, 1.5], # bottom left
        [1, -1, 1.5]   # bottom right
    ]) * scale

    # Rotate and translate
    transformed = (rotation @ frustum.T).T + position
    return transformed


def add_colmap_style_cameras(camera_positions, camera_orientations):
    """
    Generate COLMAP-style frustum geometry for visualization.
    Returns vertices and colors.
    """
    frustum_vertices = []
    frustum_colors = []

    for pos, rot in zip(camera_positions, camera_orientations):
        cam = create_camera_frustum(pos * 200, rot)
        frustum_vertices.append(cam[0])  # center
        frustum_vertices.extend(cam[1:])

        # Frustum lines in red
        frustum_colors.extend([[255, 0, 0]] * 5)

    return np.array(frustum_vertices), np.array(frustum_colors)
