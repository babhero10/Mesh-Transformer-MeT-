import open3d as o3d
import numpy as np
import pyvista as pv
import random
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def get_colormap(value, colormap='hsv'):
    """
    Map normalized value to a colormap.
    
    Args:
    - value (float): Normalized value (between 0 and 1).
    - colormap (str): Name of the colormap to use.
    
    Returns:
    - (R, G, B) tuple with RGB color values.
    """
    cmap = plt.get_cmap(colormap)
    norm = mcolors.Normalize(vmin=0, vmax=1)
    rgba = cmap(norm(value))
    return rgba[:3]  # Return RGB values

def visualize_mesh_open3d(mesh):
    if not mesh.vertex_normals:
        mesh.compute_vertex_normals()
         
    o3d.visualization.draw_geometries([mesh])

def visualize_mesh_pyvista(mesh, face_colors=None):
    # Convert Open3D mesh to NumPy arrays
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)

    # Convert Open3D mesh to PyVista format
    pv_mesh = pv.PolyData(vertices, np.hstack([np.full((len(triangles), 1), 3), triangles]))

    # Add colors to the PyVista mesh
    if face_colors is not None:
        pv_mesh['colors'] = face_colors

    # Visualize the mesh interactively with PyVista
    p = pv.Plotter()
    
    if face_colors is not None:
        p.add_mesh(pv_mesh, scalars='colors', rgb=True, show_edges=True)
    else:
        p.add_mesh(pv_mesh, show_edges=True)

    p.show()
    
def visualize_clustering_feature(mesh, clustering_features):
    num_faces, num_clusters = clustering_features.shape
    
    # Create colors for each face based on the one-hot vectors
    face_colors = np.zeros((num_faces, 3))  # RGB colors for each face
    
    # Assign colors based on the cluster from the one-hot encoding
    for i in range(num_clusters):
        color = [random.random(), random.random(), random.random()]  # Random color for each cluster
        cluster_faces = np.where(clustering_features[:, i] == 1)[0]  # Faces in this cluster
        face_colors[cluster_faces] = color
    
    # Call a visualization function (e.g., PyVista or another tool)
    visualize_mesh_pyvista(mesh, face_colors)
    
def visualize_positional_encoding(mesh, pos_encoding):    
    # Normalize positional encoding for color mapping
    min_val = np.min(pos_encoding)
    max_val = np.max(pos_encoding)
    normalized_encoding = (pos_encoding - min_val) / (max_val - min_val)  # Normalize between 0 and 1
    
    # Flatten the positional encoding to get a single average value per face
    avg_pos_encoding = normalized_encoding.mean(axis=1)
    
    # Apply the colormap for a more noticeable gradient
    face_colors = np.array([get_colormap(v) for v in avg_pos_encoding])

    visualize_mesh_pyvista(mesh, face_colors)