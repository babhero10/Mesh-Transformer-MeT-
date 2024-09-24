import open3d as o3d
import numpy as np
import pyvista as pv
import random
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import trimesh

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

def visualize_mesh_pyvista(mesh, face_colors=None, save=None):
    # Convert Open3D mesh to NumPy arrays
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)

    # Create a Trimesh object
    tm_mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)

    # Add face colors if provided
    if face_colors is not None:
        tm_mesh.visual.face_colors = np.array(face_colors)

    # Save the mesh with face colors
    if save is not None:
        tm_mesh.export(save + '.ply')
        return

    # Convert Trimesh to PyVista
    pv_mesh = pv.PolyData(vertices, np.hstack([np.full((len(triangles), 1), 3), triangles]))

    if face_colors is not None:
        # Assign the face colors for visualization
        pv_mesh.cell_data['colors'] = np.array(face_colors)

    # Visualize the mesh using PyVista
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
    
def visualize_colored_mesh(mesh, labels, save):
        """
        Visualizes the mesh with colors according to the labels.

        Args:
            mesh_file (str): Path to the .off mesh file.
            label_file (str): Path to the label file.
        """
        # Ensure the labels match the number of faces
        if labels.shape[0] != len(mesh.triangles):
            raise ValueError("The number of labels does not match the number of faces in the mesh.")

        # Assign a color to each unique label
        unique_labels = np.unique(labels)
        color_map = plt.get_cmap('jet', len(unique_labels))  # Choose a color map
        
        # Generate colors for each label
        color_dict = {label: color_map(i)[:3] for i, label in enumerate(unique_labels)}  # Ignore alpha channel
        face_colors = np.array([color_dict[label] for label in labels])

        # Convert face colors to the required format for PyVista
        face_colors_rgb = (face_colors * 255).astype(np.uint8)  # Scale to [0, 255] for RGB

        # Visualize the mesh using PyVista
        visualize_mesh_pyvista(mesh, face_colors_rgb, save)