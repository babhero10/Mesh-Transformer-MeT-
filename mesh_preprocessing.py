import numpy as np
from utils import load_mesh, export_mesh, map_labels_to_new_mesh
from visualization import visualize_mesh_pyvista
import open3d as o3d

def normalize_mesh(mesh):
    # Compute the bounding box of the mesh
    bbox = mesh.get_axis_aligned_bounding_box()

    # Get the min and max coordinates of the bounding box
    min_bound = np.array(bbox.min_bound)
    max_bound = np.array(bbox.max_bound)

    # Compute the center and scale of the bounding box
    center = (min_bound + max_bound) / 2
    scale = max(max_bound - min_bound)

    # Create a transformation matrix to translate and scale the mesh
    transformation = np.eye(4)
    transformation[:3, 3] = -center
    transformation[:3, :3] *= 2 / scale

    # Apply the transformation to the mesh
    mesh.transform(transformation)

    return mesh

def iterative_simplify_with_colors(mesh, target_vertex_count, tolerance=50):
    """
    Perform iterative mesh simplification while keeping track of face labels using colors.
    """
    num_vertices = len(mesh.vertices)
    simplified_mesh = mesh
    
    while num_vertices > target_vertex_count:
        # Estimate the target number of triangles
        target_triangles = len(mesh.triangles) * target_vertex_count // num_vertices
        
        # Simplify the mesh
        simplified_mesh = mesh.simplify_quadric_decimation(target_triangles)
        simplified_mesh.remove_duplicated_vertices()
        
        num_vertices = len(simplified_mesh.vertices)
        
        # Check if the number of vertices is within the tolerance range
        if num_vertices <= target_vertex_count + tolerance:
            break
        
        # Update the mesh for the next iteration
        mesh = simplified_mesh

    return simplified_mesh

def preprocess_mesh(mesh, target_max_vertices):
    # Normalize the mesh (assuming you have a normalize function)
    mesh_normalized = normalize_mesh(mesh)
    
    # Simplify the mesh and retrieve updated labels (if labels are provided)
    simplified_mesh = iterative_simplify_with_colors(mesh_normalized, target_max_vertices)
    
    return simplified_mesh

def read_seg_file(seg_file):
        """
        Read the segmentation file and return the labels.

        Args: 
            seg_file (str): Path to the segmentation file.

        Returns:
            np.ndarray: Labels as a numpy array.
        """
        with open(seg_file, 'r') as f:
            labels = [int(line.strip()) for line in f]
        return np.array(labels, dtype=np.int8)
    
    
def visualize_colored_mesh(mesh, labels):
    import matplotlib.pyplot as plt
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
    visualize_mesh_pyvista(mesh, face_colors_rgb)
    
if __name__ == "__main__":
    mesh = load_mesh("dataset/raw_meshes/132.off")
    face_labels = read_seg_file('dataset/seg/132.seg')
    mesh_simplified = preprocess_mesh(mesh, 1200)
    export_mesh(mesh_simplified, "simplified_mesh.obj")
    visualize_colored_mesh(mesh, face_labels)
    face_labels = map_labels_to_new_mesh(mesh, face_labels, mesh_simplified)
    visualize_colored_mesh(mesh_simplified, face_labels)
    