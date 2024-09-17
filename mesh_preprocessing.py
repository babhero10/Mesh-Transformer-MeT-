import numpy as np
from utils import load_mesh, export_mesh
from visualization import visualize_mesh_pyvista

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

def QEM_simplify(mesh, num_faces):
    simplified_mesh = mesh.simplify_quadric_decimation(num_faces)
    simplified_mesh.remove_duplicated_vertices() # Merge duplicated vertices
    return simplified_mesh

if __name__ == "__main__":
    mesh = load_mesh("7.off")
    mesh_normalize = normalize_mesh(mesh)
    mesh_simplified = QEM_simplify(mesh_normalize, 2412)
    export_mesh(mesh_simplified, "simplified_mesh.obj")
    visualize_mesh_pyvista(mesh_simplified)